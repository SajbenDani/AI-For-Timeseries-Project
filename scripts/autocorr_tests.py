"""Idea 7 (ideas.md): phase-invariant views, i.e. the autocorrelation function and autoregressive (AR) coefficients.

  python scripts/autocorr_tests.py main      # QUANT + idea 7 on all 142 datasets (first 10 resamples)
  python scripts/autocorr_tests.py summary   # -> results/autocorr/summary.txt, compared with plain QUANT

Same rules as improvement_tests.py and efficient_tests.py: QUANT itself is never changed, the forest is the paper's,
every run uses the official resamples and seed = resample index, and finished datasets are skipped when the test is
re-run. Plain QUANT is not re-run: with the same splits and seeds its accuracies are those in
results/efficient/efficient.csv. A dataset that fails is reported and skipped, so one error never stops the run.
"""

import argparse
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from quant_extensions import batched  # also puts codebase/quant/code on the path
from quant import IntervalModel  # noqa: E402
from weakness_tests import NAMES, ROOT, Dataset, append, finished, quant_features
from improvement_tests import evaluate, timed

warnings.filterwarnings("ignore")

OUT = ROOT / "results" / "autocorr"
METHOD = "QUANT + autocorrelation view + AR (idea 7)"


# == idea 7 ===========================================================================================


def levinson(r, order):
    """AR coefficients from autocorrelations r (rows: series; r[:, 0] = 1) by the Levinson-Durbin recursion,
    vectorised over series. Reflection coefficients are kept inside (-1, 1) so flat series stay finite.
    """
    a = torch.zeros(r.shape[0], order, dtype=r.dtype)
    e = r[:, 0].clone()
    for k in range(order):
        acc = r[:, k + 1] - (a[:, :k] * r[:, 1 : k + 1].flip(1)).sum(1)
        kappa = (acc / e.clamp_min(1e-12)).clamp(-0.999, 0.999)
        previous = a[:, :k].clone()
        a[:, :k] = previous - kappa[:, None] * previous.flip(1)
        a[:, k] = kappa
        e = e * (1 - kappa**2)
    return a


def autocorr_view(X, depth=6, div=4, max_order=10):
    """Two views that ignore *when* things happen: the autocorrelation function (how similar the series is to
    itself shifted by 1, 2, ... steps, up to half its length) summarised by QUANT's own windows and quantiles, and
    the AR coefficients of order min(10, length / 4). X: (n_series, length) -> (n_series, about length) features.
    """
    length = X.shape[-1]
    lags = min(length - 1, max(4, length // 2))
    order = max(1, min(max_order, length // 4))
    size = 1 << int(np.ceil(np.log2(2 * length)))

    def view(X):
        Xc = (X - X.mean(-1, keepdim=True)).double()
        acf = torch.fft.irfft(torch.fft.rfft(Xc, size).abs() ** 2, size)[:, :length]
        acf = acf / acf[:, :1].clamp_min(1e-12)
        quantiles = IntervalModel(lags, depth, div).fit_transform(
            acf[:, 1 : lags + 1].float().unsqueeze(1), None
        )
        return torch.cat([quantiles, levinson(acf[:, : order + 1], order).float()], -1)

    return batched(view, X, 512)


# == test =============================================================================================


def test_main(args):
    out = OUT / "autocorr.csv"
    done = finished(out)
    for name in args.datasets or NAMES:
        if name in done:
            continue
        try:
            t_start = time.perf_counter()
            d = Dataset(name)
            A, t = timed(autocorr_view, d.X)
            F = np.hstack([quant_features(d.X), A])
            rows = []
            for i in range(args.resamples or 10):
                train, test = d.split(i)
                rows.append(
                    {
                        "dataset": name,
                        "resample": i,
                        **evaluate(METHOD, F, d.y, train, test, i),
                        "transform_seconds": t,
                    }
                )
            append(rows, out)
            print(
                f"main {name}: done in {time.perf_counter() - t_start:.0f} s",
                flush=True,
            )
        except Exception:
            print(f"main {name}: FAILED, skipped\n{traceback.format_exc()}", flush=True)


# == summary ==========================================================================================


def test_summary(args):
    new = pd.read_csv(OUT / "autocorr.csv")
    base = pd.read_csv(ROOT / "results" / "efficient" / "efficient.csv")
    base = base[base.method == "QUANT"][["dataset", "resample", "accuracy"]]
    both = new.merge(base, on=["dataset", "resample"], suffixes=("", "_quant"))
    per = both.groupby("dataset")[["accuracy", "accuracy_quant"]].mean()
    d = np.round(per.accuracy - per.accuracy_quant, 4)
    p = (
        wilcoxon(per.accuracy, per.accuracy_quant, zero_method="pratt").pvalue
        if (d != 0).any()
        else 1.0
    )
    lines = [
        f"Idea 7: {METHOD}",
        f"{len(per)} datasets x {both['resample'].nunique()} resamples, same splits and seeds",
        "",
        f"QUANT        {100 * per.accuracy_quant.mean():.2f}%",
        f"QUANT + idea 7 {100 * per.accuracy.mean():.2f}%",
        f"change       {100 * d.mean():+.2f} pts   W/D/L {(d > 0).sum()}/{(d == 0).sum()}/{(d < 0).sum()}   "
        f"Wilcoxon p={p:.2g}",
        "",
        "Largest gains (pts):",
        (100 * d.sort_values(ascending=False).head(5)).round(2).to_string(),
        "",
        "Largest losses (pts):",
        (100 * d.sort_values().head(5)).round(2).to_string(),
    ]
    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text + "\n")
    print(text)


TESTS = {"main": test_main, "summary": test_summary}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("tests", nargs="+", choices=list(TESTS))
    p.add_argument(
        "--datasets", nargs="*", help="only these datasets (default: all 142)"
    )
    p.add_argument(
        "--resamples", type=int, help="override the number of resamples (default 10)"
    )
    args = p.parse_args()
    for t in args.tests:
        TESTS[t](args)
