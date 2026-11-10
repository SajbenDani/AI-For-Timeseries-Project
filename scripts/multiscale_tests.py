"""Idea 11 (ideas.md): multi-scale differences ("wavelet-like" views).

  python scripts/multiscale_tests.py main      # QUANT + idea 11 on all 142 datasets (first 10 resamples)
  python scripts/multiscale_tests.py summary   # -> results/multiscale/summary.txt, compared with plain QUANT

QUANT sees change either step by step (its differences) or at all speeds but without time (its Fourier view). This
adds change at medium speeds, located in time: for each scale s in 2, 4 and 8, the average of the next s points
minus the average of the previous s points, at every position (an undecimated Haar wavelet). Each scale is one more
view, described by QUANT's own windows and quantiles. Following ideas.md's warning, only 3 scales are used, since
each adds about as many features as the series is long; a scale is skipped when the series is too short for it.

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

OUT = ROOT / "results" / "multiscale"
METHOD = "QUANT + multi-scale differences (idea 11)"
SCALES = (2, 4, 8)


# == idea 11 ==========================================================================================


def haar(X, s):
    """Average of the next s points minus average of the previous s points, at every position.
    X: (n, length) -> (n, length - 2s + 1)."""
    c = torch.nn.functional.pad(X.double().cumsum(-1), (1, 0))
    mean = (
        c[:, s:] - c[:, :-s]
    ) / s  # moving average of width s, length - s + 1 values
    return (mean[:, s:] - mean[:, :-s]).float()


def multiscale_view(X, depth=6, div=4, min_length=8):
    """QUANT's windows and quantiles on the multi-scale differences. X: (n_series, length) -> features."""
    scales = [s for s in SCALES if X.shape[-1] - 2 * s + 1 >= min_length]

    def view(X):
        out = []
        for s in scales:
            D = haar(X, s)
            out.append(
                IntervalModel(D.shape[-1], depth, div).fit_transform(
                    D.unsqueeze(1), None
                )
            )
        return torch.cat(out, -1)

    return batched(view, X, 512)


# == test =============================================================================================


def test_main(args):
    out = OUT / "multiscale.csv"
    done = finished(out)
    for name in args.datasets or NAMES:
        if name in done:
            continue
        try:
            t_start = time.perf_counter()
            d = Dataset(name)
            M, t = timed(multiscale_view, d.X)
            F = np.hstack([quant_features(d.X), M])
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
    new = pd.read_csv(OUT / "multiscale.csv")
    base = pd.read_csv(ROOT / "results" / "efficient" / "efficient.csv")
    base = base[base["method"] == "QUANT"][["dataset", "resample", "accuracy"]]
    both = new.merge(base, on=["dataset", "resample"], suffixes=("", "_quant"))
    per = both.groupby("dataset")[["accuracy", "accuracy_quant"]].mean()
    d = np.round(per["accuracy"] - per["accuracy_quant"], 4)
    p = (
        wilcoxon(per["accuracy"], per["accuracy_quant"], zero_method="pratt").pvalue
        if (d != 0).any()
        else 1.0
    )
    lines = [
        f"Idea 11: {METHOD}",
        f"{len(per)} datasets x {both['resample'].nunique()} resamples, same splits and seeds",
        "",
        f"QUANT           {100 * per['accuracy_quant'].mean():.2f}%",
        f"QUANT + idea 11 {100 * per['accuracy'].mean():.2f}%",
        f"change          {100 * d.mean():+.2f} pts   W/D/L {(d > 0).sum()}/{(d == 0).sum()}/{(d < 0).sum()}   "
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
