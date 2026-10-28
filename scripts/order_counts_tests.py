"""Idea 12 (ideas.md): order-aware counts in each of QUANT's windows.

  python scripts/order_counts_tests.py main      # QUANT + idea 12 on all 142 datasets (first 10 resamples)
  python scripts/order_counts_tests.py summary   # -> results/order_counts/summary.txt, compared with plain QUANT

Quantiles ignore the order of the values inside a window: shuffle a window and QUANT's features stay the same. For
every window QUANT uses on the raw series, this adds four numbers that do depend on order: how often the series
crosses the window mean, how many local peaks it has, its longest run above the mean, and its slope.

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
from quant import make_intervals  # noqa: E402
from weakness_tests import NAMES, ROOT, Dataset, append, finished, quant_features
from improvement_tests import evaluate, timed

warnings.filterwarnings("ignore")

OUT = ROOT / "results" / "order_counts"
METHOD = "QUANT + order-aware counts (idea 12)"


# == idea 12 ==========================================================================================


def window_counts(W):
    """Four order-aware numbers for every row of W (n_series, window_length >= 3)."""
    m = W.shape[1]
    mean = W.mean(1, keepdim=True)
    above = (W > mean).long()
    crossings = (above[:, 1:] != above[:, :-1]).sum(1)
    peaks = ((W[:, 1:-1] > W[:, :-2]) & (W[:, 1:-1] > W[:, 2:])).sum(1)
    count = above.cumsum(
        1
    )  # length of the current run above the mean, at every position
    run = (
        count - torch.where(above == 0, count, torch.zeros_like(count)).cummax(1).values
    )
    t = torch.arange(m, dtype=W.dtype) - (m - 1) / 2
    slope = ((W - mean) * t).sum(1) / (t**2).sum()
    return torch.stack(
        [crossings.float(), peaks.float(), run.max(1).values.float(), slope.float()], 1
    )


def order_counts(X, depth=6, min_length=3):
    """The four counts in every window QUANT uses on the raw series (its own make_intervals), skipping windows
    shorter than `min_length`, where order means nothing. X: (n_series, length) -> (n_series, up to 4 x 120).
    """
    windows = [
        (a, b)
        for a, b in make_intervals(X.shape[-1], depth).tolist()
        if b - a >= min_length
    ]

    def counts(X):
        return torch.cat([window_counts(X[:, a:b]) for a, b in windows], 1)

    return batched(counts, X, 512)


# == test =============================================================================================


def test_main(args):
    out = OUT / "order_counts.csv"
    done = finished(out)
    for name in args.datasets or NAMES:
        if name in done:
            continue
        try:
            t_start = time.perf_counter()
            d = Dataset(name)
            K, t = timed(order_counts, d.X)
            F = np.hstack([quant_features(d.X), K])
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
    new = pd.read_csv(OUT / "order_counts.csv")
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
        f"Idea 12: {METHOD}",
        f"{len(per)} datasets x {both['resample'].nunique()} resamples, same splits and seeds",
        "",
        f"QUANT           {100 * per['accuracy_quant'].mean():.2f}%",
        f"QUANT + idea 12 {100 * per['accuracy'].mean():.2f}%",
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
