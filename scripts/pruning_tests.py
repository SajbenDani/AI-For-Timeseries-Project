"""Idea 9 (ideas.md): prune QUANT's features to make training faster.

  python scripts/pruning_tests.py main      # all 142 datasets (first 10 resamples)
  python scripts/pruning_tests.py summary   # -> results/pruning/summary.txt

Training the forest takes about 78% of QUANT's time, and that time grows with the number of features. So: fit a small
forest (50 trees, otherwise QUANT's settings) on the training data, keep only the 50%, 25% or 10% of features it
finds most important, and fit QUANT's normal forest (200 trees) on those. Only training data is used to choose them.

For a fair time comparison, plain QUANT's forest is fitted again in the same run on the same machine; its accuracies
must equal those in results/efficient/efficient.csv (the summary checks this). Times are training only (selection
forest + final forest), on all cores, relative to plain QUANT's forest in the same run.
Same splits and seeds as improvement_tests.py and efficient_tests.py; finished datasets are skipped when the test is
re-run, and a dataset that fails is reported and skipped.
"""

import argparse
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from weakness_tests import NAMES, ROOT, Dataset, append, finished, quant_features
from improvement_tests import forest

warnings.filterwarnings("ignore")

OUT = ROOT / "results" / "pruning"
KEEP = (0.5, 0.25, 0.1)
SELECTION_TREES = 50


def fit_score(F, y, train, test, seed, n_estimators=200):
    t0 = time.perf_counter()
    clf = forest(seed, n_estimators).fit(F[train], y[train])
    seconds = time.perf_counter() - t0
    return clf, (clf.predict(F[test]) == y[test]).mean(), seconds


# == test =============================================================================================


def test_main(args):
    out = OUT / "pruning.csv"
    done = finished(out)
    for name in args.datasets or NAMES:
        if name in done:
            continue
        try:
            t_start = time.perf_counter()
            d = Dataset(name)
            F = quant_features(d.X)
            rows = []
            for i in range(args.resamples or 10):
                train, test = d.split(i)
                base = {"dataset": name, "resample": i}
                _, acc, sec = fit_score(F, d.y, train, test, i)
                rows.append(
                    {
                        **base,
                        "method": "QUANT",
                        "keep": 1.0,
                        "accuracy": acc,
                        "fit_seconds": sec,
                        "num_features": F.shape[1],
                    }
                )
                t0 = time.perf_counter()
                selector = forest(i, SELECTION_TREES).fit(F[train], d.y[train])
                selection_seconds = time.perf_counter() - t0
                order = np.argsort(-selector.feature_importances_, kind="stable")
                for keep in KEEP:
                    columns = np.sort(order[: max(1, int(round(keep * F.shape[1])))])
                    _, acc, sec = fit_score(F[:, columns], d.y, train, test, i)
                    rows.append(
                        {
                            **base,
                            "method": f"QUANT, {keep:.0%} of features",
                            "keep": keep,
                            "accuracy": acc,
                            "fit_seconds": selection_seconds + sec,
                            "num_features": len(columns),
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
    df = pd.read_csv(OUT / "pruning.csv")
    q = df[df["method"] == "QUANT"].set_index(["dataset", "resample"])
    ref = pd.read_csv(ROOT / "results" / "efficient" / "efficient.csv")
    ref = ref[ref["method"] == "QUANT"].set_index(["dataset", "resample"])["accuracy"]
    check = (q["accuracy"] - ref.reindex(q.index)).abs().max()
    per_q = q.groupby(level="dataset")[["accuracy", "fit_seconds"]].mean()
    lines = [
        "Idea 9: feature pruning (selection forest of 50 trees, then QUANT's forest on the kept features)",
        f"{len(per_q)} datasets x {q.index.get_level_values('resample').nunique()} resamples; "
        f"plain QUANT re-fitted in the same run (largest difference to efficient.csv: {check:.4f})",
        "",
        f"QUANT, all features      {100 * per_q['accuracy'].mean():.2f}%   training time x1.00",
        "",
    ]
    for keep in KEEP:
        p_ = df[df["keep"] == keep].set_index(["dataset", "resample"])
        per = (
            p_.groupby(level="dataset")[["accuracy", "fit_seconds"]]
            .mean()
            .loc[per_q.index]
        )
        d = np.round(per["accuracy"] - per_q["accuracy"], 4)
        p = (
            wilcoxon(per["accuracy"], per_q["accuracy"], zero_method="pratt").pvalue
            if (d != 0).any()
            else 1.0
        )
        speed = per_q["fit_seconds"].sum() / per["fit_seconds"].sum()
        lines += [
            f"QUANT, {keep:.0%} of features   {100 * per['accuracy'].mean():.2f}%   "
            f"change {100 * d.mean():+.2f} pts   W/D/L {(d > 0).sum()}/{(d == 0).sum()}/{(d < 0).sum()}   "
            f"Wilcoxon p={p:.2g}   training {speed:.1f}x faster",
            "   largest losses (pts): "
            + ", ".join(
                f"{n} {100 * v:+.1f}" for n, v in d.sort_values().head(3).items()
            ),
            "",
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
