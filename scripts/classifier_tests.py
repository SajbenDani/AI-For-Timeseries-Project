"""Idea 13 (ideas.md): QUANT's features with another classifier instead of its ExtraTrees forest.

  python scripts/classifier_tests.py main rf       # Random Forest, all 142 datasets (first 10 resamples)
  python scripts/classifier_tests.py main lgbm     # LightGBM (gradient boosting), same
  python scripts/classifier_tests.py summary       # -> results/classifiers/summary.txt, compared with plain QUANT

Only the classifier changes; QUANT's features are the same. Random Forest gets QUANT's own settings (200 trees,
max_features=0.1, entropy), so the only difference to ExtraTrees is how split points are chosen (best instead of
random) and that every tree sees a bootstrap sample. LightGBM gets 200 boosting rounds and, like QUANT's forest, a
random 10% of the features per tree; its minimum leaf size is lowered to 2 so it can learn from datasets with only
2 training series per class (with its default of 20 it could not split at all there).

LightGBM runs in a separate Python process that never loads torch: torch and LightGBM each bring their own OpenMP
library, and on macOS the two crash when they are loaded in the same process. QUANT's features (computed with torch)
are handed over in a temporary file.

Same splits and seeds as improvement_tests.py and efficient_tests.py; plain QUANT is not re-run (its accuracies are in
results/efficient/efficient.csv). Each classifier writes its own file, so they can run separately; finished
datasets are skipped when a run is restarted, and a dataset that fails is reported and skipped.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "classifiers"


def random_forest(seed):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=200,
        max_features=0.1,
        criterion="entropy",
        n_jobs=-1,
        random_state=seed,
    )


def lightgbm(seed):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=200,
        colsample_bytree=0.1,
        min_child_samples=2,
        min_data_in_bin=1,
        n_jobs=-1,
        random_state=seed,
        verbose=-1,
    )


CLASSIFIERS = {
    "rf": "QUANT features + Random Forest",
    "lgbm": "QUANT features + LightGBM",
}


# == the classifiers ==================================================================================


def fit_predict(key, F, y, splits):
    """Accuracy and fit time of classifier `key` on every (train, test) split."""
    rows = []
    for i, (train, test) in enumerate(splits):
        t0 = time.perf_counter()
        make = random_forest if key == "rf" else lightgbm
        prediction = make(i).fit(F[train], y[train]).predict(F[test])
        rows.append(
            {
                "accuracy": float((prediction == y[test]).mean()),
                "fit_seconds": time.perf_counter() - t0,
            }
        )
    return rows


def worker(path):
    """Separate process for LightGBM: reads the features and splits, prints the results as JSON. No torch here."""
    z = np.load(path)
    splits = [(z[f"train{i}"], z[f"test{i}"]) for i in range(int(z["resamples"]))]
    print(json.dumps(fit_predict("lgbm", z["F"], z["y"], splits)))


def in_separate_process(F, y, splits):
    fd, path = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    try:
        np.savez(
            path,
            F=F,
            y=y,
            resamples=len(splits),
            **{f"train{i}": tr for i, (tr, _) in enumerate(splits)},
            **{f"test{i}": te for i, (_, te) in enumerate(splits)},
        )
        done = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "worker", path],
            capture_output=True,
            text=True,
        )
        if done.returncode != 0:
            raise RuntimeError(
                f"LightGBM process failed (exit code {done.returncode}):\n{done.stderr[-3000:]}"
            )
        return json.loads(done.stdout.strip().splitlines()[-1])
    finally:
        os.remove(path)


# == test =============================================================================================


def test_main(args):
    from weakness_tests import NAMES, Dataset, append, finished, quant_features

    for key in args.classifiers or list(CLASSIFIERS):
        out = OUT / f"{key}.csv"
        done = finished(out)
        for name in args.datasets or NAMES:
            if name in done:
                continue
            try:
                t_start = time.perf_counter()
                d = Dataset(name)
                F = quant_features(d.X)
                splits = [d.split(i) for i in range(args.resamples or 10)]
                results = (
                    in_separate_process(F, d.y, splits)
                    if key == "lgbm"
                    else fit_predict(key, F, d.y, splits)
                )
                append(
                    [
                        {
                            "dataset": name,
                            "resample": i,
                            "method": CLASSIFIERS[key],
                            **r,
                            "num_features": F.shape[1],
                        }
                        for i, r in enumerate(results)
                    ],
                    out,
                )
                print(
                    f"{key} {name}: done in {time.perf_counter() - t_start:.0f} s",
                    flush=True,
                )
            except Exception:
                print(
                    f"{key} {name}: FAILED, skipped\n{traceback.format_exc()}",
                    flush=True,
                )


# == summary ==========================================================================================


def test_summary(args):
    from scipy.stats import wilcoxon

    base = pd.read_csv(ROOT / "results" / "efficient" / "efficient.csv")
    base = base[base["method"] == "QUANT"][
        ["dataset", "resample", "accuracy", "fit_seconds"]
    ]
    lines = [
        "Idea 13: QUANT's features with another classifier (ExtraTrees = plain QUANT)",
        "",
    ]
    for key, method in CLASSIFIERS.items():
        path = OUT / f"{key}.csv"
        if not path.exists():
            continue
        new = pd.read_csv(path)
        both = new.merge(base, on=["dataset", "resample"], suffixes=("", "_quant"))
        per = both.groupby("dataset")[
            ["accuracy", "accuracy_quant", "fit_seconds", "fit_seconds_quant"]
        ].mean()
        d = np.round(per["accuracy"] - per["accuracy_quant"], 4)
        p = (
            wilcoxon(per["accuracy"], per["accuracy_quant"], zero_method="pratt").pvalue
            if (d != 0).any()
            else 1.0
        )
        lines += [
            f"{method}: {len(per)} datasets x {both['resample'].nunique()} resamples",
            f"  QUANT (ExtraTrees) {100 * per['accuracy_quant'].mean():.2f}%   "
            f"{method.split(' + ')[1]} {100 * per['accuracy'].mean():.2f}%",
            f"  change {100 * d.mean():+.2f} pts   W/D/L {(d > 0).sum()}/{(d == 0).sum()}/{(d < 0).sum()}   "
            f"Wilcoxon p={p:.2g}",
            f"  training time vs ExtraTrees (total over datasets): "
            f"x{per['fit_seconds'].sum() / per['fit_seconds_quant'].sum():.1f}",
            "  largest gains (pts): "
            + ", ".join(
                f"{n} {100 * v:+.1f}"
                for n, v in d.sort_values(ascending=False).head(5).items()
            ),
            "  largest losses (pts): "
            + ", ".join(
                f"{n} {100 * v:+.1f}" for n, v in d.sort_values().head(5).items()
            ),
            "",
        ]
    text = "\n".join(lines)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.txt").write_text(text + "\n")
    print(text)


TESTS = {"main": test_main, "summary": test_summary}

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "worker":
        worker(sys.argv[2])
        sys.exit()
    p = argparse.ArgumentParser()
    p.add_argument("test", choices=list(TESTS))
    p.add_argument("classifiers", nargs="*", help="main only: rf, lgbm (default: both)")
    p.add_argument(
        "--datasets", nargs="*", help="only these datasets (default: all 142)"
    )
    p.add_argument(
        "--resamples", type=int, help="override the number of resamples (default 10)"
    )
    args = p.parse_args()
    unknown = set(args.classifiers) - set(CLASSIFIERS)
    if unknown:
        p.error(
            f"unknown classifier(s) {sorted(unknown)}; choose from {list(CLASSIFIERS)}"
        )
    TESTS[args.test](args)
