"""Compare total compute time (training + inference over all 142 datasets) with the paper.

Usage: python scripts/compare_timing.py
Writes results/repro/summary_timing.csv.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "results" / "repro"
COLS = ["time_transform_training_seconds", "time_transform_test_seconds",
        "time_fit_classifier_seconds", "time_predict_seconds"]

runs = {
    "paper, 1 core (mean of 30 resamples)":
        pd.read_csv(ROOT / "codebase" / "quant" / "results" / "accuracy_mean_ucr142.csv", index_col=0)[COLS],
    "ours, 1 core (original split)":
        pd.read_csv(REPRO / "quant_ucr142_timing_1core.csv").set_index("dataset")[COLS],
    "ours, 12 threads (mean of 30 resamples)":
        pd.read_csv(REPRO / "quant_ucr142_30res.csv").groupby("dataset")[COLS].mean(),
}

rows = []
for name, t in runs.items():
    per_dataset = t.sum(axis=1)
    rows.append({
        "run": name, "n": len(t),
        "total_min": per_dataset.sum() / 60,
        "transform_min": (t[COLS[0]] + t[COLS[1]]).sum() / 60,
        "classifier_min": (t[COLS[2]] + t[COLS[3]]).sum() / 60,
        "median_per_dataset_s": per_dataset.median(),
        "slowest_dataset": per_dataset.idxmax(), "slowest_s": per_dataset.max(),
    })
summary = pd.DataFrame(rows).round(2)
summary.to_csv(REPRO / "summary_timing.csv", index=False)
print(summary.to_string(index=False))
