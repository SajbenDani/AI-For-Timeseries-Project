"""Compare reproduced QUANT results with the published per-resample results.

Usage: python scripts/compare_results.py results/repro/quant_ucr142_30res.csv
Writes results/repro/comparison_per_dataset.csv and results/repro/summary_accuracy.csv and prints a summary.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_ind

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "codebase" / "quant" / "results"

ours_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "repro" / "quant_ucr142_30res.csv"
ours = pd.read_csv(ours_path)
pub_res = pd.read_csv(PUB / "accuracy_resamples_ucr142.csv", index_col=0)
pub_res.columns = pub_res.columns.astype(int)
pub_mean = pd.read_csv(PUB / "accuracy_mean_ucr142.csv", index_col=0)

names = (ROOT / "codebase" / "quant" / "code" / "ucr142.txt").read_text().split()
ucr112, new30 = names[:112], names[112:]

acc = ours.pivot(index="dataset", columns="resample", values="accuracy")
complete = acc.index[acc.notna().sum(axis=1) == acc.shape[1]]
acc = acc.loc[complete]
r = acc.columns

rows = []
for d in acc.index:
    a, b = acc.loc[d].values, pub_res.loc[d, r].values
    # Welch t-test across resamples: is the gap in mean accuracy explainable by resample/seed noise?
    p = ttest_ind(a, b, equal_var=False).pvalue if (a.std() + b.std()) > 0 else (1.0 if np.allclose(a.mean(), b.mean()) else 0.0)
    rows.append({
        "dataset": d, "group": "ucr112" if d in ucr112 else "new30",
        "acc_ours": a.mean(), "acc_paper": b.mean(), "diff": a.mean() - b.mean(),
        "std_ours": a.std(ddof=1), "std_paper": b.std(ddof=1),
        "corr_per_resample": np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan,
        "welch_p": p,
    })
cmp = pd.DataFrame(rows).set_index("dataset")

# independent QUANT run by the bake-off redux authors (on the corrected dataset versions)
bakeoff = pd.read_csv(ROOT / "results" / "bakeoff_redux" / "best_QUANT_accuracy.csv", index_col=0)
bakeoff.index = [re.sub("_disc$", "Discrete", re.sub("_nmv$", "Nmv", re.sub("_eq$", "Eq", i))) for i in bakeoff.index]
cmp.insert(2, "acc_bakeoff", bakeoff.mean(axis=1).reindex(cmp.index))

out = ours_path.parent / "comparison_per_dataset.csv"
cmp.round(6).to_csv(out)

# datasets fixed by the archive maintainers after the QUANT paper (see tsml-eval bake-off notebook)
CORRECTED = ["Covid3MonthDiscrete", "FloodModeling1Discrete", "FloodModeling2Discrete", "FloodModeling3Discrete"]
groups = {
    "UCR 112": cmp[cmp.group == "ucr112"],
    "New 30": cmp[cmp.group == "new30"],
    "New 30 w/o corrected": cmp[(cmp.group == "new30") & ~cmp.index.isin(CORRECTED)],
    "Corrected 4": cmp[cmp.index.isin(CORRECTED)],
    "All 142": cmp,
    "All w/o corrected": cmp[~cmp.index.isin(CORRECTED)],
}
summary = pd.DataFrame([{
    "group": k, "n": len(c), "paper": c.acc_paper.mean(), "bakeoff": c.acc_bakeoff.mean(), "ours": c.acc_ours.mean(),
    "abs_diff_lt_0.01": int((c["diff"].abs() < 0.01).sum()), "welch_p_lt_0.01": int((c.welch_p < 0.01).sum()),
} for k, c in groups.items() if len(c)])
summary.round(4).to_csv(ours_path.parent / "summary_accuracy.csv", index=False)
print(summary.round(4).to_string(index=False), "\n")

print(f"datasets complete: {len(cmp)} / 142  (resamples per dataset: {len(r)})")
for label, idx in (("all", cmp.index), ("ucr112", cmp.index.intersection(ucr112)), ("new30", cmp.index.intersection(new30))):
    c = cmp.loc[idx]
    if len(c) == 0:
        continue
    w = (c["diff"] > 1e-9).sum(); l = (c["diff"] < -1e-9).sum(); t = len(c) - w - l
    p = wilcoxon(c["acc_ours"], c["acc_paper"]).pvalue if (c["diff"].abs() > 0).any() else 1.0
    print(f"[{label:6s}] n={len(c):3d}  mean acc ours={c.acc_ours.mean():.4f}  paper={c.acc_paper.mean():.4f}  "
          f"diff={c['diff'].mean():+.4f}  W/D/L={w}/{t}/{l}  Wilcoxon p={p:.3f}  "
          f"|diff|<0.01: {(c['diff'].abs() < 0.01).sum()}  Welch p<0.01: {(c.welch_p < 0.01).sum()}")

print("\nlargest discrepancies:")
print(cmp.reindex(cmp["diff"].abs().sort_values(ascending=False).index).head(10)[
    ["acc_ours", "acc_paper", "diff", "std_ours", "welch_p"]].round(4).to_string())

times = ours[ours.dataset.isin(cmp.index)].groupby("dataset")[
    ["time_transform_training_seconds", "time_transform_test_seconds", "time_fit_classifier_seconds", "time_predict_seconds"]].mean()
pub_t = pub_mean.loc[times.index, times.columns]
print(f"\ntotal compute time per resample (sum over datasets, mean over resamples): "
      f"ours={times.sum().sum() / 60:.1f} min  paper={pub_t.sum().sum() / 60:.1f} min")
print("  breakdown ours (min):  ", (times.sum() / 60).round(2).to_dict())
print("  breakdown paper (min): ", (pub_t.sum() / 60).round(2).to_dict())
print(f"\nwrote {out}")
