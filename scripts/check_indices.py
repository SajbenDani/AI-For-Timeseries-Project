"""Find out which resample indices (Java or Python tsml resampler) the published QUANT results used.

Runs QUANT on a few small datasets with both index sets and correlates the per-resample accuracies
with the published per-resample accuracies. The matching index set gives a high correlation;
the wrong one gives a correlation near zero.

Usage: python scripts/check_indices.py
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "checks"
DATASETS = ["Beef", "ACSF1", "Adiac", "Car", "OliveOil",  # from the original 112
            "Colposcopy", "Covid3MonthDiscrete", "Tools", "SharePriceIncrease"]  # from the new 30


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "idx_check.txt").write_text("\n".join(DATASETS) + "\n")
    for kind in ("java", "python"):
        out = OUT / f"idx_{kind}.csv"
        if not out.exists():
            subprocess.run([sys.executable, str(ROOT / "scripts" / "run_repro.py"), "-d", str(OUT / "idx_check.txt"),
                            "--indices", kind, "--seed", "0", "-o", str(out)], check=True)

    pub = pd.read_csv(ROOT / "codebase" / "quant" / "results" / "accuracy_resamples_ucr142.csv", index_col=0)
    rows = []
    for d in DATASETS:
        row = {"dataset": d}
        for kind in ("java", "python"):
            r = pd.read_csv(OUT / f"idx_{kind}.csv").pivot(index="dataset", columns="resample", values="accuracy")
            # resample 0 is the original split for both, so it is excluded from the correlation
            row[f"corr_{kind}"] = np.corrcoef(r.loc[d].values[1:], pub.loc[d].values[1:])[0, 1]
        rows.append(row)
    table = pd.DataFrame(rows).set_index("dataset").round(2)
    table.to_csv(OUT / "idx_check_correlations.csv")
    print(table.to_string())


if __name__ == "__main__":
    main()
