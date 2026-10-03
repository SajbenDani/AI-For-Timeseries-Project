"""Recreate the paper's Multiple Comparison Matrix rows (Fig. 1 and Fig. 6):
mean accuracy difference, win/draw/loss and Wilcoxon signed-rank p-value of QUANT vs other methods.

QUANT accuracies come from (a) the paper's published results, (b) the bake-off redux's independent
QUANT run, and (c) our reproduction. Other methods come from the bake-off redux result files.

Usage: python scripts/compare_methods.py [results/repro/quant_ucr142_30res.csv]
"""

import re
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
BAKEOFF = ROOT / "results" / "bakeoff_redux"
names = (ROOT / "codebase" / "quant" / "code" / "ucr142.txt").read_text().split()
UCR112 = names[:112]


def to_quant_name(name):
    name = re.sub("_eq$", "Eq", name)
    name = re.sub("_nmv$", "Nmv", name)
    return re.sub("_disc$", "Discrete", name)


def load_bakeoff(path):
    df = pd.read_csv(path, index_col=0)
    df.index = [to_quant_name(i) for i in df.index]
    return df.mean(axis=1)


def mcm_row(quant, other, datasets):
    q, o = quant.loc[datasets], other.loc[datasets]
    d = q - o
    p = wilcoxon(q, o, zero_method="pratt").pvalue
    return {"mean_other": o.mean(), "mean_quant": q.mean(), "mean_diff": d.mean(),
            "W": int((d > 0).sum()), "D": int((d == 0).sum()), "L": int((d < 0).sum()), "wilcoxon_p": p}


def main():
    quants = {"paper": pd.read_csv(ROOT / "codebase" / "quant" / "results" / "accuracy_mean_ucr142.csv",
                                   index_col=0)["accuracy"],
              "bakeoff": load_bakeoff(BAKEOFF / "best_QUANT_accuracy.csv")}
    ours_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "repro" / "quant_ucr142_30res.csv"
    if ours_path.exists():
        ours = pd.read_csv(ours_path).groupby("dataset")["accuracy"].mean()
        if len(ours) == 142:
            quants["ours"] = ours

    # rounding to 4 dp before W/D/L, so that floating-point noise does not count as a win/loss
    quants = {k: v.round(4) for k, v in quants.items()}

    fig1 = {"rSTSF": "interval_R-STSF", "DrCIF": "interval_DrCIF", "CIF": "interval_CIF",
            "STSF": "interval_STSF", "TSF": "interval_TSF"}
    fig6 = {"HC2": "best_HC2", "MultiRocket+Hydra": "best_MR-Hydra", "RDST": "best_RDST",
            "WEASEL-D": "best_WEASEL-2", "InceptionTime": "best_H-InceptionTime", "rSTSF": "interval_R-STSF",
            "FreshPRINCE": "best_FreshPRINCE", "PF": "best_PF"}

    rows = []
    for fig, methods, datasets in (("fig1_112", fig1, UCR112), ("fig6_142", fig6, names)):
        for method, file in methods.items():
            other = load_bakeoff(BAKEOFF / f"{file}_accuracy.csv").round(4)
            ds = [d for d in datasets if d in other.index]
            for source, q in quants.items():
                rows.append({"figure": fig, "method": method, "quant_source": source, "n": len(ds),
                             **mcm_row(q, other, ds)})
    out = pd.DataFrame(rows)
    out_path = ROOT / "results" / "repro" / "mcm_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.round(5).to_csv(out_path, index=False)
    with pd.option_context("display.width", 200):
        print(out.round(4).to_string(index=False))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
