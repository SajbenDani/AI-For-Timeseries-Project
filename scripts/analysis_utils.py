"""Helpers shared by the analysis and summary steps. Nothing here trains a model or needs data/: only results/ is read.

  bakeoff      published results of other methods (bake-off redux), on the same official resamples
  paired       the comparison every table uses: mean difference, wins/draws/losses and a Wilcoxon signed-rank test
  two_level    idea 18's two-level gate at one fixed setting
  quant_second_machine   plain QUANT on the computer that ran rounds 3 and 4
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
NAMES = (ROOT / "codebase" / "quant" / "code" / "ucr142.txt").read_text().split()
KEY = ["dataset", "resample"]


def bakeoff(method, resamples=30, metric="accuracy"):
    """Bake-off redux result per dataset, mean over its first `resamples` resamples (the same official splits)."""
    d = pd.read_csv(ROOT / "results" / "bakeoff_redux" / f"best_{method}_{metric}.csv", index_col=0)
    d.index = [re.sub("_disc$", "Discrete", re.sub("_nmv$", "Nmv", re.sub("_eq$", "Eq", i))) for i in d.index]
    return d.iloc[:, :resamples].mean(axis=1)


def paired(a, b):
    """Mean difference, W/D/L and Wilcoxon p of per-dataset accuracies a vs b. As in the QUANT paper, wins, draws
    and losses are counted on accuracies rounded to 4 decimals; the test uses the unrounded values."""
    d = a.round(4) - b.round(4)
    p = wilcoxon(a, b, zero_method="pratt").pvalue if (d != 0).any() and len(d) > 1 else 1.0
    return d.mean(), (d > 0).sum(), (d == 0).sum(), (d < 0).sum(), p


def fmt(stats):
    diff, w, d, l_, p = stats
    return f"diff {100 * diff:+.2f} pts  W/D/L {w}/{d}/{l_}  Wilcoxon p={p:.2g}"


def two_level(df, method, alone, threshold, min_asked):
    """Idea 18's two-level gate at one setting, per (dataset, resample): the gated accuracy at `threshold` where at
    least `min_asked` of a dataset's test series were unsure, and the accuracy without Hydra (`alone`) elsewhere."""
    g = df[(df["method"] == method) & np.isclose(df["threshold"], threshold)].set_index(KEY)
    return g["accuracy"].where(g["asked"] >= min_asked, alone.reindex(g.index))


def quant_second_machine():
    """Plain QUANT per (dataset, resample) on the computer that ran rounds 3 and 4 (accuracy and forest fit time,
    from the pruning run). With the same splits and seeds its mean accuracy equals the first computer's (85.36%), but
    about half of the single splits differ, so ideas tested on this computer are compared with this run."""
    df = pd.read_csv(ROOT / "results" / "pruning" / "pruning.csv")
    return df[df["method"] == "QUANT"].set_index(KEY)[["accuracy", "fit_seconds"]]
