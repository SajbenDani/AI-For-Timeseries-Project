"""Summarise the weakness tests into the numbers quoted in the report, plus one figure.

Usage: python scripts/weakness_analysis.py
Reads results/weaknesses/* (from scripts/weakness_tests.py), our reproduction results and the bake-off files.
Writes results/weaknesses/summary.txt and Papers/figures/weaknesses.pdf.
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "results" / "weaknesses"
NAMES = (ROOT / "codebase" / "quant" / "code" / "ucr142.txt").read_text().split()
REPRO = pd.read_csv(ROOT / "results" / "repro" / "quant_ucr142_30res.csv")
ACC = REPRO.groupby("dataset").accuracy.mean()  # our QUANT, mean over 30 resamples
STATS = pd.read_csv(ROOT / "results" / "checks" / "dataset_stats.csv", index_col=0)
TIME_COLS = ["time_transform_training_seconds", "time_transform_test_seconds",
             "time_fit_classifier_seconds", "time_predict_seconds"]
lines = []


def say(text=""):
    print(text)
    lines.append(str(text))


def bakeoff(method, metric="accuracy"):
    d = pd.read_csv(ROOT / "results" / "bakeoff_redux" / f"best_{method}_{metric}.csv", index_col=0)
    d.index = [re.sub("_disc$", "Discrete", re.sub("_nmv$", "Nmv", re.sub("_eq$", "Eq", i))) for i in d.index]
    return d.mean(axis=1)


def paired(a, b):
    """Mean difference, W/D/L and Wilcoxon p of per-dataset accuracies a vs b (W/D/L on 4 decimals)."""
    d = a.round(4) - b.round(4)
    p = wilcoxon(a, b, zero_method="pratt").pvalue if (d != 0).any() else 1.0
    return f"diff {d.mean():+.4f}  W/D/L {(d > 0).sum()}/{(d == 0).sum()}/{(d < 0).sum()}  Wilcoxon p={p:.2g}"


def train_labels(name):
    path = ROOT / "data" / "ucr142" / name / f"{name}_TRAIN.ts"
    lines_ = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = next(j for j, line in enumerate(lines_) if line.strip().lower().startswith("@data")) + 1
    return pd.Series([line.strip().rsplit(":", 1)[1] for line in lines_[start:] if line.strip()])


results = {}

# == 1. padded variable-length datasets ===============================================================
files = sorted((W / "padding").glob("*.csv.gz"))
if files:
    say("== 1. Padded variable-length datasets (11 datasets, 30 resamples) ==")
    p = pd.concat(pd.read_csv(f) for f in files)
    per = p.groupby("dataset").agg(quant=("quant_correct", "mean"), hydra=("hydra_correct", "mean"),
                                   padded=("padded_fraction", "mean"))
    say(f"sanity check, max |QUANT here - reproduction|: {(per.quant - ACC[per.index]).abs().max():.1e}")
    per["mr_hydra"] = bakeoff("MR-Hydra")[per.index]
    say(per.round(3).sort_values("padded").to_string())
    say(f"mean: QUANT {per.quant.mean():.4f}  Hydra {per.hydra.mean():.4f}  MR-Hydra (bake-off) {per.mr_hydra.mean():.4f}")
    gestures = [n for n in per.index if "Gesture" in n]
    say(f"QUANT - MR-Hydra: {len(gestures)} gesture datasets {(per.quant - per.mr_hydra)[gestures].mean():+.4f}, "
        f"original 112 datasets {(ACC - bakeoff('MR-Hydra'))[NAMES[:112]].mean():+.4f}")
    say(f"QUANT vs Hydra: {paired(per.quant, per.hydra)}")
    r = spearmanr(per.padded, per.quant - per.hydra)
    say(f"Spearman(mean padded fraction, QUANT - Hydra) over datasets: rho={r.statistic:+.2f} p={r.pvalue:.2g}")
    # within each dataset, compare the less-padded and the more-padded half of its test series, so that
    # datasets which are simply harder do not dominate the heavily padded group
    halves = {}
    for name, g in p.groupby("dataset"):
        more = g.padded_fraction > g.padded_fraction.median()
        halves[name] = {("less padded half", "QUANT"): g[~more].quant_correct.mean(),
                        ("less padded half", "Hydra"): g[~more].hydra_correct.mean(),
                        ("more padded half", "QUANT"): g[more].quant_correct.mean(),
                        ("more padded half", "Hydra"): g[more].hydra_correct.mean()}
    halves = pd.DataFrame(halves).T
    by_half = halves.mean().unstack()
    say("accuracy on the less vs more padded half of each dataset's test series (mean over datasets):")
    say(by_half.round(4).to_string())
    gap_less = halves["less padded half"].QUANT - halves["less padded half"].Hydra
    gap_more = halves["more padded half"].QUANT - halves["more padded half"].Hydra
    say(f"QUANT - Hydra: less padded {gap_less.mean():+.4f}, more padded {gap_more.mean():+.4f}; "
        f"difference Wilcoxon p={wilcoxon(gap_more, gap_less).pvalue:.2g}")
    results["padding"] = by_half
    say()

# == 2. train/test copies =============================================================================
if (W / "duplicates.csv").exists():
    say("== 2. Exact train/test copies (datasets with any duplicate series, 30 resamples) ==")
    d = pd.read_csv(W / "duplicates.csv")
    per = d.groupby("dataset").agg(copies=("n_copies", "sum"), n=("n_test", "sum"),
                                   conflicts=("n_conflicting_copies", "sum"), quant=("quant_acc", "mean"),
                                   quant_no_copies=("quant_acc_without_copies", "mean"),
                                   quant_on_copies=("quant_acc_on_copies", "mean"), nn=("nn_acc", "mean"),
                                   nn_no_copies=("nn_acc_without_copies", "mean"),
                                   nn_on_copies=("nn_acc_on_copies", "mean"))
    per["copy_share"] = per.copies / per.n
    per["quant_drop"] = per.quant - per.quant_no_copies
    say(f"sanity check, max |QUANT here - reproduction|: {(per.quant - ACC[per.index]).abs().max():.1e}")
    cols = ["copy_share", "conflicts", "quant", "quant_no_copies", "quant_on_copies", "quant_drop", "nn",
            "nn_no_copies", "nn_on_copies"]
    say(per[cols].sort_values("copy_share", ascending=False).round(3).to_string())
    dedup = ACC.copy()
    dedup[per.index] = per.quant_no_copies
    say(f"datasets: {len(per)};  QUANT accuracy on copied series: {per.quant_on_copies.mean():.3f}")
    say(f"142-dataset mean: {ACC.mean():.4f} -> {dedup.mean():.4f} without copied test series")
    say(f"datasets whose accuracy drops by >0.01 without copies: {(per.quant_drop > 0.01).sum()}")
    say()

# == 3. few training examples =========================================================================
if (W / "fewshot.csv").exists():
    say("== 3. Few training examples per class (learning curves, 10 resamples) ==")
    f = pd.read_csv(W / "fewshot.csv", dtype={"per_class": str})
    order = ["2", "5", "10", "20", "all"]
    per = f.groupby(["per_class", "method", "dataset"]).accuracy.mean().unstack("method")
    means = per.groupby(level=0).mean().loc[order]
    say(f"datasets: {per.index.get_level_values('dataset').nunique()}")
    say(means.round(4).to_string())
    for k in order:
        say(f"k={k:>3}: QUANT vs Hydra: {paired(per.loc[k].QUANT, per.loc[k].Hydra)}")
        say(f"       QUANT vs same features + ridge: {paired(per.loc[k].QUANT, per.loc[k]['QUANT features + ridge'])}")
    results["fewshot"] = means
    say()

if (W / "manyclass.csv").exists():
    say("== 3c. Many classes or <=5 examples per class (full training sets, 10 resamples) ==")
    m = pd.read_csv(W / "manyclass.csv").groupby(["dataset", "method"]).accuracy.mean().unstack()
    m["classes"] = STATS.classes[m.index]
    m["train_per_class"] = (STATS.train / STATS.classes)[m.index]
    say(m.round(3).sort_values("train_per_class").to_string())
    methods = ["QUANT", "QUANT features + ridge", "Hydra"]
    say(f"means: {m[methods].mean().round(4).to_dict()}")
    say(f"QUANT vs Hydra: {paired(m.QUANT, m.Hydra)}")
    say(f"QUANT vs same features + ridge: {paired(m.QUANT, m['QUANT features + ridge'])}")
    few = m[m.train_per_class <= 5]
    say(f"<=5 examples per class ({len(few)} datasets) means: {few[methods].mean().round(4).to_dict()}")
    say()

# small-data pattern across all 142 datasets (our results vs the bake-off's MultiRocket+Hydra)
say("== 3b. Across the 142 datasets: gap to MultiRocket+Hydra vs training examples per class ==")
gap = (ACC - bakeoff("MR-Hydra")).rename("gap")
per_class = (STATS.train / STATS.classes).rename("train_per_class")
for col, values in (("classes", STATS.classes), ("train per class", per_class)):
    r = spearmanr(values[gap.index], gap)
    say(f"Spearman({col}, QUANT - MR-Hydra): rho={r.statistic:+.2f} p={r.pvalue:.2g}")
groups = pd.cut(per_class[gap.index], [0, 5, 20, 100, np.inf], labels=["<=5", "5-20", "20-100", ">100"])
say(gap.groupby(groups, observed=True).agg(["mean", "count"]).round(4).to_string())
say()

# == 4. misalignment ==================================================================================
if (W / "shift.csv").exists():
    say("== 4. Misalignment stress test (random offsets, 10 resamples) ==")
    s = pd.read_csv(W / "shift.csv")
    per = s.groupby(["level", "method", "dataset"]).accuracy.mean().unstack("method")
    means = per.groupby(level=0).mean()
    say(f"datasets: {per.index.get_level_values('dataset').nunique()}")
    say(pd.concat([means, (means - means.loc[0.0]).add_suffix(" drop")], axis=1).round(4).to_string())
    for level in means.index:
        say(f"level {level:.2f}: QUANT vs Hydra: {paired(per.loc[level].QUANT, per.loc[level].Hydra)}")
    top = per.index.get_level_values(0).max()
    q_drop = per.loc[0.0].QUANT - per.loc[top].QUANT
    h_drop = per.loc[0.0].Hydra - per.loc[top].Hydra
    say(f"datasets losing >5 points at level {top}: QUANT {(q_drop > 0.05).sum()}, Hydra {(h_drop > 0.05).sum()}")
    say(f"largest QUANT drops: {q_drop.sort_values(ascending=False).head(5).round(3).to_dict()}")
    results["shift"] = means
    say()

# == 5. normalisation =================================================================================
if (W / "normalise.csv").exists():
    say("== 5. z-normalised vs raw input (datasets not already z-normalised, 30 resamples) ==")
    z = pd.read_csv(W / "normalise.csv").groupby("dataset").accuracy_znorm.mean()
    say(f"datasets: {len(z)};  z-normalised vs raw: {paired(z, ACC[z.index])}")
    diff = (z - ACC[z.index]).sort_values()
    say(f"largest losses from normalising: {diff.head(5).round(3).to_dict()}")
    say(f"largest gains from normalising: {diff.tail(5).round(3).to_dict()}")
    say()

# == 6. class imbalance (bake-off runs of QUANT and MultiRocket+Hydra) ================================
say("== 6. Class imbalance: balanced accuracy vs accuracy gap to MultiRocket+Hydra (bake-off runs) ==")
imbalance = pd.Series({n: (c := train_labels(n).value_counts()).max() / c.min() for n in NAMES})
extra = (bakeoff("QUANT", "balacc") - bakeoff("MR-Hydra", "balacc")) - (bakeoff("QUANT") - bakeoff("MR-Hydra"))
r = spearmanr(imbalance[extra.index], extra)
say(f"Spearman(imbalance ratio, extra balanced-accuracy gap): rho={r.statistic:+.2f} p={r.pvalue:.2g}")
say(extra.groupby(pd.cut(imbalance[extra.index], [0, 1.5, 3, np.inf], labels=["balanced (<=1.5)", "1.5-3", ">3"]),
                  observed=True).agg(["mean", "count"]).round(4).to_string())
say()

# == 7. cost on long series ===========================================================================
say("== 7. Compute cost (single-core run of the original split) ==")
t = pd.read_csv(ROOT / "results" / "repro" / "quant_ucr142_timing_1core.csv").set_index("dataset")
t["total"] = t[TIME_COLS].sum(axis=1)
t["length"] = STATS.length[t.index]
t["train"] = STATS.train[t.index]
say(f"features per time point (median): {(t.num_features / t.length).median():.2f}")
r = spearmanr(t.train * t.num_features, t.time_fit_classifier_seconds)
say(f"Spearman(train size x features, classifier time): rho={r.statistic:+.2f}")
longest = t.sort_values("length").tail(10)
say(f"10 longest-series datasets: {100 * longest.total.sum() / t.total.sum():.0f}% of total time "
    f"({longest.total.sum() / 60:.1f} of {t.total.sum() / 60:.1f} min)")

(W / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

# == figure ===========================================================================================
if {"shift", "fewshot", "padding"} <= results.keys():
    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    STYLE = {"QUANT": dict(color="#2a78d6", marker="o"), "Hydra": dict(color="#eb6834", marker="s"),
             "QUANT features + ridge": dict(color="#1baf7a", marker="^")}
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIX Two Text", "DejaVu Serif"],
                         "font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.15))

    def draw(ax, x, table, ticks, title, xlabel):
        for method in [m for m in STYLE if m in table.columns]:
            ax.plot(x, table[method].values, lw=1.5, ms=5, mec="white", mew=0.8, label=method, **STYLE[method])
        ax.set_xticks(x, ticks)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)

    shift = results["shift"]
    draw(axes[0], np.arange(len(shift)), shift, [f"{int(100 * v)}%" for v in shift.index],
         "(a) Patterns at random positions", "extra window length")
    axes[0].set_ylabel("mean accuracy")
    few = results["fewshot"]
    draw(axes[1], np.arange(len(few)), few, list(few.index), "(b) Few training examples",
         "training examples per class")
    pad = results["padding"]
    draw(axes[2], np.arange(len(pad)), pad, ["less padded\nhalf", "more padded\nhalf"],
         "(c) Padded variable-length series", "test series of each dataset")
    axes[2].set_xlim(-0.4, 1.4)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = ROOT / "Papers" / "figures" / "weaknesses.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"wrote {out}")
