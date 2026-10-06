"""Summarise the improvement tests into the numbers quoted in report.md, plus one figure.

Usage: python scripts/improvement_analysis.py
Reads results/improvements/* (from scripts/improvement_tests.py). Plain QUANT's results come from our reproduction
(same resamples and seeds), Hydra's from the ensemble test, QUANT's and Hydra's misalignment results from the
weakness tests, and MultiRocket+Hydra's and HC2's from the bake-off redux (30 resamples; for reference only).
Writes results/improvements/summary.txt and Papers/figures/improvements.pdf/.png.
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "improvements"
NAMES = (ROOT / "codebase" / "quant" / "code" / "ucr142.txt").read_text().split()
REPRO = pd.read_csv(ROOT / "results" / "repro" / "quant_ucr142_30res.csv")
MANYCLASS = sorted(pd.read_csv(ROOT / "results" / "weaknesses" / "manyclass.csv").dataset.unique())
GESTURES = [n for n in NAMES if "Gesture" in n and n.endswith("Eq")]
lines = []


def say(text=""):
    print(text)
    lines.append(str(text))


def bakeoff(method, resamples=30):
    """Bake-off redux accuracy per dataset, mean over its first `resamples` resamples (the same official splits)."""
    d = pd.read_csv(ROOT / "results" / "bakeoff_redux" / f"best_{method}_accuracy.csv", index_col=0)
    d.index = [re.sub("_disc$", "Discrete", re.sub("_nmv$", "Nmv", re.sub("_eq$", "Eq", i))) for i in d.index]
    return d.iloc[:, :resamples].mean(axis=1)


def paired(a, b):
    """Mean difference, W/D/L and Wilcoxon p of per-dataset accuracies a vs b (W/D/L on 4 decimals)."""
    d = a.round(4) - b.round(4)
    p = wilcoxon(a, b, zero_method="pratt").pvalue if (d != 0).any() else 1.0
    return d.mean(), (d > 0).sum(), (d == 0).sum(), (d < 0).sum(), p


def fmt(stats):
    diff, w, d, l_, p = stats
    return f"diff {100 * diff:+.2f} pts  W/D/L {w}/{d}/{l_}  Wilcoxon p={p:.2g}"


def quant_reference(resamples):
    r = REPRO[REPRO["resample"].isin(resamples)]
    return r.groupby("dataset").accuracy.mean(), r.groupby("dataset").time_fit_classifier_seconds.sum()


def load(name):
    path = R / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


results = {}

# == 0. reference ====================================================================================
q10, q10_time = quant_reference(range(10))
first = REPRO[REPRO["resample"] == 0].set_index("dataset")
q_transform = first.time_transform_training_seconds + first.time_transform_test_seconds  # all series, once
say("== 0. Reference: plain QUANT (our reproduction) ==")
say(f"142 datasets, 30 resamples: {REPRO.groupby('dataset').accuracy.mean().mean():.4f} (paper 0.8568)")
say(f"142 datasets, first 10 resamples (the improvement tests use these): {q10.mean():.4f}")
say()

# == 1. idea 10: variable-length QUANT ================================================================
v = load("varlen")
if v is not None:
    say("== 1. Idea 10: QUANT on variable-length series without padding (11 datasets, 30 resamples) ==")
    per = v.pivot_table(index="dataset", columns="method", values="accuracy", aggfunc="mean")
    base = "QUANT (padded, original)"
    q30 = REPRO.groupby("dataset").accuracy.mean()
    say(f"sanity check, max |padded QUANT here - reproduction|: {(per[base] - q30[per.index]).abs().max():.1e}")
    per["MR-Hydra (bake-off)"] = bakeoff("MR-Hydra")[per.index]
    order = [base, "QUANT (stretched to equal length) + length", "QUANT (native windows)",
             "QUANT (native windows) + length", "QUANT (padded + native windows)", "MR-Hydra (bake-off)"]
    order = [m for m in order if m in per.columns]
    say(per[order].round(3).to_string())
    say("means: " + ", ".join(f"{m} {per[m].mean():.4f}" for m in order))
    for m in order[1:-1]:
        diff = per[m] - per[base]
        gap = per["MR-Hydra (bake-off)"] - per[base]
        say(f"{m} vs padded: {fmt(paired(per[m], per[base]))}; closes {100 * diff.mean() / gap.mean():.0f}% of the "
            f"gap to MR-Hydra; now ahead of MR-Hydra on {(per[m] > per['MR-Hydra (bake-off)']).sum()} of 11")
    say(f"native + length vs stretched + length: "
        f"{fmt(paired(per['QUANT (native windows) + length'], per['QUANT (stretched to equal length) + length']))}")
    gain = (per["QUANT (native windows) + length"] - per[base]).mean()
    say(f"effect on the 142-dataset mean if native + length replaced padded QUANT on these 11: "
        f"{q30.mean():.4f} -> {q30.mean() + gain * 11 / 142:.4f}")
    results["varlen"] = per[order].mean()
    say()

# == 2. ideas 1, 2, 3, 5 on all 142 datasets ==========================================================
say("== 2. Ideas 1, 2, 3, 5 on the 142 datasets (first 10 resamples, seed = resample) ==")
ens = load("ensemble")
frames = [d for d in (load("pooled"), load("convq"), ens, load("augment")) if d is not None]
table = []
if frames:
    allres = pd.concat(frames)
    allres = allres[allres["resample"] < 10]
    complete = allres.groupby(["method", "dataset"])["resample"].nunique().eq(10)
    if ens is not None:
        qe = ens[ens.method == "QUANT"].groupby("dataset").accuracy.mean()
        say(f"sanity check, max |QUANT in the ensemble run - reproduction|: {(qe - q10[qe.index]).abs().max():.1e}")
        fit_ref = ens[ens.method == "QUANT"].groupby("dataset").fit_seconds.sum()  # QUANT timed in the same run
    else:
        fit_ref = q10_time
    for method, g in allres.groupby("method", sort=False):
        done = [n for n in NAMES if complete.get((method, n), False)]
        if method == "QUANT" or not done:
            continue
        acc = g[g.dataset.isin(done)].groupby("dataset").accuracy.mean()[done]
        stats = paired(acc, q10[done])
        sub = lambda names: (acc - q10[done])[[n for n in names if n in done]].mean()  # noqa: E731
        fit = g[g.dataset.isin(done)].groupby("dataset").fit_seconds.sum()
        both = [n for n in done if n in fit_ref.index]
        extra_per = g[g.dataset.isin(done)].groupby("dataset").transform_seconds.first()
        extra = extra_per.sum()
        # end to end for one train/test split: features of all series (QUANT's, if used, plus the idea's own)
        # + training + prediction, vs plain QUANT's features + training + prediction, same run and machine
        own_quant = 0 if method in ("Hydra", "conv quantiles alone") else q_transform[both]
        mean_fit = g[g.dataset.isin(both)].groupby("dataset").fit_seconds.mean()
        end_to_end = (own_quant + extra_per[both] + mean_fit[both]).sum()
        reference = (q_transform[both] + fit_ref[both] / 10).sum()
        table.append({"method": method, "datasets": len(done), "accuracy": acc.mean(), "QUANT": q10[done].mean(),
                      "diff": stats[0], "W": stats[1], "D": stats[2], "L": stats[3], "p": stats[4],
                      "manyclass_diff": sub(MANYCLASS), "gesture_diff": sub(GESTURES),
                      "time_ratio": fit[both].sum() / fit_ref[both].sum(),
                      "extra_feature_minutes": extra / 60, "quant_feature_minutes": q_transform[done].sum() / 60,
                      "end_to_end_minutes": end_to_end / 60, "quant_end_to_end_minutes": reference / 60,
                      "end_to_end_ratio": end_to_end / reference})
        say(f"{method:32s} ({len(done)} datasets) {acc.mean():.4f} vs QUANT {q10[done].mean():.4f}: {fmt(stats)}")
        d = (acc - q10[done]).sort_values()
        say(f"    many-class/few-example (18): {100 * sub(MANYCLASS):+.2f} pts   gestures (10): "
            f"{100 * sub(GESTURES):+.2f} pts   training time x{table[-1]['time_ratio']:.1f}   extra feature time "
            f"{extra / 60:.1f} min (QUANT's own: {q_transform[done].sum() / 60:.1f} min)")
        say(f"    end to end, one split of these datasets, all cores: {end_to_end / 60:.1f} min vs QUANT "
            f"{reference / 60:.1f} min (x{end_to_end / reference:.1f})")
        say(f"    largest gains: {', '.join(f'{n} {100 * x:+.1f}' for n, x in d[::-1][:4].items())}")
        say(f"    worst changes: {', '.join(f'{n} {100 * x:+.1f}' for n, x in d[:4].items())}")
    table = pd.DataFrame(table)
    if len(table):
        (R / "summary_142.csv").write_text(table.round(5).to_csv(index=False), encoding="utf-8")
        results["benchmark"] = table
    say()
    say("bake-off redux on the same first 10 resamples: " + ", ".join(
        f"{m} {bakeoff(m, 10)[NAMES].mean():.4f}" for m in ("QUANT", "MR-Hydra", "HC2")))
    if len(table):
        best = table.sort_values("diff").iloc[-1]
        g = allres[allres.method == best.method]
        if g.dataset.nunique() == len(NAMES):
            acc = g.groupby("dataset").accuracy.mean()[NAMES]
            for m in ("MR-Hydra", "HC2"):
                say(f"{best.method} vs {m} (bake-off, same 10 resamples): {fmt(paired(acc, bakeoff(m, 10)[NAMES]))}")
    if ens is not None:
        w = ens[ens.method == "QUANT + Hydra (weighted)"]
        say(f"weighted ensemble: QUANT's mean weight {w.weight_quant.mean():.2f}, median temperature "
            f"{w.temperature.median():.2g}")
        best = ens.groupby(["method", "dataset"]).accuracy.mean().unstack(0)
        say(f"oracle (best of QUANT and Hydra per dataset): {best[['QUANT', 'Hydra']].max(axis=1).mean():.4f}")
    say()

# == 3. misalignment stress test ======================================================================
s = load("shift")
if s is not None and frames:
    say("== 3. Misalignment stress test (30 datasets, first 5 resamples) ==")
    done = sorted(s.groupby("dataset").level.nunique().loc[lambda x: x == 3].index)
    resamples = sorted(s["resample"].unique())
    base = pd.read_csv(ROOT / "results" / "weaknesses" / "shift.csv")
    base = base[base.dataset.isin(done) & base["resample"].isin(resamples)]
    level0 = allres[allres.dataset.isin(done) & allres["resample"].isin(resamples)].assign(level=0.0)
    every = pd.concat([base, level0[level0.method != "QUANT"], s[s.dataset.isin(done)]])
    curve = every.groupby(["level", "method"]).accuracy.mean().unstack()
    curve = curve[[c for c in curve.columns if curve[c].notna().all()]]
    say(f"datasets: {len(done)}; resamples {resamples}")
    say(curve.round(4).T.to_string())
    say("drop from level 0 to 0.5 (pts): " + ", ".join(
        f"{m} {100 * (curve[m][0.5] - curve[m][0.0]):+.1f}" for m in curve.columns))
    per = every[every.level == 0.5].groupby(["method", "dataset"]).accuracy.mean().unstack(0)
    for m in [c for c in curve.columns if c not in ("QUANT", "Hydra")]:
        say(f"level 0.5, {m} vs QUANT: {fmt(paired(per[m], per['QUANT']))}")
    results["shift"] = curve
    say()

# == 4. compute time on one core =====================================================================
t = load("timing_1core")
if t is not None:
    say("== 4. Compute time on one CPU core (original split; features of all series + training + prediction) ==")
    t = t.set_index("dataset")
    repro_1core = pd.read_csv(ROOT / "results" / "repro" / "quant_ucr142_timing_1core.csv").set_index("dataset")
    cols = [c for c in repro_1core.columns if c.startswith("time_")]
    same = (t.quant_accuracy.round(6) == repro_1core.accuracy[t.index].round(6)).sum()
    say(f"datasets: {len(t)}; QUANT's accuracy equals the reproduction's on {same} of them")
    totals = {
        "QUANT": t.quant_features + t.quant_fit_predict,
        "QUANT + pooled quantiles (idea 1)": t.quant_features + t.pooled_features + t.pooled_fit_predict,
        "QUANT + conv quantiles (idea 2)": t.quant_features + t.conv_features + t.conv_fit_predict,
        "Hydra alone": t.hydra_features + t.hydra_fit_predict,
        "QUANT + Hydra, equal weights (idea 3)": t.quant_features + t.hydra_features + t.quant_fit_predict
        + t.hydra_fit_predict,
        "QUANT + Hydra, weighted (idea 3)": t.quant_features + t.hydra_features + t.weighted_fit_predict,
    }
    base_time = totals["QUANT"].sum()
    say(f"sanity: QUANT here {base_time / 60:.1f} min vs the reproduction's timing run "
        f"{repro_1core.loc[t.index, cols].sum().sum() / 60:.1f} min (paper: 14.9 min for all 142)")
    for k, v in totals.items():
        say(f"{k:40s} {v.sum() / 60:7.1f} min   x{v.sum() / base_time:.1f}")
    say("parts (min): " + ", ".join(f"{c} {t[c].sum() / 60:.1f}" for c in t.columns if not c.endswith("accuracy")))
    cv = t.weighted_fit_predict - t.quant_fit_predict - t.hydra_fit_predict
    say(f"cross-validation for the weights: {cv.sum() / 60:.1f} min")
    slow = (totals["QUANT + Hydra, weighted (idea 3)"] / 60).sort_values().tail(5)
    say("slowest datasets for the weighted ensemble (min): " + ", ".join(f"{n} {m:.1f}" for n, m in slow.items()))
    pd.DataFrame({"minutes": {k: v.sum() / 60 for k, v in totals.items()},
                  "x_quant": {k: v.sum() / base_time for k, v in totals.items()}}).round(3).to_csv(
        R / "timing_1core_summary.csv")
    say()

R.mkdir(parents=True, exist_ok=True)
(R / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

# == figure ===========================================================================================
if {"benchmark", "shift", "varlen"} <= results.keys():
    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9893"
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIX Two Text", "DejaVu Serif"],
                         "font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45), gridspec_kw={"width_ratios": [1.35, 1, 1]})

    def tidy(ax, title):
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
        ax.set_axisbelow(True)

    t = results["benchmark"].set_index("method")
    shown = [m for m in ["QUANT + pooled quantiles", "QUANT + conv quantiles", "conv quantiles alone",
                         "QUANT + Hydra (equal weights)", "QUANT + Hydra (weighted)", "QUANT + augmentation",
                         "QUANT + augmentation + TTA"] if m in t.index][::-1]
    y = np.arange(len(shown))
    ax = axes[0]
    ax.barh(y + 0.18, 100 * t.loc[shown, "diff"], 0.36, color=BLUE, label="all 142 datasets")
    ax.barh(y - 0.18, 100 * t.loc[shown, "manyclass_diff"], 0.36, color=ORANGE, label="18 many-class datasets")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y, [m.replace("QUANT + ", "+ ") for m in shown])
    ax.set_xlabel("accuracy change vs QUANT (points)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    tidy(ax, "(a) Change in accuracy")

    c = results["shift"]
    ax = axes[1]
    styles = {"QUANT": (BLUE, "o", "-"), "Hydra": (ORANGE, "s", "-"), "QUANT + pooled quantiles": (GREY, "^", "--"),
              "QUANT + conv quantiles": (GREEN, "D", "--"), "QUANT + augmentation + TTA": (INK, "v", ":"),
              "QUANT + Hydra (weighted)": ("#8a4fd1", "P", "-")}
    x = np.arange(len(c))
    for m, (color, marker, ls) in styles.items():
        if m in c.columns:
            ax.plot(x, c[m].values, ls=ls, color=color, marker=marker, lw=1.3, ms=4, mec="white", mew=0.6,
                    label=m.replace("QUANT + ", "+ "))
    ax.set_xticks(x, [f"{int(100 * v)}%" for v in c.index])
    ax.set_xlabel("extra window length (random offset)")
    ax.set_ylabel("mean accuracy")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=6, loc="lower left")
    tidy(ax, "(b) Patterns at random positions")

    vl = results["varlen"]
    ax = axes[2]
    short = {"QUANT (padded, original)": "padded\n(QUANT)", "QUANT (stretched to equal length) + length":
             "stretched\n+ length", "QUANT (native windows)": "native", "QUANT (native windows) + length":
             "native\n+ length", "QUANT (padded + native windows)": "padded\n+ native"}
    bars = [m for m in short if m in vl.index]
    ax.bar(range(len(bars)), vl[bars].values, color=[BLUE, GREY] + [GREEN] * (len(bars) - 2), width=0.6)
    ax.axhline(vl["MR-Hydra (bake-off)"], color=ORANGE, lw=1.2, ls="--")
    ax.text(len(bars) - 0.65, vl["MR-Hydra (bake-off)"] + 0.003, "MR-Hydra", color=ORANGE, ha="right",
            va="bottom", fontsize=7)
    ax.set_xticks(range(len(bars)), [short[m] for m in bars], fontsize=6.5)
    ax.set_ylim(0.7, max(vl.values) + 0.03)
    ax.grid(axis="y", color=GRID, lw=0.6)
    tidy(ax, "(c) 11 variable-length datasets")
    fig.tight_layout()
    out = ROOT / "Papers" / "figures" / "improvements.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"wrote {out}")
