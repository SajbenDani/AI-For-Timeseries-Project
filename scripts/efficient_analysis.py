"""Summarise the tests of the efficient approaches (ideas 14-18) into the numbers quoted in report.md, plus one figure.

Usage: python scripts/efficient_analysis.py
Reads results/efficient/* (from scripts/efficient_tests.py; partial results are fine). Plain QUANT's results come
from our reproduction (same resamples and seeds), QUANT's and Hydra's misalignment results from the weakness tests,
MultiRocket+Hydra's and HC2's from the bake-off redux, and the dataset groups from results/ideas/per_dataset.csv.
Writes results/efficient/summary.txt and Papers/figures/efficient.pdf/.png.
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
R = ROOT / "results" / "efficient"
NAMES = (ROOT / "codebase" / "quant" / "code" / "ucr142.txt").read_text().split()
REPRO = pd.read_csv(ROOT / "results" / "repro" / "quant_ucr142_30res.csv")
STATS = pd.read_csv(ROOT / "results" / "checks" / "dataset_stats.csv", index_col="dataset")
IDEAS = pd.read_csv(ROOT / "results" / "ideas" / "per_dataset.csv", index_col="dataset")
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


def load(name):
    path = R / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


# dataset groups (all chosen before these tests, from the weakness tests and results/ideas/checks.txt)
MANYCLASS = sorted(pd.read_csv(ROOT / "results" / "weaknesses" / "manyclass.csv").dataset.unique())
GESTURES = [n for n in NAMES if "Gesture" in n and n.endswith("Eq")]
PERIODIC = [n for n in NAMES if IDEAS.periodicity[n] >= 0.3]
PF_WINS = [n for n in NAMES if bakeoff("PF")[n] - bakeoff("QUANT")[n] > 0.01]
SKEWED = [n for n in NAMES if IDEAS["unbalanced all"][n] >= IDEAS["unbalanced all"].quantile(0.75)]
GROUPS = {"many-class": MANYCLASS, "gestures": GESTURES, "repeating": PERIODIC, "elastic wins": PF_WINS,
          "skewed features": SKEWED}
IDEA = {"QUANT + speed quantiles": ("14", "elastic wins"), "QUANT + cycle view": ("15", "repeating"),
        "QUANT + bispectrum view": ("16", "many-class"), "QUANT, rank-uniform splits": ("17", "skewed features")}
GATED = {"QUANT + Hydra, gated": ("Hydra", "hydra"), "QUANT + Hydra-16, gated": ("Hydra-16", "hydra16")}
results, complete, table = {}, [], []
q10 = REPRO[REPRO["resample"] < 10].groupby("dataset").accuracy.mean()

# == 0. reference ====================================================================================
e = load("efficient")
say("== 0. Reference: plain QUANT (our reproduction, first 10 resamples) ==")
say(f"142 datasets: {q10[NAMES].mean():.4f}")
say("groups: " + ", ".join(f"{k} {len(v)}" for k, v in GROUPS.items()))
if e is not None:
    e = e[e["resample"] < 10]
    complete = [n for n in NAMES if e[e.dataset == n]["resample"].nunique() == 10]
    qe = e[e.method == "QUANT"].groupby("dataset").accuracy.mean()
    say(f"datasets finished: {len(complete)} of 142; sanity check, max |QUANT in this run - reproduction|: "
        f"{(qe - q10[qe.index]).abs().max():.1e}")
say()

# == 1. ideas 14-17 on the 142 datasets ===============================================================
if e is not None and complete:
    say("== 1. Ideas 14-17 on the 142 datasets (first 10 resamples, seed = resample) ==")
    q = q10[complete]
    hydra_gap = e[e.method == "Hydra"].groupby("dataset").accuracy.mean() - q10  # Hydra (this run) - QUANT
    MECHANISM = {"14": (bakeoff("PF") - bakeoff("QUANT"), "elastic method's lead over QUANT (PF - QUANT)"),
                 "15": (IDEAS.periodicity, "how strongly the series repeat"),
                 "16": (hydra_gap, "Hydra's lead over QUANT (position-free shapes)"),
                 "17": (IDEAS["unbalanced all"], "share of skewed features")}
    for method, (idea, target) in IDEA.items():
        g = e[(e.method == method) & e.dataset.isin(complete)]
        acc = g.groupby("dataset").accuracy.mean()[complete]
        stats = paired(acc, q)
        groups = {k: (acc - q)[[n for n in v if n in complete]] for k, v in GROUPS.items()}
        row = {"idea": idea, "method": method, "datasets": len(complete), "accuracy": acc.mean(), "QUANT": q.mean(),
               "diff": stats[0], "W": stats[1], "D": stats[2], "L": stats[3], "p": stats[4], "target": target,
               "target_diff": groups[target].mean(),
               "features_ratio": (g.groupby("dataset").num_features.first() /
                                  e[e.method == "QUANT"].groupby("dataset").num_features.first())[complete].mean()}
        for k, v in groups.items():
            row[f"{k}_diff"] = v.mean()
        table.append(row)
        say(f"idea {idea}: {method:28s} {acc.mean():.4f} vs QUANT {q.mean():.4f}: {fmt(stats)}")
        sub = [n for n in GROUPS[target] if n in complete]
        if sub:
            say(f"    target group ({target}, {len(sub)}): {fmt(paired(acc[sub], q[sub]))}")
        say("    groups (pts): " + ", ".join(f"{k} {100 * v.mean():+.2f}" for k, v in groups.items()))
        d = (acc - q).sort_values()
        say(f"    largest gains: {', '.join(f'{n} {100 * x:+.1f}' for n, x in d[::-1][:5].items())}")
        say(f"    worst changes: {', '.join(f'{n} {100 * x:+.1f}' for n, x in d[:5].items())}")
        say(f"    features x{row['features_ratio']:.2f} (cost on one core: section 4)")
        driver, label = MECHANISM[idea]
        rho, p_rho = spearmanr((acc - q).values, driver[complete].values)
        row["mechanism_rho"], row["mechanism_p"] = rho, p_rho
        say(f"    does the gain follow the idea's mechanism? Spearman(gain, {label}) = {rho:+.2f} (p = {p_rho:.2g})")
    table = pd.DataFrame(table)
    results["benchmark"] = table
    say()

# == 2. idea 18: Hydra only where QUANT is unsure =====================================================
t = load("timing_1core")
timing_source = "results/efficient/timing_1core.csv"
if t is None or not set(complete) <= set(t.dataset):  # until the new timing run is done: the earlier one
    t, timing_source = pd.read_csv(ROOT / "results" / "improvements" / "timing_1core.csv"), \
        "results/improvements/timing_1core.csv (earlier run; Hydra-16 not timed)"
t = t.set_index("dataset")
quant_seconds = t.quant_features + t.quant_fit_predict
quant_minutes = quant_seconds[complete].sum() / 60 if complete else np.nan
MIN_ASKED = [0, 0.02, 0.05, 0.1, 0.2]  # two-level gate: train Hydra only if this share of test series is unsure


def gate_table(g, key):
    """Every threshold and minimum share of unsure test series: each dataset's mean accuracy and one-core seconds over
    the resamples. Hydra is trained (features of all training series + classifier) only if at least that share, and at
    least one, of the test series is unsure, and then computes features only for those test series (its transform
    works series by series, so its time is proportional to the number of series it transforms)."""
    have_time = f"{key}_features" in t.columns
    base = g[g.threshold == 0].set_index(["dataset", "resample"]).accuracy
    out = []
    for th, gt in g.groupby("threshold"):
        gt = gt.set_index(["dataset", "resample"])
        ds = gt.index.get_level_values(0)
        n_test, n_train = STATS.test.reindex(ds).values, STATS.train.reindex(ds).values
        n_asked = gt.asked.values * n_test
        hydra = (t[f"{key}_features"].reindex(ds).values * (n_train + n_asked) / (n_train + n_test)
                 + t[f"{key}_fit_predict"].reindex(ds).values) if have_time else np.full(len(gt), np.nan)
        for m in MIN_ASKED:
            use = (gt.asked.values >= m) & (n_asked > 0)
            frame = pd.DataFrame({"dataset": ds, "hydra": use, "asked": np.where(use, gt.asked.values, 0),
                                  "accuracy": np.where(use, gt.accuracy.values, base.reindex(gt.index).values),
                                  "seconds": quant_seconds.reindex(ds).values + np.where(use, hydra, 0)})
            out.append(frame.groupby("dataset").mean().assign(threshold=th, min_asked=m))
    return pd.concat(out).reset_index()


def summarise(tab):
    by = tab.groupby(["threshold", "min_asked"])
    return pd.DataFrame({"asked": by.asked.mean(), "hydra_datasets": by.hydra.mean(), "accuracy": by.accuracy.mean(),
                         "minutes": by.seconds.agg(lambda v: v.to_numpy().sum()) / 60})  # NaN if not timed


def choose(tab, halves, keep=0.9):
    """Operating point picked without the test results of the datasets it is used on: on each half of the datasets,
    the cheapest setting that keeps `keep` of the full ensemble's gain there is applied to the other half.
    Returns each dataset's row at the setting chosen on the other half, and the two settings."""
    rows, settings = [], []
    for a, b in ((halves[0], halves[1]), (halves[1], halves[0])):
        s = summarise(tab[tab.dataset.isin(a)])
        base, top = s.accuracy.loc[(0, 0)], s.accuracy.loc[(s.index.get_level_values(0).max(), 0)]
        best = s[s.accuracy >= base + keep * (top - base)].sort_values(["minutes", "asked"]).index[0]
        settings.append(best)
        rows.append(tab[tab.dataset.isin(b) & (tab.threshold == best[0]) & (tab.min_asked == best[1])])
    return pd.concat(rows).set_index("dataset"), settings


if e is not None and complete:
    say("== 2. Idea 18: Hydra only where QUANT is unsure (142 datasets, first 10 resamples) ==")
    say(f"one-core times from {timing_source}; QUANT alone {quant_minutes:.1f} min")
    ens = pd.read_csv(ROOT / "results" / "improvements" / "ensemble.csv")
    old = ens[ens.method == "QUANT + Hydra (equal weights)"].groupby("dataset").accuracy.mean()
    rule = (STATS.classes >= 10) | (STATS.train / STATS.classes <= 10)  # uses only the training data's size
    halves = [[n for n in NAMES[0::2] if n in complete], [n for n in NAMES[1::2] if n in complete]]
    q = q10[complete]
    gating = {}
    for label, (alone, key) in GATED.items():
        g = e[(e.method == label) & e.dataset.isin(complete)]
        if g.empty:
            continue
        tab = gate_table(g, key)
        top = tab.threshold.max()
        full = tab[(tab.threshold == top) & (tab.min_asked == 0)].set_index("dataset")
        hydra_acc = e[(e.method == alone) & e.dataset.isin(complete)].groupby("dataset").accuracy.mean()
        say(f"{alone} alone {hydra_acc.mean():.4f}. QUANT + {alone} with every test series asked (equal weights, free "
            f"calibration): {full.accuracy[complete].mean():.4f}, {full.seconds[complete].to_numpy().sum() / 60:.1f} min: "
            f"{fmt(paired(full.accuracy[complete], q))}")
        if alone == "Hydra":
            both = [n for n in complete if n in old.index]
            say(f"    vs idea 3's equal weights with cross-validated calibration: {full.accuracy[both].mean():.4f} vs "
                f"{old[both].mean():.4f} ({fmt(paired(full.accuracy[both], old[both]))})")
        gain = full.accuracy[complete].mean() - q.mean()
        curve = summarise(tab[tab.min_asked == 0]).droplevel(1)
        outside = tab[(tab.min_asked == 0) & ~tab.dataset.map(rule)]
        sized = summarise(pd.concat([tab[(tab.min_asked == 0) & tab.dataset.map(rule)],
                                     outside.assign(accuracy=q[outside.dataset].values, asked=0.0, hydra=False,
                                                    seconds=quant_seconds[outside.dataset].values)])).droplevel(1)
        curve["accuracy_size_rule"], curve["minutes_size_rule"] = sized.accuracy, sized.minutes
        curve["gain_kept"] = (curve.accuracy - q.mean()) / gain
        say("per-series gate (Hydra asked when QUANT's top-two margin < threshold); size rule = Hydra only on datasets "
            "with >= 10 classes or <= 10 training series per class:")
        say(curve.round(4).to_string())
        two = summarise(tab)
        front = two.sort_values(["minutes", "accuracy"], ascending=[True, False])
        front = front[front.accuracy > front.accuracy.cummax().shift(fill_value=-1)]
        say("two-level gate (also skip Hydra for a dataset when fewer than min_asked of its test series are unsure), "
            "settings on the accuracy/cost front:")
        say(front.round(4).to_string())
        for variant, sub in (("per-series gate", tab[tab.min_asked == 0]), ("two-level gate", tab)):
            pick, settings = choose(sub, halves)
            acc, minutes = pick.accuracy[complete], pick.seconds[complete].to_numpy().sum() / 60
            say(f"{variant}, settings chosen on the other half {[(float(x), float(m)) for x, m in settings]} (threshold, min share unsure): "
                f"{acc.mean():.4f}, {minutes:.1f} min on one core (x{minutes / quant_minutes:.2f} QUANT), Hydra trained "
                f"on {pick.hydra[complete].mean():.0%} of datasets, asked {pick.asked[complete].mean():.1%} of test "
                f"series, keeps {(acc.mean() - q.mean()) / gain:.0%} of the full gain")
            say(f"    vs QUANT: {fmt(paired(acc, q))}; many-class "
                f"{100 * (acc - q)[[n for n in MANYCLASS if n in complete]].mean():+.2f} pts")
            gating[(label, variant)] = {"accuracy": acc.mean(), "minutes": minutes, "stats": paired(acc, q),
                                        "manyclass": (acc - q)[[n for n in MANYCLASS if n in complete]].mean(),
                                        "full": full.accuracy[complete].mean(),
                                        "full_minutes": full.seconds[complete].to_numpy().sum() / 60}
        if len(complete) == len(NAMES):
            # fresh-data check: the two-level gate was designed while the run was going, when the first 60 datasets
            # were finished; pick its setting on those 60 alone and test it on the 82 it had never seen
            seen, fresh = NAMES[:60], NAMES[60:]
            s = summarise(tab[tab.dataset.isin(seen)])
            base, top = s.accuracy.loc[(0, 0)], s.accuracy.loc[(s.index.get_level_values(0).max(), 0)]
            best = s[s.accuracy >= base + 0.9 * (top - base)].sort_values(["minutes", "asked"]).index[0]
            f = tab[tab.dataset.isin(fresh) & (tab.threshold == best[0]) & (tab.min_asked == best[1])].set_index(
                "dataset")
            qf, ff = q[fresh].mean(), full.accuracy[fresh].mean()
            minutes, q_min = f.seconds[fresh].to_numpy().sum() / 60, quant_seconds[fresh].sum() / 60
            say(f"fresh check: two-level setting {(float(best[0]), float(best[1]))} chosen on the first 60 datasets; on "
                f"the other 82: {f.accuracy[fresh].mean():.4f} vs QUANT {qf:.4f} and full ensemble {ff:.4f}, keeps "
                f"{(f.accuracy[fresh].mean() - qf) / (ff - qf):.0%} of the gain at x{minutes / q_min:.2f} QUANT's time "
                f"(full x{full.seconds[fresh].to_numpy().sum() / 60 / q_min:.2f}); vs QUANT: "
                f"{fmt(paired(f.accuracy[fresh], q[fresh]))}")
        gating[label] = (curve, front)
    for m in ("MR-Hydra", "HC2"):
        say(f"bake-off, same 10 resamples: {m} {bakeoff(m, 10)[complete].mean():.4f}")
    results["gating"] = gating
    say()

# == 3. stress tests ==================================================================================
w = load("warp")
if w is not None and e is not None:
    say("== 3a. Speed stress test: time-warped series (idea 14; 30 datasets, first 5 resamples) ==")
    done = sorted(w.groupby("dataset")["level"].nunique().loc[lambda x: x == 3].index)
    level0 = e[e.dataset.isin(done) & (e["resample"] < 5) & e.method.isin(["QUANT", "QUANT + speed quantiles"])]
    every = pd.concat([level0.assign(level=0.0), w[w.dataset.isin(done)]])
    curve = every.groupby(["level", "method"]).accuracy.mean().unstack()
    say(f"datasets: {len(done)}")
    say(curve.round(4).T.to_string())
    say("drop from level 0 to the strongest warp (pts): " + ", ".join(
        f"{m} {100 * (curve[m].iloc[-1] - curve[m].iloc[0]):+.1f}" for m in curve.columns))
    for level in curve.index[1:]:
        per = every[every["level"] == level].groupby(["method", "dataset"]).accuracy.mean().unstack(0)
        say(f"level {level}: + speed quantiles vs QUANT: {fmt(paired(per['QUANT + speed quantiles'], per['QUANT']))}")
    results["warp"] = curve
    say()

s = load("shift")
if s is not None and e is not None:
    say("== 3b. Misalignment stress test (ideas 15 and 16; 30 datasets, first 5 resamples) ==")
    done = sorted(s.groupby("dataset")["level"].nunique().loc[lambda x: x == 3].index)
    base = pd.read_csv(ROOT / "results" / "weaknesses" / "shift.csv")
    base = base[base.dataset.isin(done) & (base["resample"] < 5)]
    new = ["QUANT + cycle view", "QUANT + bispectrum view"]
    level0 = e[e.dataset.isin(done) & (e["resample"] < 5) & e.method.isin(new)].assign(level=0.0)
    every = pd.concat([base, level0, s[s.dataset.isin(done)]])
    curve = every.groupby(["level", "method"]).accuracy.mean().unstack()
    say(f"datasets: {len(done)}")
    say(curve.round(4).T.to_string())
    say("drop from level 0 to 0.5 (pts): " + ", ".join(
        f"{m} {100 * (curve[m][0.5] - curve[m][0.0]):+.1f}" for m in curve.columns))
    per = every[every["level"] == 0.5].groupby(["method", "dataset"]).accuracy.mean().unstack(0)
    for m in new:
        say(f"level 0.5, {m} vs QUANT: {fmt(paired(per[m], per['QUANT']))}")
    periodic = [n for n in done if n in PERIODIC]
    if periodic:
        say(f"repeating datasets among them ({len(periodic)}): level 0.5, cycle view vs QUANT "
            f"{fmt(paired(per.loc[periodic, 'QUANT + cycle view'], per.loc[periodic, 'QUANT']))}")
    results["shift"] = curve
    say()

# == 4. compute time on one core =====================================================================
tn = load("timing_1core")
if tn is not None:
    say("== 4. Compute time on one CPU core (original split; features of all series + training + prediction) ==")
    tn = tn.set_index("dataset")
    first = REPRO[REPRO["resample"] == 0].set_index("dataset").accuracy
    say(f"datasets: {len(tn)}; QUANT's accuracy equals the reproduction's on "
        f"{(tn.quant_accuracy.round(6) == first[tn.index].round(6)).sum()} of them")
    totals = {
        "QUANT": tn.quant_features + tn.quant_fit_predict,
        "QUANT + speed quantiles (14)": tn.quant_features + tn.speed_features + tn.speed_fit_predict,
        "QUANT + cycle view (15)": tn.quant_features + tn.cycle_features + tn.cycle_fit_predict,
        "QUANT + bispectrum view (16)": tn.quant_features + tn.bispectrum_features + tn.bispectrum_fit_predict,
        "QUANT, rank-uniform splits (17)": tn.quant_features + tn.rank_fit_predict,
        "Hydra alone": tn.hydra_features + tn.hydra_fit_predict,
        "Hydra-16 alone": tn.hydra16_features + tn.hydra16_fit_predict,
        "QUANT + Hydra, equal weights, free calibration": tn.quant_features + tn.quant_fit_predict
        + tn.hydra_features + tn.hydra_fit_predict,
        "QUANT + Hydra, equal weights, calibrated as in idea 3": tn.quant_features + tn.quant_fit_predict
        + tn.hydra_features + tn.hydra_fit_predict + tn.hydra_cv_calibration,
        "QUANT + Hydra-16, equal weights, free calibration": tn.quant_features + tn.quant_fit_predict
        + tn.hydra16_features + tn.hydra16_fit_predict,
    }
    base_time = totals["QUANT"].sum()
    for k, v in totals.items():
        say(f"{k:55s} {v.sum() / 60:7.1f} min   x{v.sum() / base_time:.2f}")
    say("parts (min): " + ", ".join(f"{c} {tn[c].sum() / 60:.1f}" for c in tn.columns if not c.endswith("accuracy")))
    pd.DataFrame({"minutes": {k: v.sum() / 60 for k, v in totals.items()},
                  "x_quant": {k: v.sum() / base_time for k, v in totals.items()}}).round(3).to_csv(
        R / "timing_1core_summary.csv")
    results["timing"] = {k: v.sum() / base_time for k, v in totals.items()}
    say()

R.mkdir(parents=True, exist_ok=True)
(R / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
if len(table):
    table.round(5).to_csv(R / "summary_142.csv", index=False)

# == figure ===========================================================================================
if {"benchmark", "gating", "warp", "shift"} <= results.keys():
    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    BLUE, ORANGE, GREEN, GREY, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#9a9893", "#8a4fd1"
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIX Two Text", "DejaVu Serif"],
                         "font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.9))

    def tidy(ax, title):
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
        ax.set_axisbelow(True)

    # (a) change in accuracy: all datasets and each idea's target group
    tb = results["benchmark"].set_index("method")
    op = results["gating"][("QUANT + Hydra, gated", "two-level gate")]
    labels = [f"{tb.idea[m]}. {m.replace('QUANT + ', '+ ').replace('QUANT, ', '')}\n(target: {tb.target[m]})"
              for m in tb.index] + ["18. + Hydra, two-level gate\n(target: many-class)"]
    all_diff = list(100 * tb["diff"]) + [100 * (op["accuracy"] - q10[complete].mean())]
    target_diff = list(100 * tb["target_diff"]) + [100 * op["manyclass"]]
    y = np.arange(len(labels))[::-1]
    ax = axes[0, 0]
    ax.barh(y + 0.18, all_diff, 0.36, color=BLUE, label="142 datasets")
    ax.barh(y - 0.18, target_diff, 0.36, color=ORANGE, label="target group")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y, labels, fontsize=6.5)
    ax.set_xlabel("accuracy change vs QUANT (points)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.legend(frameon=False, loc="center right", fontsize=6.5)
    tidy(ax, "(a) Change in accuracy")

    # (b) idea 18: accuracy against one-core time
    ax = axes[0, 1]
    for label, color in (("QUANT + Hydra, gated", PURPLE), ("QUANT + Hydra-16, gated", GREEN)):
        if label not in results["gating"]:
            continue
        curve, front = results["gating"][label]
        name = label.replace("QUANT + ", "+ ").replace(", gated", "")
        ax.plot(curve.minutes, 100 * curve.accuracy, "-o", color=color, ms=3, lw=1.2, mec="white", mew=0.5,
                label=f"{name}: ask when unsure")
        ax.plot(front.minutes, 100 * front.accuracy, "--", color=color, lw=1.0, label=f"{name}: two-level gate")
        o = results["gating"][(label, "two-level gate")]
        ax.plot(o["minutes"], 100 * o["accuracy"], "*", color=color, ms=10, mec="white", mew=0.5)
    ax.plot(quant_minutes, 100 * q10[complete].mean(), "o", color=BLUE, ms=5)
    ax.annotate("QUANT", (quant_minutes, 100 * q10[complete].mean()), xytext=(4, -9), textcoords="offset points",
                fontsize=7, color=BLUE)
    mr = 100 * bakeoff("MR-Hydra", 10)[complete].mean()
    ax.axhline(mr, color=ORANGE, lw=1, ls=":")
    ax.annotate("MultiRocket+Hydra (about 1 h)", (1, mr), xycoords=("axes fraction", "data"), xytext=(-2, 2),
                textcoords="offset points", ha="right", va="bottom", fontsize=6.5, color=ORANGE)
    ax.set_xlabel("minutes on one core (all 142 datasets, one split)")
    ax.set_ylabel("mean accuracy (%)")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    tidy(ax, "(b) Idea 18: accuracy vs cost (star: chosen setting)")

    # (c) speed stress test
    ax = axes[1, 0]
    c = results["warp"]
    x = np.arange(len(c))
    for m, color, marker in (("QUANT", BLUE, "o"), ("QUANT + speed quantiles", GREEN, "D")):
        ax.plot(x, 100 * c[m].values, "-", marker=marker, color=color, lw=1.3, ms=4, mec="white", mew=0.6,
                label=m.replace("QUANT + ", "+ "))
    ax.set_xticks(x, ["none"] + [f"±{v:.1f}" for v in c.index[1:]])
    ax.set_xlabel("random time warp (largest change of log-speed)")
    ax.set_ylabel("mean accuracy (%)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    tidy(ax, "(c) Series played at different speeds")

    # (d) misalignment stress test
    ax = axes[1, 1]
    c = results["shift"]
    x = np.arange(len(c))
    styles = {"QUANT": (BLUE, "o", "-"), "Hydra": (ORANGE, "s", "-"), "QUANT + cycle view": (GREEN, "D", "--"),
              "QUANT + bispectrum view": (PURPLE, "^", "--")}
    for m, (color, marker, ls) in styles.items():
        if m in c.columns:
            ax.plot(x, 100 * c[m].values, ls=ls, color=color, marker=marker, lw=1.3, ms=4, mec="white", mew=0.6,
                    label=m.replace("QUANT + ", "+ "))
    ax.set_xticks(x, [f"{int(100 * v)}%" for v in c.index])
    ax.set_xlabel("extra window length (random offset)")
    ax.set_ylabel("mean accuracy (%)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    tidy(ax, "(d) Patterns at random positions")
    fig.tight_layout()
    out = ROOT / "Papers" / "figures" / "efficient.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"wrote {out}")
