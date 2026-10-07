"""Quick, read-only checks behind the "efficient approaches" (ideas 14-18) in ideas.md.

Usage: python scripts/idea_checks.py
Nothing here trains a new model. The script only measures properties of the data (the original training split of
every dataset), of QUANT's features, and of results we already have (bake-off redux, results/improvements/).
Writes results/ideas/checks.txt and results/ideas/per_dataset.csv.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from weakness_tests import NAMES, ROOT, Dataset  # noqa: E402  (same loader as every other test)
from quant import IntervalModel, Quant  # noqa: E402  (on the path once weakness_tests has imported run_repro)

OUT = ROOT / "results" / "ideas"
IMP = ROOT / "results" / "improvements"
STATS = pd.read_csv(ROOT / "results" / "checks" / "dataset_stats.csv", index_col="dataset")
VIEWS = ["raw", "1st difference", "2nd difference", "Fourier magnitude"]
lines = []


def say(text=""):
    print(text)
    lines.append(str(text))


def bakeoff(method, resamples=30):
    """Bake-off redux accuracy per dataset, mean over its first `resamples` resamples (the same official splits)."""
    d = pd.read_csv(ROOT / "results" / "bakeoff_redux" / f"best_{method}_accuracy.csv", index_col=0)
    d.index = [re.sub("_disc$", "Discrete", re.sub("_nmv$", "Nmv", re.sub("_eq$", "Eq", i))) for i in d.index]
    return d.iloc[:, :resamples].mean(axis=1)


# == per-dataset measurements ==========================================================================

def periodicity(X):
    """How strongly the series repeat themselves: the highest autocorrelation peak at a lag that fits at least 3
    times into the series (median over series, 0 = no repetition), and the number of cycles at that lag."""
    X = X - X.mean(1, keepdims=True)
    L = X.shape[1]
    lo, hi = 3, L // 3
    if hi < lo + 2:
        return 0.0, np.nan
    acf = np.fft.irfft(np.abs(np.fft.rfft(X, n=2 * L)) ** 2)[:, :L]
    acf = acf / np.maximum(acf[:, :1], 1e-12)
    a = acf[:, lo - 1:hi + 2]  # lags lo-1 .. hi+1, so that every lag lo .. hi has both neighbours
    mid = a[:, 1:-1]
    peaks = np.where((mid > a[:, :-2]) & (mid >= a[:, 2:]), mid, -np.inf)
    best, found = peaks.max(1), np.isfinite(peaks.max(1))
    cycles = L / (peaks.argmax(1) + lo)
    best = np.where(found, np.maximum(best, 0.0), 0.0)  # a negative peak means no repetition
    return float(np.median(best)), float(np.median(cycles[found])) if found.any() else np.nan


def views(X):
    """QUANT's features exactly as in quant.py, but kept apart per view (raw, differences, Fourier magnitude)."""
    Xt = torch.from_numpy(X).unsqueeze(1)
    out = []
    for function in Quant().representation_functions:
        Z = function(Xt)
        out.append(IntervalModel(Z.shape[-1]).fit_transform(Z, None).numpy())
    return out


def split_balance(F):
    """Expected share of training series on the smaller side of a root split when the threshold is drawn uniformly
    between the feature's minimum and maximum, as scikit-learn's ExtraTrees does. A threshold drawn uniformly
    over the *ranks* instead gives about 0.25 for every feature."""
    S = np.sort(F, axis=0)
    k = np.arange(1, len(S))[:, None] / len(S)  # share of series below a threshold between neighbours k-1 and k
    span = (S[-1] - S[0]).astype(np.float64)
    share = (np.diff(S, axis=0).astype(np.float64) * np.minimum(k, 1 - k)).sum(0) / np.where(span > 0, span, np.nan)
    return share[np.isfinite(share)]


def measure():
    """Per-dataset measurements; reused from results/ideas/per_dataset.csv if it exists (delete it to redo)."""
    if (OUT / "per_dataset.csv").exists():
        return pd.read_csv(OUT / "per_dataset.csv", index_col="dataset")
    rows = []
    for name in NAMES:
        d = Dataset(name)
        X = d.X[:d.n_train]  # original training split, float32 as QUANT sees it
        p, c = periodicity(X.astype(np.float64))
        shares = [split_balance(F) for F in views(X)]
        rows.append({"dataset": name, "periodicity": p, "cycles": c,
                     **{f"median_share {v}": np.median(s) for v, s in zip(VIEWS, shares)},
                     **{f"unbalanced {v}": np.mean(s < 0.1) for v, s in zip(VIEWS, shares)},
                     "unbalanced all": np.mean(np.concatenate(shares) < 0.1)})
        print(f"{name:32s} periodicity {p:.2f}  cycles {c:6.1f}  unbalanced {rows[-1]['unbalanced all']:.2f}", flush=True)
    per = pd.DataFrame(rows).set_index("dataset")
    per.to_csv(OUT / "per_dataset.csv", float_format="%.4f")
    return per


# == summaries ===========================================================================================

def cycles_section(per, acc):
    say("== 1. Periodic series (idea 15: cycle view) ==")
    say("periodicity = median over training series of the highest autocorrelation peak at a lag that fits >= 3 times")
    gap = acc["QUANT"] - acc["MR-Hydra"]
    rho, p = spearmanr(per.periodicity, gap)
    say(f"Spearman(periodicity, QUANT - MR-Hydra) = {rho:+.2f} (p = {p:.2g}) over 142 datasets")
    periodic = per.periodicity >= 0.3
    for ref in ["MR-Hydra", "HC2"]:
        g = acc.QUANT - acc[ref]
        say(f"  QUANT - {ref}: {100 * g[periodic].mean():+.2f} pts on the {periodic.sum()} periodic datasets (>= 0.3) vs "
            f"{100 * g[~periodic].mean():+.2f} on the other {(~periodic).sum()} (Mann-Whitney p = "
            f"{mannwhitneyu(g[periodic], g[~periodic]).pvalue:.2g})")
    say("Most periodic datasets (QUANT, and its gap to MR-Hydra and HC2, in points):")
    top = per.sort_values("periodicity", ascending=False).head(25)
    for name, r in top.iterrows():
        say(f"  {name:32s} periodicity {r.periodicity:.2f}  cycles {r.cycles:5.1f}  QUANT {100 * acc.QUANT[name]:5.1f}  "
            f"vs MR-Hydra {100 * gap[name]:+6.1f}  vs HC2 {100 * (acc.QUANT - acc.HC2)[name]:+6.1f}  {STATS.type.get(name, '')}")
    say()


def splits_section(per, acc):
    say("== 2. Random split thresholds on QUANT's features (idea 17: rank-uniform splits) ==")
    say("share = expected share of training series on the smaller side of a root split, threshold uniform in [min, max]")
    say("(scikit-learn's ExtraTrees); a threshold uniform over the ranks would give about 0.25 for every feature")
    for v in VIEWS:
        say(f"  {v:18s}: median share {per[f'median_share {v}'].median():.3f}; "
            f"features with share < 0.10: {100 * per[f'unbalanced {v}'].mean():4.1f}% (mean over datasets)")
    say(f"  all views         : features with share < 0.10: {100 * per['unbalanced all'].mean():.1f}% (mean over datasets)")
    rho, p = spearmanr(per["unbalanced all"], acc.QUANT - acc["MR-Hydra"])
    many = pd.read_csv(ROOT / "results" / "weaknesses" / "manyclass.csv").dataset.unique()
    say(f"  does it line up with QUANT's losses? Spearman(unbalanced share, QUANT - MR-Hydra) = {rho:+.2f} (p = {p:.2g}); "
        f"many-class datasets {100 * per.loc[many, 'unbalanced all'].mean():.0f}% vs others "
        f"{100 * per.drop(many)['unbalanced all'].mean():.0f}%")
    say()


def gating_section():
    say("== 3. Where the ensemble's gain and cost come from (idea 18: second opinion only when needed) ==")
    e = pd.read_csv(IMP / "ensemble.csv").pivot_table(index="dataset", columns="method", values="accuracy")
    gain = (e["QUANT + Hydra (equal weights)"] - e["QUANT"]).loc[NAMES]
    t = pd.read_csv(IMP / "timing_1core.csv", index_col="dataset").loc[NAMES]
    cost = t.hydra_features + t.hydra_fit_predict  # extra seconds on one core for the equal-weights ensemble
    quant_min = (t.quant_features + t.quant_fit_predict).sum() / 60
    s = STATS.loc[NAMES]
    say(f"one core, resample 0: QUANT {quant_min:.1f} min; Hydra adds {cost.sum() / 60:.1f} min "
        f"({t.hydra_features.sum() / 60:.1f} of it computing features)")
    test_part = (t.hydra_features * s.test / (s.train + s.test)).sum() / t.hydra_features.sum()
    say(f"test series are {s.test.sum() / (s.train.sum() + s.test.sum()):.0%} of all series and take {test_part:.0%} of "
        f"Hydra's feature time")
    order = gain.sort_values(ascending=False).index
    for k in (20, 40):
        top = order[:k]
        say(f"  the {k} datasets with the largest gains: {gain[top].sum() / gain.sum():.0%} of the total gain, "
            f"{cost[top].sum() / cost.sum():.0%} of Hydra's extra time (found with test results: an upper bound)")
    per_class = s.train / s.classes
    rules = {"classes >= 10": s.classes >= 10, "<= 10 training series per class": per_class <= 10,
             "classes >= 10 or <= 10 per class": (s.classes >= 10) | (per_class <= 10)}
    say("Simple rules that use only the training data's size (QUANT accuracy 10 resamples: "
        f"{100 * e.QUANT.loc[NAMES].mean():.2f}%, full ensemble {100 * e['QUANT + Hydra (equal weights)'].loc[NAMES].mean():.2f}%):")
    for rule, m in rules.items():
        acc = np.where(m, e["QUANT + Hydra (equal weights)"].loc[NAMES], e.QUANT.loc[NAMES]).mean()
        say(f"  Hydra only if {rule:34s}: {m.sum():3d} datasets, accuracy {100 * acc:.2f}%, keeps "
            f"{gain[m].sum() / gain.sum():.0%} of the gain, total {quant_min + cost[m].sum() / 60:.1f} min on one core")
    say()


def elastic_section(acc):
    say("== 4. Where an elastic method beats QUANT (idea 14: speed-invariant quantiles) ==")
    d = (acc.PF - acc.QUANT).loc[NAMES]
    better = d > 0.01
    types = STATS.type.reindex(NAMES).fillna("new (no type)")
    say(f"Proximity Forest (elastic distances such as DTW) beats QUANT by > 1 point on {better.sum()} of 142 datasets; "
        f"mean gap there {100 * d[better].mean():+.1f} pts")
    say("  by type: " + ", ".join(f"{k} {v}" for k, v in types[better].value_counts().items()))
    say("  share of each type where PF wins: " + ", ".join(
        f"{k} {better[types == k].mean():.0%}" for k in ["Motion", "ECG", "Image", "Sensor", "Spectro", "Device"]))
    for name in d.sort_values(ascending=False).index[:12]:
        say(f"  {name:32s} PF - QUANT {100 * d[name]:+5.1f} pts  {types[name]}")
    say()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per = measure()
    acc = pd.DataFrame({m: bakeoff(m) for m in ["QUANT", "MR-Hydra", "HC2", "PF"]}).loc[NAMES]
    say("Read-only checks for the efficient approaches in ideas.md (bake-off redux: mean of 30 resamples).")
    say()
    cycles_section(per, acc)
    splits_section(per, acc)
    gating_section()
    elastic_section(acc)
    (OUT / "checks.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
