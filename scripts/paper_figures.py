"""The summary figure of the paper: Papers/figures/main_results.pdf (and .png).

Usage: python scripts/paper_figures.py
Every number is read from the result files, through scripts/improvement_analysis.py (round 1: ideas 1, 2, 3, 5, 10)
and scripts/efficient_analysis.py (round 2: ideas 14-18), which are run first (their printed output is hidden).
(a) Mean accuracy against one-core compute time, both relative to plain QUANT measured in the same timing run.
(b) How many accuracy points each method loses when the same patterns appear at random positions, or when the series
    are played at random speeds (the stress tests).
"""

import contextlib
import io
import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9a9893"  # second opinion, new views of the series, QUANT itself
# MultiRocket+Hydra takes about 1 h 15 min against QUANT's under 15 min in the QUANT paper's comparison (bake-off
# timings, other hardware), so only its position relative to QUANT is meaningful
MR_HYDRA_RATIO = 75 / 15

with contextlib.redirect_stdout(io.StringIO()):
    r1 = runpy.run_path(str(ROOT / "scripts" / "improvement_analysis.py"))
    r2 = runpy.run_path(str(ROOT / "scripts" / "efficient_analysis.py"))
one = r1["results"]["benchmark"].set_index("method")
two = r2["results"]["benchmark"].set_index("method")
t1 = pd.read_csv(ROOT / "results" / "improvements" / "timing_1core_summary.csv", index_col=0).x_quant
t2 = pd.read_csv(ROOT / "results" / "efficient" / "timing_1core_summary.csv", index_col=0).x_quant
gate = r2["results"]["gating"]
q = r2["q10"][r2["NAMES"]].mean()
mr, hc2 = r2["bakeoff"]("MR-Hydra", 10)[r2["NAMES"]].mean(), r2["bakeoff"]("HC2", 10)[r2["NAMES"]].mean()

second = [  # (label, time vs QUANT, accuracy)
    ("gate, small Hydra", gate[("QUANT + Hydra-16, gated", "two-level gate")]["minutes"] / r2["quant_minutes"],
     gate[("QUANT + Hydra-16, gated", "two-level gate")]["accuracy"]),
    ("gate", gate[("QUANT + Hydra, gated", "two-level gate")]["minutes"] / r2["quant_minutes"],
     gate[("QUANT + Hydra, gated", "two-level gate")]["accuracy"]),
    ("equal weights", t2["QUANT + Hydra, equal weights, free calibration"],
     gate[("QUANT + Hydra, gated", "two-level gate")]["full"]),
    ("weighted", t1["QUANT + Hydra, weighted (idea 3)"], one.accuracy["QUANT + Hydra (weighted)"]),
]
views = [
    ("speed quantiles", t2["QUANT + speed quantiles (14)"], two.accuracy["QUANT + speed quantiles"]),
    ("pooled quantiles", t1["QUANT + pooled quantiles (idea 1)"], one.accuracy["QUANT + pooled quantiles"]),
    ("conv. quantiles", t1["QUANT + conv quantiles (idea 2)"], one.accuracy["QUANT + conv quantiles"]),
]

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIX Two Text", "DejaVu Serif"],
                     "font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.16, 2.6), gridspec_kw={"width_ratios": [1.15, 1]})


def tidy(axis, title):
    axis.set_title(title, fontsize=8.5, color=INK, loc="left")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(length=0)
    axis.set_axisbelow(True)


# (a) accuracy against cost --------------------------------------------------------------------------------------
ax.axhline(100 * hc2, color=MUTED, lw=0.9, ls="--")
ax.text(0.95, 100 * hc2 + 0.05, f"HIVE-COTE 2: {100 * hc2:.2f}% (several weeks)", ha="left", va="bottom", fontsize=6.5,
        color=MUTED)
ax.plot(MR_HYDRA_RATIO, 100 * mr, "D", ms=6, mfc="white", mec=INK, mew=1.1, zorder=4)
ax.annotate(f"MultiRocket+Hydra\n{100 * mr:.2f}% (about 5×)", (MR_HYDRA_RATIO, 100 * mr), xytext=(0, -24),
            textcoords="offset points", ha="center", fontsize=6.5, color=INK)
x, y = zip(*[(c, 100 * a) for _, c, a in second])
ax.plot(x, y, "-", color=BLUE, lw=1.2, zorder=2)
ax.plot(x, y, "o", ms=6, color=BLUE, mec="white", mew=1, zorder=3, label="QUANT + Hydra (second opinion)")
offsets = {"gate, small Hydra": (7, -7), "gate": (-6, 6), "equal weights": (-4, 6), "weighted": (2, -12)}
for label, c, a in second:
    dx, dy = offsets[label]
    ax.annotate(label, (c, 100 * a), xytext=(dx, dy), textcoords="offset points", fontsize=6.5, color=INK,
                ha="left" if dx > 0 else "right")
for label, c, a in views:
    ax.plot(c, 100 * a, "s", ms=5.5, color=ORANGE, mec="white", mew=1, zorder=3)
    ax.annotate(label, (c, 100 * a), xytext=(6, -3), textcoords="offset points", fontsize=6.5, color=INK)
ax.plot([], [], "s", ms=5.5, color=ORANGE, mec="white", label="QUANT + new view of the series")
ax.plot(1, 100 * q, "o", ms=6, color=GREY, mec="white", mew=1, zorder=3)
ax.annotate("QUANT", (1, 100 * q), xytext=(6, -4), textcoords="offset points", fontsize=6.5, color=INK)
ax.set_xscale("log")
ax.set_xticks([1, 1.5, 2, 3, 4, 6], ["1×", "1.5×", "2×", "3×", "4×", "6×"])
ax.minorticks_off()
ax.set_xlim(0.9, 7.8)
ax.set_ylim(85.0, 87.9)
ax.set_xlabel("compute time on one CPU core, relative to QUANT")
ax.set_ylabel("mean accuracy, 142 datasets (%)")
ax.grid(color=GRID, lw=0.6)
ax.legend(frameon=False, fontsize=6.5, loc="center left", bbox_to_anchor=(0.0, 0.42))
tidy(ax, "(a) Accuracy against cost")

# (b) robustness ---------------------------------------------------------------------------------------------------
s1, s2, w = r1["results"]["shift"], r2["results"]["shift"], r2["results"]["warp"]
lost = lambda c, m: 100 * (c[m].iloc[0] - c[m].iloc[-1])  # noqa: E731
rows = [("QUANT", lost(s1, "QUANT"), GREY), ("Hydra (reference)", lost(s1, "Hydra"), "white"),
        ("QUANT + Hydra", lost(s1, "QUANT + Hydra (weighted)"), BLUE),
        ("+ conv. quantiles", lost(s1, "QUANT + conv quantiles"), ORANGE),
        ("+ augmentation", lost(s1, "QUANT + augmentation"), ORANGE),
        ("+ pooled quantiles", lost(s1, "QUANT + pooled quantiles"), ORANGE),
        ("+ cycle view", lost(s2, "QUANT + cycle view"), ORANGE),
        ("+ bispectrum view", lost(s2, "QUANT + bispectrum view"), ORANGE),
        "Random speed changes (strongest warp)",
        ("QUANT", lost(w, "QUANT"), GREY), ("+ speed quantiles", lost(w, "QUANT + speed quantiles"), ORANGE)]
rows.insert(0, "Patterns at random positions (50% longer window)")
y = 0
ticks, labels = [], []
for row in rows:
    if isinstance(row, str):  # group heading
        y += 0.35 if y else 0
        bx.text(0.05, y, row, ha="left", va="center", fontsize=6.8, color=MUTED, style="italic")
        y += 1
        continue
    label, value, color = row
    bx.barh(y, value, 0.68, color=color, edgecolor=INK if color == "white" else color, lw=0.8)
    bx.text(value + 0.12, y, f"{value:.1f}", va="center", fontsize=6.5, color=INK)
    ticks.append(y)
    labels.append(label)
    y += 1
bx.set_yticks(ticks, labels, fontsize=6.8)
bx.set_ylim(y - 0.4, -1.0)
bx.set_xlim(0, 11.2)
bx.set_xlabel("accuracy points lost (smaller is more robust)")
bx.grid(axis="x", color=GRID, lw=0.6)
tidy(bx, "(b) Robustness (30 datasets, stress tests)")

fig.tight_layout(w_pad=1.5)
out = ROOT / "Papers" / "figures" / "main_results.pdf"
fig.savefig(out)
fig.savefig(out.with_suffix(".png"), dpi=220)
print(f"wrote {out}")
print("panel (a):", [(n, round(float(c), 2), round(100 * float(a), 2)) for n, c, a in second + views])
print("panel (b):", [(r[0], round(float(r[1]), 1)) for r in rows if not isinstance(r, str)])
