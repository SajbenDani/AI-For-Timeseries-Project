"""Do the gains stack? Idea 18 (gated QUANT + Hydra) on top of ideas 1 (pooled quantiles) and 14 (speed quantiles).

  python scripts/stacking_tests.py main      # all 142 datasets (first 10 resamples)
  python scripts/stacking_tests.py summary   # -> results/stacking/summary.txt

Every split trains, on the same machine and with the same seeds:
  * plain QUANT, and QUANT with the features of ideas 1 and 14 added (code from quant_extensions.py, unchanged);
  * Hydra (efficient_tests.py's leave-one-out calibration), asked about the test series where each of the two
    forests is unsure (efficient_tests.gate, every threshold).
The summary compares idea 18 and the stacked version at one fixed two-level gate setting (gap < 0.4, Hydra used only
when at least 10% of a dataset's test series are unsure; inside the range chosen in report.md), so both are scored
exactly the same way; plain QUANT and idea 18 are re-run here rather than read from efficient.csv. This setting was
fixed after round 2, not picked on held-out datasets as in efficient_analysis.py, so idea 18 scores 87.22% here
against 87.16% there. The stacked version is also compared with MultiRocket+Hydra and HIVE-COTE 2 (bake-off redux,
same resamples).
Finished datasets are skipped when the test is re-run; a dataset that fails is reported and skipped.
"""

import argparse
import time
import traceback
import warnings

import numpy as np
import pandas as pd

from analysis_utils import KEY, bakeoff, paired, two_level
from quant_extensions import pooled_features, speed_quantiles
from weakness_tests import (
    NAMES,
    ROOT,
    Dataset,
    append,
    finished,
    hydra_features,
    quant_features,
)
from improvement_tests import forest, proba
from efficient_tests import gate, hydra_loo

warnings.filterwarnings("ignore")

OUT = ROOT / "results" / "stacking"
THRESHOLD, MIN_ASKED = 0.4, 0.1
STACKED = "QUANT + pooled + speed quantiles"


def test_main(args):
    out = OUT / "stacking.csv"
    done = finished(out)
    for name in args.datasets or NAMES:
        if name in done:
            continue
        try:
            t_start = time.perf_counter()
            d = Dataset(name)
            C = int(d.y.max()) + 1
            F = quant_features(d.X)
            G = np.hstack([F, pooled_features(d.X), speed_quantiles(d.X)])
            Z = hydra_features(d.X)
            rows = []
            for i in range(args.resamples or 10):
                train, test = d.split(i)
                base = {"dataset": name, "resample": i}
                p_hydra, _ = hydra_loo(Z, d.y, train, test, C)
                for method, H in (("QUANT", F), (STACKED, G)):
                    p = proba(forest(i).fit(H[train], d.y[train]), H[test], C)
                    rows.append(
                        {
                            **base,
                            "method": method,
                            "accuracy": (p.argmax(1) == d.y[test]).mean(),
                            "num_features": H.shape[1],
                        }
                    )
                    rows += [
                        {**base, "method": f"{method} + Hydra, gated", **g}
                        for g in gate(p, p_hydra, d.y[test])
                    ]
            append(rows, out)
            print(
                f"main {name}: done in {time.perf_counter() - t_start:.0f} s",
                flush=True,
            )
        except Exception:
            print(f"main {name}: FAILED, skipped\n{traceback.format_exc()}", flush=True)


def test_summary(args):
    df = pd.read_csv(OUT / "stacking.csv")
    q = df[df["method"] == "QUANT"].set_index(KEY)["accuracy"]
    s = df[df["method"] == STACKED].set_index(KEY)["accuracy"]
    methods = {
        "QUANT": q,
        "Ideas 1 + 14 (pooled + speed quantiles)": s,
        "Idea 18 (QUANT + Hydra, gated)": two_level(df, "QUANT + Hydra, gated", q, THRESHOLD, MIN_ASKED),
        "Ideas 1 + 14 + 18 (stacked)": two_level(df, f"{STACKED} + Hydra, gated", s, THRESHOLD, MIN_ASKED),
    }
    per = pd.DataFrame(
        {m: v.groupby(level="dataset").mean() for m, v in methods.items()}
    ).dropna()

    def vs(a, b):
        diff, w, d, l_, p = paired(per[a], per[b])
        return f"vs {b.split(' (')[0]}: {100 * diff:+.2f} pts, {w}/{d}/{l_}, p={p:.2g}"

    lines = [
        f"Do the gains stack? {len(per)} datasets x {q.index.get_level_values('resample').nunique()} resamples; "
        f"two-level gate at gap < {THRESHOLD}, Hydra only if >= {MIN_ASKED:.0%} of test series unsure",
        "",
    ]
    for m in per:
        line = f"{m:<42} {100 * per[m].mean():6.2f}%"
        if m != "QUANT":
            line += "   " + vs(m, "QUANT")
        if m == "Ideas 1 + 14 + 18 (stacked)":
            line += "   " + vs(m, "Idea 18 (QUANT + Hydra, gated)")
        lines.append(line)
    for label, method in (
        ("idea 18", "QUANT + Hydra, gated"),
        ("stacked", f"{STACKED} + Hydra, gated"),
    ):
        g = df[(df["method"] == method) & np.isclose(df["threshold"], THRESHOLD)]
        used = g["asked"] >= MIN_ASKED
        lines.append(
            f"Hydra asked ({label}): {g['asked'].where(used, 0).mean():.0%} of test series, "
            f"used on {used.groupby(g['dataset']).mean().mean():.0%} of datasets"
        )
    resamples = q.index.get_level_values("resample").nunique()
    for label, method in (("MultiRocket+Hydra", "MR-Hydra"), ("HIVE-COTE 2", "HC2")):
        other = bakeoff(method, resamples)[per.index]
        diff, w, d, l_, p = paired(per["Ideas 1 + 14 + 18 (stacked)"], other)
        lines.append(f"Stacked vs {label} (bake-off, same {resamples} resamples, {100 * other.mean():.2f}%): "
                     f"{100 * diff:+.2f} pts, {w}/{d}/{l_}, p={p:.2g}")
    d = per["Ideas 1 + 14 + 18 (stacked)"] - per["Idea 18 (QUANT + Hydra, gated)"]
    lines += [
        "",
        "Stacked vs idea 18, largest gains (pts): "
        + ", ".join(
            f"{n} {100 * v:+.1f}"
            for n, v in d.sort_values(ascending=False).head(5).items()
        ),
        "Stacked vs idea 18, largest losses (pts): "
        + ", ".join(f"{n} {100 * v:+.1f}" for n, v in d.sort_values().head(5).items()),
    ]
    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


TESTS = {"main": test_main, "summary": test_summary}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("tests", nargs="+", choices=list(TESTS))
    p.add_argument(
        "--datasets", nargs="*", help="only these datasets (default: all 142)"
    )
    p.add_argument(
        "--resamples", type=int, help="override the number of resamples (default 10)"
    )
    args = p.parse_args()
    for t in args.tests:
        TESTS[t](args)
