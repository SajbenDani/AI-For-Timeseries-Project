"""Idea 19 (ideas.md): cross-channel views for multivariate QUANT.

  python scripts/cross_channel_tests.py check     # sanity check of the views -> results/cross_channel/check.txt
  python scripts/cross_channel_tests.py main      # QUANT + views, and idea 18 on top, 19 UEA datasets x 5
  python scripts/cross_channel_tests.py summary   # -> results/cross_channel/summary.txt, vs results/multivariate
  python scripts/cross_channel_tests.py rotate          # rotation stress test: all 5 methods, 6 sensor datasets tilted
  python scripts/cross_channel_tests.py rotate_summary  # -> results/cross_channel/rotate_summary.txt

QUANT computes its features on every channel alone, so it never sees how channels move together, and a sensor held at
another angle changes every channel. This adds new signals built from several channels, each described by QUANT's own
windows and quantiles (its code is not changed):
  * for every 3-axis sensor (x, y, z): its magnitude sqrt(x^2 + y^2 + z^2), and the signal along the sensor's own main
    direction of movement (first principal axis of that series). Both stay the same when the sensor is rotated.
  * for all datasets: differences between pairs of channels (each channel first standardised per series, so the
    difference shows how they move relative to each other; at most 15 fixed random pairs), and the signal along the
    main direction of all (standardised) channels.
Everything is computed per series and never looks at the labels, so, like QUANT, it is computed once for all series.
Same splits, seeds and forest as scripts/multivariate_tests.py, so the results compare directly with QUANT and
idea 18 there.
A dataset that fails is reported and skipped, so one error never stops the run.
"""

import argparse
import itertools
import time
import traceback
import warnings

import numpy as np
import pandas as pd

from analysis_utils import KEY, paired
from analysis_utils import two_level as gate_at
from multivariate_tests import (
    MIN_ASKED,
    NAMES,
    OUT as MV_OUT,
    ROOT,
    THRESHOLD,
    UEADataset,
    hydra_features,
    quant_features,
)
from weakness_tests import append, finished
from improvement_tests import forest, proba, timed
from efficient_tests import gate, hydra_loo

warnings.filterwarnings("ignore")

OUT = ROOT / "results" / "cross_channel"

# datasets whose channels are the x, y, z axes of 3-axis sensors, in this order (UEA archive descriptions)
SENSOR_TRIPLES = {
    "BasicMotions": [(0, 1, 2), (3, 4, 5)],  # accelerometer, gyroscope
    "Cricket": [(0, 1, 2), (3, 4, 5)],  # left wrist, right wrist accelerometers
    "Epilepsy": [(0, 1, 2)],  # wrist accelerometer
    "Handwriting": [(0, 1, 2)],  # wrist accelerometer
    "RacketSports": [(0, 1, 2), (3, 4, 5)],  # accelerometer, gyroscope
    "UWaveGestureLibrary": [(0, 1, 2)],  # accelerometer
}


# == views ============================================================================================


def standardise(X):
    """Every channel of every series to mean 0, standard deviation 1 (flat channels stay 0)."""
    s = X.std(-1, keepdims=True)
    return (X - X.mean(-1, keepdims=True)) / np.where(s > 1e-8, s, 1.0)


def channel_pairs(num_channels, max_pairs=15, seed=0):
    pairs = list(itertools.combinations(range(num_channels), 2))
    if len(pairs) > max_pairs:
        chosen = np.random.default_rng(seed).choice(
            len(pairs), max_pairs, replace=False
        )
        pairs = [pairs[i] for i in sorted(chosen)]
    return pairs


def main_direction(X):
    """Every series (n, k channels, length) projected on its own first principal axis. The sign is set so the
    projection's skewness is >= 0, a rule that does not depend on how the sensor is rotated.
    """
    Xc = X - X.mean(-1, keepdims=True)
    _, vectors = np.linalg.eigh(
        Xc @ Xc.transpose(0, 2, 1)
    )  # eigenvalues in ascending order
    p = np.einsum("nk,nkl->nl", vectors[:, :, -1], Xc)
    return p * np.where((p**3).sum(-1) >= 0, 1.0, -1.0)[:, None]


def sensor_signals(X, name):
    """Magnitude and main direction of every 3-axis sensor: both unchanged when the sensor is rotated."""
    signals = []
    for triple in SENSOR_TRIPLES.get(name, []):
        T = X[:, list(triple)].astype(np.float64)
        signals += [np.sqrt((T**2).sum(1)), main_direction(T)]
    return signals


def cross_channel_signals(X, name):
    """(n, channels, length) -> (n, new signals, length)."""
    Z = standardise(X.astype(np.float64))
    signals = sensor_signals(X, name) + [
        Z[:, i] - Z[:, j] for i, j in channel_pairs(X.shape[1])
    ]
    signals.append(main_direction(Z))
    return np.ascontiguousarray(np.stack(signals, 1), dtype=np.float32)


def cross_channel_features(X, name):
    """QUANT's features (its own code) of the new signals."""
    return quant_features(cross_channel_signals(X, name))


# == sanity check =====================================================================================


def random_rotations(n, rng):
    """n random 3-D rotations, uniformly distributed."""
    out = []
    for _ in range(n):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q = q * np.sign(np.diag(r))
        if np.linalg.det(q) < 0:
            q[:, 0] = -q[:, 0]
        out.append(q)
    return np.stack(out)


def rotations_up_to(n, max_degrees, rng):
    """n rotations about random axes, each by a random angle between 0 and max_degrees (Rodrigues' formula)."""
    axes = rng.normal(size=(n, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    angles = np.radians(rng.uniform(0, max_degrees, n))[:, None, None]
    K = np.zeros((n, 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -axes[:, 2], axes[:, 1], -axes[:, 0]
    K = K - K.transpose(0, 2, 1)
    return np.eye(3) + np.sin(angles) * K + (1 - np.cos(angles)) * K @ K


def rotate(X, name, rng, max_degrees=None):
    """Every series with its device held at a random angle: the same random rotation of each of its 3-axis sensors.
    max_degrees=None: any orientation (uniform); otherwise a random tilt of at most max_degrees.
    """
    R = (
        random_rotations(len(X), rng)
        if max_degrees is None
        else rotations_up_to(len(X), max_degrees, rng)
    )
    X = X.copy()
    for triple in SENSOR_TRIPLES[name]:
        X[:, list(triple)] = np.einsum("nij,njl->nil", R, X[:, list(triple)])
    return X


def change(a, b):
    """Relative change from a to b, in %."""
    return 100 * np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-12)


def test_check(args):
    lines = [
        "Sanity check of the cross-channel views (idea 19)",
        "",
        "1. Rotation: every series of the 3-axis sensor datasets rotated at random (seed 0).",
        "   Relative change (%) of: raw channels, QUANT's features, the rotation-proof signals (magnitude + main",
        "   direction of each sensor) and QUANT's features of those signals. Expected: large, large, ~0, ~0.",
        "",
    ]
    rows = []
    for name in SENSOR_TRIPLES:
        d = UEADataset(name)
        Xr = rotate(d.X, name, np.random.default_rng(0))
        S, Sr = np.stack(sensor_signals(d.X, name), 1), np.stack(
            sensor_signals(Xr, name), 1
        )
        rows.append(
            {
                "dataset": name,
                "raw channels": change(d.X, Xr),
                "QUANT features": change(quant_features(d.X), quant_features(Xr)),
                "rotation-proof signals": change(S, Sr),
                "their QUANT features": change(
                    quant_features(S.astype(np.float32)),
                    quant_features(Sr.astype(np.float32)),
                ),
            }
        )
    lines.append(pd.DataFrame(rows).set_index("dataset").round(3).to_string())
    lines += ["", "2. Every dataset: new signals and features, any NaN/inf, time.", ""]
    rows = []
    for name in args.datasets or NAMES:
        d = UEADataset(name)
        G, t = timed(cross_channel_features, d.X, name)
        rows.append(
            {
                "dataset": name,
                "channels": d.X.shape[1],
                "sensors": len(SENSOR_TRIPLES.get(name, [])),
                "new signals": cross_channel_signals(d.X, name).shape[1],
                "QUANT features": quant_features(d.X).shape[1],
                "new features": G.shape[1],
                "NaN/inf": int((~np.isfinite(G)).sum()),
                "seconds": round(t, 2),
            }
        )
    lines.append(pd.DataFrame(rows).set_index("dataset").to_string())
    text = "\n".join(lines)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "check.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


# == idea 19, and idea 18 on top ======================================================================


def test_main(args):
    out = OUT / "main.csv"
    done = finished(out)
    for name in args.datasets or NAMES:
        if name in done:
            continue
        try:
            t_start = time.perf_counter()
            d = UEADataset(name)
            C = int(d.y.max()) + 1
            F, t_quant = timed(quant_features, d.X)
            G, t_views = timed(cross_channel_features, d.X, name)
            H = np.hstack([F, G])
            Z, t_hydra = timed(hydra_features, d.X)
            rows = []
            for i in range(args.resamples or 5):
                train, test = d.split(i)
                base = {"dataset": name, "resample": i}
                t0 = time.perf_counter()
                p = proba(forest(i).fit(H[train], d.y[train]), H[test], C)
                rows.append(
                    {
                        **base,
                        "method": "QUANT + cross-channel views",
                        "accuracy": (p.argmax(1) == d.y[test]).mean(),
                        "fit_seconds": time.perf_counter() - t0,
                        "transform_seconds": t_quant + t_views,
                        "num_features": H.shape[1],
                    }
                )
                p_hydra, _ = hydra_loo(Z, d.y, train, test, C)
                rows += [
                    {
                        **base,
                        "method": "QUANT + cross-channel views + Hydra, gated",
                        **g,
                        "transform_seconds": t_hydra,
                    }
                    for g in gate(p, p_hydra, d.y[test])
                ]
            append(rows, out)
            print(
                f"main {name}: done in {time.perf_counter() - t_start:.0f} s",
                flush=True,
            )
        except Exception:
            print(f"main {name}: FAILED, skipped\n{traceback.format_exc()}", flush=True)


# == summary ==========================================================================================


def two_level(df, method, quant):
    """Idea 18's two-level gate at the setting used in multivariate_tests.py."""
    return gate_at(df, method, quant, THRESHOLD, MIN_ASKED)


def test_summary(args):
    old = pd.read_csv(MV_OUT / "main.csv")
    new = pd.read_csv(OUT / "main.csv")
    q = old[old["method"] == "QUANT"].set_index(KEY)["accuracy"]
    q4 = new[new["method"] == "QUANT + cross-channel views"].set_index(KEY)["accuracy"]
    methods = {
        "QUANT": q,
        "Idea 18 (QUANT + Hydra, gated)": two_level(old, "QUANT + Hydra, gated", q),
        "Idea 19 (QUANT + cross-channel views)": q4,
        "Ideas 18 + 19 (views + Hydra, gated)": two_level(
            new, "QUANT + cross-channel views + Hydra, gated", q4
        ),
    }
    per = pd.DataFrame(
        {m: s.groupby(level="dataset").mean() for m, s in methods.items()}
    ).dropna()
    sensors = per.index.isin(list(SENSOR_TRIPLES))
    lines = [
        f"Idea 19: cross-channel views. {len(per)} UEA datasets x 5 resamples (same splits and seeds as "
        f"multivariate_tests.py)",
        "",
    ]
    for label, rows in (
        ("All datasets", per),
        (f"The {sensors.sum()} 3-axis sensor datasets", per[sensors]),
    ):
        lines.append(f"{label}:")
        for m in per:
            line = f"  {m:<40} {100 * rows[m].mean():6.2f}%"
            for ref in ("QUANT", "Idea 18 (QUANT + Hydra, gated)"):
                if (
                    m != ref
                    and not (ref != "QUANT" and m.startswith("Idea 19"))
                    and m != "QUANT"
                    and len(rows) > 1
                ):
                    diff, w, dr, l_, p = paired(rows[m], rows[ref])
                    line += f"   vs {ref.split(' (')[0]}: {100 * diff:+.2f} pts, {w}/{dr}/{l_}, p={p:.2g}"
            lines.append(line)
        lines.append("")
    lines += ["Per dataset (%):", (100 * per).round(2).to_string()]
    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


# == rotation stress test =============================================================================

TILTS = [
    30,
    90,
    180,
]  # largest tilt of every series, in degrees (0 = the normal data, from the main runs)


def test_rotate(args):
    """All five methods on the 3-axis sensor datasets with every series tilted at random. Train and test series are
    both tilted, each by its own random rotation (seeded per dataset and tilt), so a method must recognise a movement
    at any angle. Same splits and seeds as the main runs, so the only change is the tilt.
    """
    out = OUT / "rotate.csv"
    done = finished(out)
    for name in args.datasets or list(SENSOR_TRIPLES):
        if name in done:
            continue
        try:
            t_start = time.perf_counter()
            d = UEADataset(name)
            C = int(d.y.max()) + 1
            rows = []
            for li, tilt in enumerate(TILTS):
                X = rotate(
                    d.X, name, np.random.default_rng([NAMES.index(name), li]), tilt
                ).astype(np.float32)
                F = quant_features(X)
                H = np.hstack([F, cross_channel_features(X, name)])
                Z = hydra_features(X)
                for i in range(args.resamples or 5):
                    train, test = d.split(i)
                    base = {"dataset": name, "resample": i, "tilt": tilt}
                    p_quant = proba(forest(i).fit(F[train], d.y[train]), F[test], C)
                    p_views = proba(forest(i).fit(H[train], d.y[train]), H[test], C)
                    p_hydra, _ = hydra_loo(Z, d.y, train, test, C)
                    for method, p in (
                        ("QUANT", p_quant),
                        ("Hydra", p_hydra),
                        ("QUANT + cross-channel views", p_views),
                    ):
                        rows.append(
                            {
                                **base,
                                "method": method,
                                "accuracy": (p.argmax(1) == d.y[test]).mean(),
                            }
                        )
                    rows += [
                        {**base, "method": "QUANT + Hydra, gated", **g}
                        for g in gate(p_quant, p_hydra, d.y[test])
                    ]
                    rows += [
                        {
                            **base,
                            "method": "QUANT + cross-channel views + Hydra, gated",
                            **g,
                        }
                        for g in gate(p_views, p_hydra, d.y[test])
                    ]
            append(rows, out)
            print(
                f"rotate {name}: done in {time.perf_counter() - t_start:.0f} s",
                flush=True,
            )
        except Exception:
            print(
                f"rotate {name}: FAILED, skipped\n{traceback.format_exc()}", flush=True
            )


def five_methods(old, new):
    """Per (dataset, resample) accuracy of the five methods; old has QUANT/Hydra/idea 18, new idea 19 and 18 + 19."""
    q = old[old["method"] == "QUANT"].set_index(KEY)["accuracy"]
    q4 = new[new["method"] == "QUANT + cross-channel views"].set_index(KEY)["accuracy"]
    return {
        "QUANT": q,
        "Hydra": old[old["method"] == "Hydra"].set_index(KEY)["accuracy"],
        "Idea 18 (QUANT + Hydra, gated)": two_level(old, "QUANT + Hydra, gated", q),
        "Idea 19 (QUANT + cross-channel views)": q4,
        "Ideas 18 + 19 (views + Hydra, gated)": two_level(
            new, "QUANT + cross-channel views + Hydra, gated", q4
        ),
    }


def test_rotate_summary(args):
    sensors = list(SENSOR_TRIPLES)
    old = pd.read_csv(MV_OUT / "main.csv")
    new = pd.read_csv(OUT / "main.csv")
    rot = pd.read_csv(OUT / "rotate.csv")
    by_tilt = {
        0: five_methods(
            old[old["dataset"].isin(sensors)], new[new["dataset"].isin(sensors)]
        )
    }
    for tilt in TILTS:
        r = rot[rot["tilt"] == tilt]
        by_tilt[tilt] = five_methods(r, r)
    per = {
        t: pd.DataFrame({m: s.groupby(level="dataset").mean() for m, s in ms.items()})
        for t, ms in by_tilt.items()
    }
    done = sorted(set(rot["dataset"]))
    table = pd.DataFrame({f"{t} deg": per[t].loc[done].mean() for t in per})
    table["lost (0 -> 180)"] = table["0 deg"] - table["180 deg"]
    lines = [
        f"Rotation stress test: {len(done)} 3-axis sensor datasets x 5 resamples, every series tilted at random by",
        "up to 30, 90 or 180 degrees (train and test). Mean accuracy (%).",
        "",
        (100 * table).round(2).to_string(),
        "",
    ]
    worst = per[TILTS[-1]].loc[done]
    for m in worst:
        if m != "QUANT" and len(done) > 1:
            diff, w, dr, l_, p = paired(worst[m], worst["QUANT"])
            lines.append(
                f"At 180 deg, {m} vs QUANT: {100 * diff:+.2f} pts, W/D/L {w}/{dr}/{l_}, Wilcoxon p={p:.2g}"
            )
    lines += ["", "Per dataset at 180 deg (%):", (100 * worst).round(2).to_string()]
    text = "\n".join(lines)
    (OUT / "rotate_summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


TESTS = {
    "check": test_check,
    "main": test_main,
    "summary": test_summary,
    "rotate": test_rotate,
    "rotate_summary": test_rotate_summary,
}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("tests", nargs="+", choices=list(TESTS))
    p.add_argument(
        "--datasets",
        nargs="*",
        help="only these datasets (default: results/multivariate/datasets.csv)",
    )
    p.add_argument(
        "--resamples", type=int, help="override the number of resamples (default 5)"
    )
    args = p.parse_args()
    for t in args.tests:
        TESTS[t](args)
