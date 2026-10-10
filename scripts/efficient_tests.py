"""Tests of the five efficient approaches (ideas.md: ideas 14-18).

  python scripts/efficient_tests.py main     # ideas 14-18 on all 142 datasets (first 10 resamples)
  python scripts/efficient_tests.py warp     # idea 14 in a speed stress test: time-warped series (30 datasets)
  python scripts/efficient_tests.py shift    # ideas 15 and 16 in the misalignment stress test (30 datasets)
  python scripts/efficient_tests.py timing   # single-core compute time of every part (original split)

Same rules as scripts/improvement_tests.py: QUANT itself is never changed, every run uses the official resamples
and seed = resample index, and finished datasets are skipped when a test is re-run. The main test also runs plain
QUANT, because idea 18 needs its probabilities; its accuracies must equal the reproduction's. Results go to
results/efficient/.
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import StratifiedKFold

import improvement_tests
from improvement_tests import evaluate, forest, hydra_scores, proba, softmax, timed
from quant_extensions import bispectrum_view, cycle_view, speed_quantiles
from weakness_tests import (LEVELS, NAMES, ROOT, Dataset, SparseScaler, append, finished, hydra_features, misalign,
                            quant_features)
from hydra import Hydra  # noqa: E402  (on the path once weakness_tests is imported)

warnings.filterwarnings("ignore", message="The least populated class")

THRESHOLDS = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.01]  # ask Hydra when QUANT's margin is below
TEMPERATURES = np.logspace(np.log10(0.05), 1, 24)  # softmax temperatures tried for Hydra's scores, as in idea 3
WARPS = [0.0, 0.2, 0.4, 0.6]  # speed stress test: largest change of log-speed (see warp)


def shift_datasets():
    """The 30 datasets of the misalignment stress test (weakness_tests.py shift)."""
    return sorted(pd.read_csv(ROOT / "results" / "weaknesses" / "shift.csv").dataset.unique())


# == idea 17: rank-uniform splits =====================================================================

def rank_scale(F, train, test, chunk=2048):
    """Every feature as its rank among the training series (0 = smallest, 1 = largest; tied values share their
    middle rank); test values are interpolated between the training values. The change is monotone, so the forest
    can make exactly the same splits, but ExtraTrees' random split points, drawn uniformly between a feature's
    smallest and largest value, now fall evenly among the series instead of mostly into the gap left by a few
    extreme values. Vectorised over features (scikit-learn's QuantileTransformer loops over them in Python)."""
    out_train, out_test = np.empty(F[train].shape, np.float32), np.empty(F[test].shape, np.float32)
    n = len(train)
    for a in range(0, F.shape[1], chunk):
        S = torch.from_numpy(np.ascontiguousarray(F[train, a:a + chunk].T)).sort(-1).values  # (features, n)
        for rows, out in ((train, out_train), (test, out_test)):
            V = torch.from_numpy(np.ascontiguousarray(F[rows, a:a + chunk].T))
            lo, hi = torch.searchsorted(S, V, side="left"), torch.searchsorted(S, V, side="right")
            i = lo.clamp(1, n - 1)
            s0, s1 = S.gather(1, i - 1), S.gather(1, i)
            between = (i - 1) + ((V - s0) / (s1 - s0).clamp_min(1e-30)).clamp(0, 1)
            rank = torch.where(hi > lo, (lo + hi - 1) / 2, between)
            out[:, a:a + chunk] = (rank / (n - 1)).T.numpy()
    return out_train, out_test


# == idea 18: Hydra only where QUANT is unsure ========================================================

def hydra_small(X, seed=0):
    """Hydra with 16 instead of 64 groups of kernels: about a quarter of the cost."""
    with torch.no_grad():
        return Hydra(X.shape[-1], g=16, seed=seed).batch(torch.from_numpy(X).unsqueeze(1))


def hydra_loo(Z, y, train, test, num_classes):
    """Hydra as in its reference example (sparse scaling + ridge), with probabilities calibrated at no extra cost:
    the ridge classifier's alpha search already computes every training series' leave-one-out score (exactly), and
    the softmax temperature is fitted on those scores instead of on idea 3's cross-validation.
    Returns (probabilities for the test series, temperature)."""
    train_t, test_t = torch.as_tensor(train), torch.as_tensor(test)
    scaler = SparseScaler()
    Z_train, Z_test = scaler.fit_transform(Z[train_t]).numpy(), scaler.transform(Z[test_t]).numpy()
    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), store_cv_results=True).fit(Z_train, y[train])
    best = int(np.argmin(np.abs(clf.alphas - clf.alpha_)))
    s_train, s_test = clf.decision_function(Z_train), clf.decision_function(Z_test)
    two = s_test.ndim == 1  # two classes: one score, positive for the second class
    if two:
        s_train, s_test = s_train[:, None], s_test[:, None]
        target = np.where(y[train] == clf.classes_[1], 1.0, -1.0)[:, None]
    else:
        target = np.where(y[train][:, None] == clf.classes_[None, :], 1.0, -1.0)
    # leave-one-out score = target - leave-one-out residual; the residual's size is stored by the alpha search and
    # its sign is that of the ordinary (in-sample) residual
    loo = target - np.sign(target - s_train) * np.sqrt(clf.cv_results_[:, :, best])
    if two:
        loo, s_test = np.hstack([-loo, loo]), np.hstack([-s_test, s_test])
    truth = np.searchsorted(clf.classes_, y[train])
    nll = [-np.log(np.clip(softmax(loo, T)[np.arange(len(train)), truth], 1e-12, 1)).mean() for T in TEMPERATURES]
    T = TEMPERATURES[int(np.argmin(nll))]
    scores = np.full((len(test), num_classes), -1e9)
    scores[:, clf.classes_] = s_test
    return softmax(scores, T), T


def gate(p_quant, p_hydra, y_test):
    """Average QUANT's and Hydra's probabilities only for the test series where QUANT is unsure: the gap between its
    two most likely classes is below the threshold. Threshold 0 is plain QUANT, above 1 the full ensemble."""
    top2 = np.sort(p_quant, 1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    return [{"threshold": t, "asked": (margin < t).mean(),
             "accuracy": ((p_quant + (margin < t)[:, None] * p_hydra).argmax(1) == y_test).mean()}
            for t in THRESHOLDS]


# == all five ideas on the 142 datasets ===============================================================

def test_main(args):
    out = args.out / "efficient.csv"
    done = finished(out)
    for name in args.datasets or NAMES:
        if name in done:
            continue
        d = Dataset(name)
        C = d.y.max() + 1
        F, t_quant = timed(quant_features, d.X)
        views = {"QUANT + speed quantiles": timed(speed_quantiles, d.X),  # idea 14
                 "QUANT + cycle view": timed(cycle_view, d.X),  # idea 15
                 "QUANT + bispectrum view": timed(bispectrum_view, d.X)}  # idea 16
        views = {method: (np.hstack([F, V]), t) for method, (V, t) in views.items()}
        hydras = {"Hydra": timed(hydra_features, d.X), "Hydra-16": timed(hydra_small, d.X)}  # idea 18
        rows = []
        for i in range(args.resamples or 10):
            train, test = d.split(i)
            base = {"dataset": name, "resample": i}
            t0 = time.perf_counter()
            p_quant = proba(forest(i).fit(F[train], d.y[train]), F[test], C)
            rows.append({**base, "method": "QUANT", "accuracy": (p_quant.argmax(1) == d.y[test]).mean(),
                         "fit_seconds": time.perf_counter() - t0, "transform_seconds": t_quant,
                         "num_features": F.shape[1]})
            for method, (G, t) in views.items():
                rows.append({**base, **evaluate(method, G, d.y, train, test, i), "transform_seconds": t_quant + t})
            t0 = time.perf_counter()  # idea 17
            R_train, R_test = rank_scale(F, train, test)
            prediction = forest(i).fit(R_train, d.y[train]).predict(R_test)
            rows.append({**base, "method": "QUANT, rank-uniform splits", "accuracy": (prediction == d.y[test]).mean(),
                         "fit_seconds": time.perf_counter() - t0, "transform_seconds": t_quant,
                         "num_features": F.shape[1]})
            for label, (Z, t) in hydras.items():
                t0 = time.perf_counter()
                p_hydra, T = hydra_loo(Z, d.y, train, test, C)
                rows.append({**base, "method": label, "accuracy": (p_hydra.argmax(1) == d.y[test]).mean(),
                             "fit_seconds": time.perf_counter() - t0, "transform_seconds": t, "temperature": T})
                rows += [{**base, "method": f"QUANT + {label}, gated", **g, "transform_seconds": t}
                         for g in gate(p_quant, p_hydra, d.y[test])]
        append(rows, out)
        print(f"main     {name}: done", flush=True)


# == stress tests =====================================================================================

def warp(X, strength, rng):
    """Play every series faster at the start and slower at the end, or the reverse, by a random amount: time
    t -> t^g (t from 0 to 1), g = e^u with u uniform in [-strength, strength]. Start, end and length stay."""
    if strength == 0:
        return X
    length = X.shape[1]
    u = np.linspace(0, 1, length)
    out = np.empty_like(X)
    for j, g in enumerate(np.exp(rng.uniform(-strength, strength, len(X)))):
        out[j] = np.interp(u ** g * (length - 1), np.arange(length), X[j])
    return out


def test_warp(args):
    """Idea 14 on time-warped series. Level 0 (unchanged data) is covered by the main test."""
    out = args.out / "warp.csv"
    done = finished(out)
    for name in args.datasets or shift_datasets():
        if name in done:
            continue
        d = Dataset(name)
        rows = []
        for li, level in enumerate(WARPS):
            if level == 0:
                continue
            X = warp(d.X, level, np.random.default_rng([NAMES.index(name), li]))
            F = quant_features(X)
            variants = {"QUANT": F, "QUANT + speed quantiles": np.hstack([F, speed_quantiles(X)])}
            for i in range(args.resamples or 5):
                train, test = d.split(i)
                rows += [{"dataset": name, "resample": i, "level": level, **evaluate(m, V, d.y, train, test, i)}
                         for m, V in variants.items()]
        append(rows, out)
        print(f"warp     {name}: done", flush=True)


def test_shift(args):
    """Ideas 15 and 16 on the same misaligned data as weakness_tests.py shift (same datasets, offsets and seeds),
    so QUANT's and Hydra's results there are the reference. Level 0 is covered by the main test."""
    out = args.out / "shift.csv"
    done = finished(out)
    for name in args.datasets or shift_datasets():
        if name in done:
            continue
        d = Dataset(name)
        rows = []
        for li, level in enumerate(LEVELS):
            if level == 0:
                continue
            X = misalign(d.X, level, np.random.default_rng([NAMES.index(name), li]))
            F = quant_features(X)
            variants = {"QUANT + cycle view": np.hstack([F, cycle_view(X)]),
                        "QUANT + bispectrum view": np.hstack([F, bispectrum_view(X)])}
            for i in range(args.resamples or 5):
                train, test = d.split(i)
                rows += [{"dataset": name, "resample": i, "level": level, **evaluate(m, V, d.y, train, test, i)}
                         for m, V in variants.items()]
        append(rows, out)
        print(f"shift    {name}: done", flush=True)


# == compute time on one core ========================================================================

def test_timing(args):
    """Compute time on one CPU core for the original train/test split of every dataset, measured like the QUANT
    paper's timing. Each step is timed on its own, so the cost of every idea can be added up from its parts."""
    from threadpoolctl import threadpool_limits
    improvement_tests.N_JOBS = 1
    torch.set_num_threads(1)
    threadpool_limits(1)
    out = args.out / "timing_1core.csv"
    done = finished(out)
    for name in args.datasets or NAMES:
        if name in done:
            continue
        d = Dataset(name)
        C = d.y.max() + 1
        train, test = d.split(0)
        row = {"dataset": name}
        F, row["quant_features"] = timed(quant_features, d.X)
        S, row["speed_features"] = timed(speed_quantiles, d.X)
        Y, row["cycle_features"] = timed(cycle_view, d.X)
        B, row["bispectrum_features"] = timed(bispectrum_view, d.X)
        Z, row["hydra_features"] = timed(hydra_features, d.X)
        Z16, row["hydra16_features"] = timed(hydra_small, d.X)
        for key, G in (("quant", F), ("speed", np.hstack([F, S])), ("cycle", np.hstack([F, Y])),
                       ("bispectrum", np.hstack([F, B]))):
            t0 = time.perf_counter()
            row[f"{key}_accuracy"] = (forest(0).fit(G[train], d.y[train]).predict(G[test]) == d.y[test]).mean()
            row[f"{key}_fit_predict"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        R_train, R_test = rank_scale(F, train, test)
        row["rank_accuracy"] = (forest(0).fit(R_train, d.y[train]).predict(R_test) == d.y[test]).mean()
        row["rank_fit_predict"] = time.perf_counter() - t0  # including the rank transform
        for key, Zx in (("hydra", Z), ("hydra16", Z16)):
            (p, _), row[f"{key}_fit_predict"] = timed(hydra_loo, Zx, d.y, train, test, C)  # incl. calibration
            row[f"{key}_accuracy"] = (p.argmax(1) == d.y[test]).mean()
        # what idea 3 spent on calibrating Hydra by cross-validation (its out-of-fold Hydra scores)
        k = int(min(5, max(2, np.bincount(d.y[train]).min())))
        t0 = time.perf_counter()
        for fit, check in StratifiedKFold(k, shuffle=True, random_state=0).split(train, d.y[train]):
            hydra_scores(Z, d.y, train[fit], train[check], C)
        row["hydra_cv_calibration"] = time.perf_counter() - t0
        append([row], out)
        print(f"timing   {name}: done", flush=True)


TESTS = {"main": test_main, "warp": test_warp, "shift": test_shift, "timing": test_timing}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("tests", nargs="+", choices=list(TESTS))
    p.add_argument("--datasets", nargs="*", help="only these datasets (default: the test's own selection)")
    p.add_argument("--resamples", type=int, help="override the number of resamples")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "efficient")
    args = p.parse_args()
    for t in args.tests:
        TESTS[t](args)
