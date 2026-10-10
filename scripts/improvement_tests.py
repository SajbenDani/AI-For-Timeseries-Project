"""Tests of five ideas for improving QUANT (ideas.md: 1, 2, 3, 5 and 10).

  python scripts/improvement_tests.py varlen     # idea 10: the 11 variable-length datasets without padding (30 resamples)
  python scripts/improvement_tests.py varlen --datasets AsphaltObstaclesUniEq AsphaltPavementTypeUniEq \
      AsphaltRegularityUniEq --variants "QUANT (padded, original)" "QUANT (native windows)" \
      "QUANT (padded + native windows)" --out results/improvements/varlen_fresh
                                                 # idea 10 on 3 fresh variable-length datasets (originals in data/raw/native5)
  python scripts/improvement_tests.py pooled     # idea 1:  QUANT + position-free pooled window quantiles (142 datasets)
  python scripts/improvement_tests.py convq      # idea 2:  QUANT + quantiles of random convolutions (142 datasets)
  python scripts/improvement_tests.py ensemble   # idea 3:  QUANT + Hydra, probabilities averaged (142 datasets)
  python scripts/improvement_tests.py augment    # idea 5:  shifted / stretched training copies + test-time averaging
  python scripts/improvement_tests.py shift      # ideas 1, 2, 3, 5 in the misalignment stress test (30 datasets)
  python scripts/improvement_tests.py timing     # single-core compute time of QUANT and ideas 1, 2, 3 (original split)

QUANT itself is never changed: each idea adds features, training copies or a second model, and keeps QUANT's
forest (200 extremely randomised trees, max_features=0.1, entropy). Every run uses the official resamples and
seed = resample index, exactly as run_repro.py, so plain QUANT's results come from the reproduction. The
142-dataset tests use the first 10 of the 30 resamples to keep the cost manageable on a laptop. Results go to
results/improvements/; finished datasets are skipped when a test is re-run.
"""

import argparse
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import StratifiedKFold

from quant_extensions import AUGMENTATIONS, conv_quantiles, pooled_features, varlen_quant
from weakness_tests import (LEVELS, NAMES, ROOT, VARLEN, Dataset, SparseScaler, append, finished, hydra_features,
                            misalign, quant_features)

warnings.filterwarnings("ignore", message="The least populated class")


# == classifiers ======================================================================================

N_JOBS = -1  # all cores; the timing test sets 1


def forest(seed, n_estimators=200):
    """QUANT's classifier, configured as in the paper."""
    return ExtraTreesClassifier(n_estimators=n_estimators, max_features=0.1, criterion="entropy", n_jobs=N_JOBS,
                                random_state=seed)


def proba(clf, F, num_classes):
    """Class probabilities with one column per class, also for classes missing from the training data."""
    out = np.zeros((len(F), num_classes))
    out[:, clf.classes_] = clf.predict_proba(F)
    return out


def hydra_scores(Z, y, train, test, num_classes):
    """Hydra as in its reference example (sparse scaling + ridge); one score column per class."""
    train_t, test_t = torch.as_tensor(train), torch.as_tensor(test)
    scaler = SparseScaler()
    Z_train, Z_test = scaler.fit_transform(Z[train_t]).numpy(), scaler.transform(Z[test_t]).numpy()
    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(Z_train, y[train])
    s = clf.decision_function(Z_test)
    if s.ndim == 1:  # two classes: one score, positive for the second class
        s = np.column_stack([-s, s])
    out = np.full((len(test), num_classes), -1e9)
    out[:, clf.classes_] = s
    return out


def softmax(s, temperature):
    e = np.exp((s - s.max(1, keepdims=True)) / temperature)
    return e / e.sum(1, keepdims=True)


def evaluate(method, F, y, train, test, seed):
    t0 = time.perf_counter()
    clf = forest(seed).fit(F[train], y[train])
    t1 = time.perf_counter()
    return {"method": method, "accuracy": (clf.predict(F[test]) == y[test]).mean(), "fit_seconds": t1 - t0,
            "num_features": F.shape[1]}


def rows_for(name, i, rows, transform_seconds=np.nan):
    return [{"dataset": name, "resample": i, **r, "transform_seconds": transform_seconds} for r in rows]


def timed(function, *args):
    t0 = time.perf_counter()
    return function(*args), time.perf_counter() - t0


# == idea 10: variable-length QUANT ===================================================================

FRESH = {"AsphaltObstaclesUniEq": "AsphaltObstacles", "AsphaltPavementTypeUniEq": "AsphaltPavementType",
         "AsphaltRegularityUniEq": "AsphaltRegularity"}  # fresh check of idea 10: originals from
# timeseriesclassification.com (data/raw/native5/); each padded series starts with exactly its original


def native_series(name, labels):
    """The original, unpadded series of an "Eq" dataset, in the same pool order as the padded version."""
    series, native_labels = [], []
    for split in ("TRAIN", "TEST"):
        if name in FRESH:
            path = ROOT / "data" / "raw" / "native5" / FRESH[name] / f"{FRESH[name]}_{split}.ts"
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        else:
            native = name[:-2]  # e.g. AllGestureWiimoteXEq -> AllGestureWiimoteX
            archive = zipfile.ZipFile(ROOT / "data" / "raw" / "Univariate2018_ts.zip")
            lines = archive.read(f"Univariate_ts/{native}/{native}_{split}.ts").decode("utf-8", "ignore").splitlines()
        start = next(j for j, line in enumerate(lines) if line.strip().lower().startswith("@data")) + 1
        for line in filter(str.strip, lines[start:]):
            v, label = line.strip().rsplit(":", 1)
            series.append(np.array(v.split(","), dtype=np.float32))
            native_labels.append(label.lower())
    assert native_labels == [str(y).lower() for y in labels], f"{name}: native and padded versions differ"
    return series


def test_varlen(args):
    out = args.out / "varlen.csv"
    done = finished(out)
    for name in args.datasets or VARLEN:
        if name in done:
            continue
        d = Dataset(name)
        series = native_series(name, d.labels)
        L = d.X.shape[1]  # padded length = longest series
        length = np.array([[len(s)] for s in series], dtype=np.float32)
        stretched = lambda: np.stack([np.interp(np.linspace(0, len(s) - 1, L), np.arange(len(s)), s)  # noqa: E731
                                      for s in series]).astype(np.float32)
        native, t_native = timed(varlen_quant, series, L)
        padded = quant_features(d.X)
        variants = {  # built only if selected (--variants)
            "QUANT (padded, original)": lambda: padded,
            "QUANT (stretched to equal length) + length": lambda: np.hstack([quant_features(stretched()), length]),
            "QUANT (native windows)": lambda: native,
            "QUANT (native windows) + length": lambda: np.hstack([native, length]),
            "QUANT (padded + native windows)": lambda: np.hstack([padded, native]),  # the forest picks the view
        }
        variants = {m: make() for m, make in variants.items() if not args.variants or m in args.variants}
        rows = []
        for i in range(args.resamples or 30):
            train, test = d.split(i)
            rows += rows_for(name, i, [evaluate(m, F, d.y, train, test, i) for m, F in variants.items()], t_native)
        append(rows, out)
        print(f"varlen   {name}: done", flush=True)


# == ideas 1 and 2: extra features ====================================================================

def test_pooled(args):
    out = args.out / "pooled.csv"
    done = finished(out)
    for name in args.datasets or NAMES:
        if name in done:
            continue
        d = Dataset(name)
        P, t = timed(pooled_features, d.X)
        F = np.hstack([quant_features(d.X), P])
        rows = []
        for i in range(args.resamples or 10):
            train, test = d.split(i)
            rows += rows_for(name, i, [evaluate("QUANT + pooled quantiles", F, d.y, train, test, i)], t)
        append(rows, out)
        print(f"pooled   {name}: done", flush=True)


def test_convq(args):
    out = args.out / "convq.csv"
    done = finished(out)
    for name in args.datasets or NAMES:
        if name in done:
            continue
        d = Dataset(name)
        C, t = timed(conv_quantiles, d.X)
        F = np.hstack([quant_features(d.X), C])
        rows = []
        for i in range(args.resamples or 10):
            train, test = d.split(i)
            rows += rows_for(name, i, [evaluate("QUANT + conv quantiles", F, d.y, train, test, i),
                                       evaluate("conv quantiles alone", C, d.y, train, test, i)], t)
        append(rows, out)
        print(f"convq    {name}: done", flush=True)


# == idea 3: QUANT + Hydra ensemble ===================================================================

def ensemble(F, Z, y, train, test, seed):
    """QUANT's and Hydra's class probabilities, averaged. Hydra's ridge scores become probabilities through a
    softmax whose temperature is fitted on out-of-fold predictions of the training set. The same out-of-fold
    predictions estimate each model's accuracy, which sets its weight as in HIVE-COTE's CAWPE (accuracy^4)."""
    C = y.max() + 1
    t0 = time.perf_counter()
    p_quant = proba(forest(seed).fit(F[train], y[train]), F[test], C)
    t1 = time.perf_counter()
    s_hydra = hydra_scores(Z, y, train, test, C)
    t2 = time.perf_counter()

    # out-of-fold predictions on the training set (a 50-tree forest keeps this cheap)
    k = int(min(5, max(2, np.bincount(y[train]).min())))
    oof_quant, oof_hydra = np.zeros((len(train), C)), np.zeros((len(train), C))
    for fit, check in StratifiedKFold(k, shuffle=True, random_state=seed).split(train, y[train]):
        oof_quant[check] = proba(forest(seed, 50).fit(F[train[fit]], y[train[fit]]), F[train[check]], C)
        oof_hydra[check] = hydra_scores(Z, y, train[fit], train[check], C)
    y_train = y[train]
    seen = oof_hydra[np.arange(len(train)), y_train] > -1e8  # true class was in the fold's training part
    temperatures = np.logspace(np.log10(0.05), 1, 24)  # lower bound: a model that happens to be perfect on the
    # training folds would otherwise get near-0/1 probabilities and silence the other model completely
    nll = [-np.log(np.clip(softmax(oof_hydra[seen], T)[np.arange(seen.sum()), y_train[seen]], 1e-12, 1)).mean()
           for T in temperatures]
    T = temperatures[int(np.argmin(nll))]
    w_quant = (oof_quant.argmax(1) == y_train).mean() ** 4
    w_hydra = (oof_hydra.argmax(1) == y_train).mean() ** 4
    if w_quant + w_hydra == 0:
        w_quant = w_hydra = 1
    p_hydra = softmax(s_hydra, T)
    t3 = time.perf_counter()

    acc = lambda p: (p.argmax(1) == y[test]).mean()  # noqa: E731
    return [
        {"method": "QUANT", "accuracy": acc(p_quant), "fit_seconds": t1 - t0},
        {"method": "Hydra", "accuracy": acc(p_hydra), "fit_seconds": t2 - t1},
        {"method": "QUANT + Hydra (equal weights)", "accuracy": acc(p_quant + p_hydra), "fit_seconds": t2 - t0},
        {"method": "QUANT + Hydra (weighted)", "accuracy": acc(w_quant * p_quant + w_hydra * p_hydra),
         "fit_seconds": t3 - t0, "weight_quant": w_quant / (w_quant + w_hydra), "temperature": T},
    ]


def test_ensemble(args):
    out = args.out / "ensemble.csv"
    done = finished(out)
    for name in args.datasets or NAMES:
        if name in done:
            continue
        d = Dataset(name)
        Z, t = timed(hydra_features, d.X)
        F = quant_features(d.X)
        rows = []
        for i in range(args.resamples or 10):
            train, test = d.split(i)
            rows += rows_for(name, i, ensemble(F, Z, d.y, train, test, i), t)
        append(rows, out)
        print(f"ensemble {name}: done", flush=True)


# == idea 5: augmentation =============================================================================

def augmented(F, copies, y, train, test, seed):
    """Train on the series plus their shifted / stretched copies; test on the series alone and, for test-time
    augmentation, on the average of the probabilities for the series and its copies."""
    C = y.max() + 1
    t0 = time.perf_counter()
    clf = forest(seed).fit(np.vstack([F[train]] + [A[train] for A in copies]), np.tile(y[train], 1 + len(copies)))
    t1 = time.perf_counter()
    p = proba(clf, F[test], C)
    p_tta = p + sum(proba(clf, A[test], C) for A in copies)
    t2 = time.perf_counter()
    return [
        {"method": "QUANT + augmentation", "accuracy": (p.argmax(1) == y[test]).mean(), "fit_seconds": t1 - t0},
        {"method": "QUANT + augmentation + TTA", "accuracy": (p_tta.argmax(1) == y[test]).mean(),
         "fit_seconds": t2 - t0},
    ]


def test_augment(args):
    out = args.out / "augment.csv"
    done = finished(out)
    for name in args.datasets or NAMES:
        if name in done:
            continue
        d = Dataset(name)
        F = quant_features(d.X)
        copies, t = timed(lambda X: [quant_features(f(X)) for f in AUGMENTATIONS.values()], d.X)
        rows = []
        for i in range(args.resamples or 10):
            train, test = d.split(i)
            rows += rows_for(name, i, augmented(F, copies, d.y, train, test, i), t)
        append(rows, out)
        print(f"augment  {name}: done", flush=True)


# == the ideas in the misalignment stress test ========================================================

def test_shift(args):
    """Same misaligned data as weakness_tests.py shift (same datasets, offsets and seeds), so QUANT's and Hydra's
    results there are the reference. Level 0 is unchanged data, covered by the 142-dataset tests."""
    out = args.out / "shift.csv"
    done = finished(out)
    names = args.datasets or sorted(pd.read_csv(ROOT / "results" / "weaknesses" / "shift.csv").dataset.unique())
    for name in names:
        if name in done:
            continue
        d = Dataset(name)
        rows = []
        for li, level in enumerate(LEVELS):
            if level == 0:
                continue
            X = misalign(d.X, level, np.random.default_rng([NAMES.index(name), li]))
            F = quant_features(X)
            with_pooled = np.hstack([F, pooled_features(X)])
            with_conv = np.hstack([F, conv_quantiles(X)])
            copies = [quant_features(f(X)) for f in AUGMENTATIONS.values()]
            Z = hydra_features(X)
            for i in range(args.resamples or 5):
                train, test = d.split(i)
                results = [evaluate("QUANT + pooled quantiles", with_pooled, d.y, train, test, i),
                           evaluate("QUANT + conv quantiles", with_conv, d.y, train, test, i)]
                results += augmented(F, copies, d.y, train, test, i)
                results += [r for r in ensemble(F, Z, d.y, train, test, i) if r["method"] not in ("QUANT", "Hydra")]
                rows += [{**r, "level": level} for r in rows_for(name, i, results)]
        append(rows, out)
        print(f"shift    {name}: done", flush=True)


# == compute time on one core ========================================================================

def test_timing(args):
    """Compute time on one CPU core for the original train/test split of every dataset, measured like the QUANT
    paper's timing (and run_repro.py --n_jobs 1 --torch_threads 1). Each step is timed on its own, so the cost of
    every idea can be added up from its parts: features of all series, then training + prediction."""
    global N_JOBS
    from threadpoolctl import threadpool_limits
    N_JOBS = 1
    torch.set_num_threads(1)
    threadpool_limits(1)  # numpy / scipy (Hydra's ridge classifier) on one core as well
    out = args.out / "timing_1core.csv"
    done = finished(out)
    for name in args.datasets or NAMES:
        if name in done:
            continue
        d = Dataset(name)
        train, test = d.split(0)
        row = {"dataset": name}
        F, row["quant_features"] = timed(quant_features, d.X)
        P, row["pooled_features"] = timed(pooled_features, d.X)
        C, row["conv_features"] = timed(conv_quantiles, d.X)
        Z, row["hydra_features"] = timed(hydra_features, d.X)
        for key, G in (("pooled", np.hstack([F, P])), ("conv", np.hstack([F, C]))):
            t0 = time.perf_counter()
            row[f"{key}_accuracy"] = (forest(0).fit(G[train], d.y[train]).predict(G[test]) == d.y[test]).mean()
            row[f"{key}_fit_predict"] = time.perf_counter() - t0
        parts = {r["method"]: r for r in ensemble(F, Z, d.y, train, test, 0)}
        row["quant_fit_predict"] = parts["QUANT"]["fit_seconds"]  # forest: fit + predict
        row["hydra_fit_predict"] = parts["Hydra"]["fit_seconds"]  # scaler + ridge: fit + predict
        row["weighted_fit_predict"] = parts["QUANT + Hydra (weighted)"]["fit_seconds"]  # both + cross-validation
        row["quant_accuracy"] = parts["QUANT"]["accuracy"]
        row["ensemble_accuracy"] = parts["QUANT + Hydra (weighted)"]["accuracy"]
        append([row], out)
        print(f"timing   {name}: done", flush=True)


TESTS = {"varlen": test_varlen, "pooled": test_pooled, "convq": test_convq, "ensemble": test_ensemble,
         "augment": test_augment, "shift": test_shift, "timing": test_timing}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("tests", nargs="+", choices=list(TESTS))
    p.add_argument("--datasets", nargs="*", help="only these datasets (default: the test's own selection)")
    p.add_argument("--resamples", type=int, help="override the number of resamples")
    p.add_argument("--variants", nargs="*", help="varlen only: run just these variants (default: all)")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "improvements")
    args = p.parse_args()
    for t in args.tests:
        TESTS[t](args)
