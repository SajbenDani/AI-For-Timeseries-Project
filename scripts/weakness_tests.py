"""Diagnostic tests that measure where QUANT is weak. Nothing here changes QUANT; the tests only measure it.

  python scripts/weakness_tests.py padding      # padded variable-length datasets: errors vs amount of padding
  python scripts/weakness_tests.py duplicates   # datasets with train/test copies: accuracy with and without copies
  python scripts/weakness_tests.py fewshot      # accuracy vs number of training examples per class
  python scripts/weakness_tests.py manyclass    # many classes / very few examples: is the forest the bottleneck?
  python scripts/weakness_tests.py shift        # misalignment stress test: patterns at random time offsets
  python scripts/weakness_tests.py normalise    # per-series z-normalised vs raw input

QUANT runs exactly as in the paper and in run_repro.py (same loader, resamples and seeds). As a reference we use
Hydra (Dempster et al., 2023; codebase/hydra, unchanged): it counts which convolutional patterns occur anywhere
in a series, so unlike QUANT it does not depend on where a pattern happens. Results are written to
results/weaknesses/; datasets that are already finished are skipped when a test is re-run.
"""

import argparse
import sys
import zipfile
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codebase" / "hydra" / "code"))
import run_repro  # noqa: E402  (same .ts loader, resample indices and Quant as the reproduction)
from hydra import Hydra, SparseScaler  # noqa: E402

DATA = ROOT / "data" / "ucr142"
NAMES = (ROOT / "codebase" / "quant" / "code" / "ucr142.txt").read_text().split()
UCR112 = NAMES[:112]
# the 11 new datasets that were originally variable-length and were padded to equal length ("Eq")
VARLEN = ["AllGestureWiimoteXEq", "AllGestureWiimoteYEq", "AllGestureWiimoteZEq", "GestureMidAirD1Eq",
          "GestureMidAirD2Eq", "GestureMidAirD3Eq", "GesturePebbleZ1Eq", "GesturePebbleZ2Eq",
          "PickupGestureWiimoteZEq", "ShakeGestureWiimoteZEq", "PLAIDEq"]
SHOTS = [2, 5, 10, 20, None]  # training examples per class; None = full training set
LEVELS = [0.0, 0.1, 0.25, 0.5]  # misalignment: extra window length as a fraction of the series length


# == data =============================================================================================

def read_split(name, split):
    """(values text, label) of every series in one .ts file."""
    with open(DATA / name / f"{name}_{split}.ts", encoding="utf-8", errors="ignore") as f:
        in_data = False
        for line in f:
            line = line.strip()
            if not in_data:
                in_data = line.lower().startswith("@data")
            elif line:
                yield tuple(line.rsplit(":", 1))


def read_text(name):
    """Values text and label of every series in pool order (original train, then original test)."""
    rows = list(read_split(name, "TRAIN")) + list(read_split(name, "TEST"))
    return np.array([v for v, _ in rows], dtype=object), np.array([y for _, y in rows])


def sizes(name):
    """(n_train, n_test, length, smallest class count, number of classes) of the original split."""
    train = list(read_split(name, "TRAIN"))
    n_test = sum(1 for _ in read_split(name, "TEST"))
    length = len(train[0][0].split(","))
    counts = pd.Series([y for _, y in train]).value_counts()
    return len(train), n_test, length, counts.min(), len(counts)


class Dataset:
    """All series of a dataset in one pool (original train, then original test), as run.py builds it."""

    def __init__(self, name):
        X_tr, y_tr = run_repro.load_from_tsfile(f"{DATA}/{name}/{name}_TRAIN.ts")
        X_te, y_te = run_repro.load_from_tsfile(f"{DATA}/{name}/{name}_TEST.ts")
        self.name, self.n_train = name, len(y_tr)
        self.X64 = np.concatenate((X_tr, X_te))[:, 0]  # (n_series, length), float64 as loaded
        self.X = self.X64.astype(np.float32)
        self.labels = np.concatenate((y_tr, y_te))
        self.y = LabelEncoder().fit_transform(self.labels)
        self.idx_dir = run_repro.indices_dir(SimpleNamespace(indices="mixed", resample_indices_path=None), name)

    def split(self, i):
        """Pool positions of the training and test series of resample i (same rule as run.py)."""
        if i == 0:
            return np.arange(self.n_train), np.arange(self.n_train, len(self.y))
        train = np.loadtxt(self.idx_dir / self.name / f"resample{i}Indices_TRAIN.txt", dtype=np.int64)
        return train, np.setdiff1d(np.arange(len(self.y)), train)


# == methods ==========================================================================================

def quant_features(X):
    """QUANT features of every series. The transform has no fitted parameters (its windows depend only on the
    series length), so computing it once for the pool gives exactly what run.py computes split by split."""
    return run_repro.Quant().fit_transform(torch.from_numpy(X).unsqueeze(1), None).numpy()


def hydra_features(X, seed=0):
    with torch.no_grad():
        return Hydra(X.shape[-1], seed=seed).batch(torch.from_numpy(X).unsqueeze(1))


def quant(F, y, train, test, seed):
    """QUANT's classifier, configured as in the paper."""
    clf = ExtraTreesClassifier(n_estimators=200, max_features=0.1, criterion="entropy", n_jobs=-1,
                               random_state=seed)
    return clf.fit(F[train], y[train]).predict(F[test])


def quant_ridge(F, y, train, test):
    """Probe: the same QUANT features with a linear classifier, to separate feature and classifier effects."""
    scaler = StandardScaler().fit(F[train])
    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(scaler.transform(F[train]), y[train])
    return clf.predict(scaler.transform(F[test]))


def hydra(Z, y, train, test):
    """Hydra as in its reference example: sparse scaling and a ridge classifier."""
    train_t, test_t = torch.as_tensor(train), torch.as_tensor(test)
    scaler = SparseScaler()
    Z_train, Z_test = scaler.fit_transform(Z[train_t]).numpy(), scaler.transform(Z[test_t]).numpy()
    return RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(Z_train, y[train]).predict(Z_test)


# == helpers ==========================================================================================

def append(rows, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)


def finished(out):
    return set(pd.read_csv(out).dataset) if out.exists() else set()


def sample(candidates, size, seed):
    candidates = sorted(candidates)
    if len(candidates) <= size:
        return candidates
    return sorted(str(n) for n in np.random.default_rng(seed).choice(candidates, size=size, replace=False))


# == tests ============================================================================================

def test_padding(args):
    """Per test series: is it classified correctly, and how much of it is padding?"""
    folder = args.out / "padding"
    archive = zipfile.ZipFile(ROOT / "data" / "raw" / "Univariate2018_ts.zip")
    for name in args.datasets or VARLEN:
        out = folder / f"{name}.csv.gz"
        if out.exists():
            continue
        d = Dataset(name)
        native = name[:-2]  # e.g. AllGestureWiimoteXEq -> AllGestureWiimoteX (original, unpadded version)
        lengths, labels = [], []
        for split in ("TRAIN", "TEST"):
            lines = archive.read(f"Univariate_ts/{native}/{native}_{split}.ts").decode("utf-8", "ignore").splitlines()
            start = next(j for j, line in enumerate(lines) if line.strip().lower().startswith("@data")) + 1
            for line in filter(str.strip, lines[start:]):
                v, y = line.strip().rsplit(":", 1)
                lengths.append(len(v.split(",")))
                labels.append(y.lower())
        lengths = np.array(lengths)
        assert labels == [str(y).lower() for y in d.labels], f"{name}: native and padded versions differ in order"

        F, Z = quant_features(d.X), hydra_features(d.X)
        rows = []
        for i in range(args.resamples or 30):
            train, test = d.split(i)
            rows.append(pd.DataFrame({
                "dataset": name, "resample": i, "series": test, "native_length": lengths[test],
                "padded_fraction": 1 - lengths[test] / d.X.shape[1],
                "quant_correct": quant(F, d.y, train, test, seed=i) == d.y[test],
                "hydra_correct": hydra(Z, d.y, train, test) == d.y[test],
            }))
        folder.mkdir(parents=True, exist_ok=True)
        pd.concat(rows).to_csv(out, index=False)
        print(f"padding    {name}: done", flush=True)


def test_duplicates(args):
    """Accuracy with and without test series that have an exact copy in the training set."""
    out = args.out / "duplicates.csv"
    done = finished(out)
    for name in args.datasets or NAMES:
        keys, _ = read_text(name)
        if name in done or not pd.Series(keys).duplicated().any():
            continue
        d = Dataset(name)
        assert len(keys) == len(d.y)
        F = quant_features(d.X)
        rows = []
        for i in range(args.resamples or 30):
            train, test = d.split(i)
            seen = {}
            for j in train:
                seen.setdefault(keys[j], set()).add(d.y[j])
            copy = np.array([keys[j] in seen for j in test])
            conflict = np.array([keys[j] in seen and d.y[j] not in seen[keys[j]] for j in test])
            q = quant(F, d.y, train, test, seed=i) == d.y[test]
            nn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1).fit(d.X[train], d.y[train]).predict(d.X[test]) \
                == d.y[test]
            rows.append({
                "dataset": name, "resample": i, "n_test": len(test), "n_copies": int(copy.sum()),
                "n_conflicting_copies": int(conflict.sum()),
                "quant_acc": q.mean(), "quant_acc_without_copies": q[~copy].mean(),
                "quant_acc_on_copies": q[copy].mean() if copy.any() else np.nan,
                "nn_acc": nn.mean(), "nn_acc_without_copies": nn[~copy].mean(),
                "nn_acc_on_copies": nn[copy].mean() if copy.any() else np.nan,
            })
        append(rows, out)
        print(f"duplicates {name}: done", flush=True)


def test_fewshot(args):
    """Learning curves: train on k examples per class, test on the full test set."""
    out = args.out / "fewshot.csv"
    done = finished(out)
    if args.datasets:
        names = args.datasets
    else:
        # datasets with at least 20 training examples in every class, so k = 2..20 can be compared
        ok = [n for n in NAMES if (s := sizes(n))[3] >= 20 and s[1] <= 3000 and s[2] <= 1000]
        names = sample(ok, 25, seed=0)
        print(f"fewshot datasets ({len(names)} of {len(ok)} eligible): {names}", flush=True)
    for name in names:
        if name in done:
            continue
        d = Dataset(name)
        F, Z = quant_features(d.X), hydra_features(d.X)
        rows = []
        for i in range(args.resamples or 10):
            train, test = d.split(i)
            for k in SHOTS:
                if k is None:
                    tr = train
                else:
                    rng = np.random.default_rng(1000 * i + k)
                    tr = np.sort(np.concatenate([rng.choice(train[d.y[train] == c], k, replace=False)
                                                 for c in np.unique(d.y[train])]))
                for method, pred in (("QUANT", quant(F, d.y, tr, test, seed=i)),
                                     ("QUANT features + ridge", quant_ridge(F, d.y, tr, test)),
                                     ("Hydra", hydra(Z, d.y, tr, test))):
                    rows.append({"dataset": name, "resample": i, "per_class": "all" if k is None else k,
                                 "n_train": len(tr), "method": method, "accuracy": (pred == d.y[test]).mean()})
        append(rows, out)
        print(f"fewshot    {name}: done", flush=True)


def test_manyclass(args):
    """Full training sets of datasets with many classes or very few examples per class: is the forest the
    bottleneck there? Same three methods as the few-shot test."""
    out = args.out / "manyclass.csv"
    done = finished(out)
    if args.datasets:
        names = args.datasets
    else:
        names = [n for n in NAMES if (s := sizes(n))[0] / s[4] <= 5 or s[4] >= 25]
        print(f"manyclass datasets ({len(names)}): {names}", flush=True)
    for name in names:
        if name in done:
            continue
        d = Dataset(name)
        F, Z = quant_features(d.X), hydra_features(d.X)
        rows = []
        for i in range(args.resamples or 10):
            train, test = d.split(i)
            for method, pred in (("QUANT", quant(F, d.y, train, test, seed=i)),
                                 ("QUANT features + ridge", quant_ridge(F, d.y, train, test)),
                                 ("Hydra", hydra(Z, d.y, train, test))):
                rows.append({"dataset": name, "resample": i, "method": method,
                             "accuracy": (pred == d.y[test]).mean()})
        append(rows, out)
        print(f"manyclass  {name}: done", flush=True)


def misalign(X, level, rng):
    """Place each series at a random offset inside a window that is `level` x its length longer. The extra
    points repeat the first / last value plus a little noise, as if nothing happened before or after."""
    n, length = X.shape
    extra = int(round(level * length))
    if extra == 0:
        return X
    out = np.empty((n, length + extra), dtype=np.float32)
    for j, s in enumerate(rng.integers(0, extra + 1, size=n)):
        sd = 0.01 * X[j].std()
        out[j, :s] = X[j, 0] + rng.normal(0, sd, s)
        out[j, s:s + length] = X[j]
        out[j, s + length:] = X[j, -1] + rng.normal(0, sd, extra - s)
    return out


def test_shift(args):
    """Accuracy when the same patterns appear at random time offsets (train and test both misaligned)."""
    out = args.out / "shift.csv"
    done = finished(out)
    if args.datasets:
        names = args.datasets
    else:
        ok = [n for n in UCR112 if (s := sizes(n))[0] + s[1] <= 2000 and 60 <= s[2] <= 1000]
        names = sample(ok, 30, seed=1)
        print(f"shift datasets ({len(names)} of {len(ok)} eligible): {names}", flush=True)
    for name in names:
        if name in done:
            continue
        d = Dataset(name)
        rows = []
        for li, level in enumerate(LEVELS):
            X = misalign(d.X, level, np.random.default_rng([NAMES.index(name), li]))
            F, Z = quant_features(X), hydra_features(X)
            for i in range(args.resamples or 10):
                train, test = d.split(i)
                for method, pred in (("QUANT", quant(F, d.y, train, test, seed=i)),
                                     ("Hydra", hydra(Z, d.y, train, test))):
                    rows.append({"dataset": name, "resample": i, "level": level, "method": method,
                                 "accuracy": (pred == d.y[test]).mean()})
        append(rows, out)
        print(f"shift      {name}: done", flush=True)


def test_normalise(args):
    """QUANT on per-series z-normalised input, for datasets whose series are not already z-normalised."""
    out = args.out / "normalise.csv"
    done = finished(out)
    for name in args.datasets or NAMES:
        if name in done:
            continue
        head = [np.array(v.split(","), dtype=float) for v, _ in islice(read_split(name, "TRAIN"), 100)]
        if not args.datasets and np.mean([abs(v.mean()) for v in head]) <= 0.05 \
                and abs(np.mean([v.std() for v in head]) - 1) <= 0.05:
            continue  # already z-normalised (same rule as our earlier check)
        d = Dataset(name)
        sd = d.X64.std(axis=1, keepdims=True)
        X = ((d.X64 - d.X64.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1)).astype(np.float32)
        F = quant_features(X)
        rows = []
        for i in range(args.resamples or 30):
            train, test = d.split(i)
            rows.append({"dataset": name, "resample": i,
                         "accuracy_znorm": (quant(F, d.y, train, test, seed=i) == d.y[test]).mean()})
        append(rows, out)
        print(f"normalise  {name}: done", flush=True)


TESTS = {"padding": test_padding, "duplicates": test_duplicates, "fewshot": test_fewshot,
         "manyclass": test_manyclass, "shift": test_shift, "normalise": test_normalise}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("tests", nargs="+", choices=list(TESTS))
    p.add_argument("--datasets", nargs="*", help="only these datasets (default: the test's own selection)")
    p.add_argument("--resamples", type=int, help="override the number of resamples (for quick checks)")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "weaknesses")
    args = p.parse_args()
    for t in args.tests:
        TESTS[t](args)
