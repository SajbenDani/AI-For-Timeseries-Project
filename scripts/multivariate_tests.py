"""Multivariate QUANT on UEA datasets (plan steps 3 and 6; ideas.md idea 19 comes later).

  python scripts/multivariate_tests.py main      # M1-M3: QUANT, Hydra, QUANT + Hydra gated (idea 18), 5 resamples
  python scripts/multivariate_tests.py summary   # -> results/multivariate/summary.txt

Datasets: results/multivariate/datasets.csv (made by multivariate_datasets.py; data in data/UEA/).
Same rules as the univariate tests: QUANT is not changed, its forest is the paper's, seed = resample index, and
finished datasets are skipped when the test is re-run. QUANT's own code already handles several channels (every
window's quantiles are taken per channel). Hydra uses its authors' multivariate version (codebase/hydra,
hydra_multivariate.py), with a fixed seed. Resample 0 is the official split; resamples 1-4 keep the official
number of training series per class, drawn at random with seed = resample index.
A dataset that fails is reported and skipped, so one error never stops the whole run.
"""

import argparse
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.preprocessing import LabelEncoder

from weakness_tests import (
    ROOT,
    append,
    finished,
)  # also puts codebase/hydra/code on the path
import run_repro
from improvement_tests import forest, proba, timed
from efficient_tests import gate, hydra_loo
from hydra_multivariate import HydraMultivariate  # noqa: E402

warnings.filterwarnings("ignore")

DATA = ROOT / "data" / "UEA"
OUT = ROOT / "results" / "multivariate"
NAMES = pd.read_csv(OUT / "datasets.csv").dataset.tolist()

# idea 18's two-level gate, as chosen on the univariate benchmark (report.md: gap 0.3-0.4, share 10-20%)
THRESHOLD = (
    0.4  # ask Hydra when the gap between QUANT's two most likely classes is below this
)
MIN_ASKED = (
    0.1  # ... and only if at least this share of a dataset's test series is unsure
)


# == data =============================================================================================


class UEADataset:
    """All series of a UEA dataset in one pool (official train, then official test)."""

    def __init__(self, name):
        from aeon.datasets import load_classification

        X_tr, y_tr = load_classification(name, split="train", extract_path=str(DATA))
        X_te, y_te = load_classification(name, split="test", extract_path=str(DATA))
        self.name, self.n_train = name, len(y_tr)
        self.X = np.ascontiguousarray(
            np.concatenate([np.asarray(X_tr), np.asarray(X_te)]), dtype=np.float32
        )
        self.labels = np.concatenate([np.asarray(y_tr), np.asarray(y_te)]).astype(str)
        self.y = LabelEncoder().fit_transform(self.labels)

    def split(self, i):
        """Pool positions of the training and test series of resample i."""
        n = len(self.y)
        if i == 0:
            return np.arange(self.n_train), np.arange(self.n_train, n)
        rng = np.random.default_rng(i)
        y_train = self.y[: self.n_train]
        train = [
            rng.choice(
                np.flatnonzero(self.y == c), int((y_train == c).sum()), replace=False
            )
            for c in np.unique(y_train)
        ]
        train = np.sort(np.concatenate(train))
        return train, np.setdiff1d(np.arange(n), train)


# == methods ==========================================================================================


def quant_features(X):
    """QUANT features of (n_series, channels, length) series: its own code, unchanged."""
    return run_repro.Quant().fit_transform(torch.from_numpy(X), None).numpy()


def hydra_features(X, seed=0):
    """Hydra's authors' multivariate version (random kernels over random channel groups), fixed seed."""
    torch.manual_seed(seed)
    with torch.no_grad():
        return HydraMultivariate(X.shape[-1], X.shape[1]).batch(torch.from_numpy(X))


# == M1-M3 ============================================================================================


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
            Z, t_hydra = timed(hydra_features, d.X)
            rows = []
            for i in range(args.resamples or 5):
                train, test = d.split(i)
                base = {"dataset": name, "resample": i}
                t0 = time.perf_counter()
                p_quant = proba(forest(i).fit(F[train], d.y[train]), F[test], C)
                rows.append(
                    {
                        **base,
                        "method": "QUANT",
                        "accuracy": (p_quant.argmax(1) == d.y[test]).mean(),
                        "fit_seconds": time.perf_counter() - t0,
                        "transform_seconds": t_quant,
                        "num_features": F.shape[1],
                    }
                )
                t0 = time.perf_counter()
                p_hydra, T = hydra_loo(Z, d.y, train, test, C)
                rows.append(
                    {
                        **base,
                        "method": "Hydra",
                        "accuracy": (p_hydra.argmax(1) == d.y[test]).mean(),
                        "fit_seconds": time.perf_counter() - t0,
                        "transform_seconds": t_hydra,
                        "num_features": Z.shape[1],
                        "temperature": T,
                    }
                )
                rows += [
                    {
                        **base,
                        "method": "QUANT + Hydra, gated",
                        **g,
                        "transform_seconds": t_hydra,
                    }
                    for g in gate(p_quant, p_hydra, d.y[test])
                ]
            append(rows, out)
            print(
                f"main {name}: done in {time.perf_counter() - t_start:.0f} s",
                flush=True,
            )
        except Exception:
            print(f"main {name}: FAILED, skipped\n{traceback.format_exc()}", flush=True)


# == summary ==========================================================================================


def paired(a, b):
    """Mean difference, W/D/L and Wilcoxon p of per-dataset accuracies a vs b (as in efficient_analysis.py)."""
    d = np.round(np.asarray(a) - np.asarray(b), 4)
    p = (
        wilcoxon(a, b, zero_method="pratt").pvalue
        if (d != 0).any() and len(d) > 1
        else 1.0
    )
    return d.mean(), int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum()), p


def test_summary(args):
    df = pd.read_csv(OUT / "main.csv")
    q = df[df.method == "QUANT"].set_index(["dataset", "resample"]).accuracy
    h = df[df.method == "Hydra"].set_index(["dataset", "resample"]).accuracy
    g = df[df.method == "QUANT + Hydra, gated"]
    full = (
        g[g.threshold == g.threshold.max()].set_index(["dataset", "resample"]).accuracy
    )
    at = g[np.isclose(g.threshold, THRESHOLD)].set_index(["dataset", "resample"])
    two_level = at.accuracy.where(at.asked >= MIN_ASKED, q.reindex(at.index))
    methods = {
        "M1 QUANT": q,
        "M2 Hydra": h,
        "QUANT + Hydra, always (full ensemble)": full,
        f"M3 QUANT + Hydra, two-level gate (gap < {THRESHOLD}, >= {MIN_ASKED:.0%} unsure)": two_level,
    }
    per_dataset = pd.DataFrame(
        {m: s.groupby(level="dataset").mean() for m, s in methods.items()}
    )
    resamples = df.groupby("dataset")["resample"].nunique()

    lines = [
        f"Multivariate QUANT: {len(per_dataset)} UEA datasets x up to {resamples.max()} resamples "
        f"(mean accuracy per dataset, then over datasets)",
        "",
    ]
    base = per_dataset["M1 QUANT"]
    for m in per_dataset:
        line = f"{m:<70} {100 * per_dataset[m].mean():6.2f}%"
        if m != "M1 QUANT":
            diff, w, dr, l_, p = paired(per_dataset[m], base)
            line += f"   vs QUANT {100 * diff:+.2f} pts  W/D/L {w}/{dr}/{l_}  Wilcoxon p={p:.2g}"
        lines.append(line)
    asked = at.asked.where(at.asked >= MIN_ASKED, 0)
    lines += [
        "",
        f"Gate: Hydra asked on {asked.mean():.0%} of test series, "
        f"used in {(at.asked >= MIN_ASKED).groupby(level='dataset').mean().mean():.0%} of datasets",
        "",
        "Per dataset (%):",
        (100 * per_dataset).round(2).to_string(),
    ]
    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text + "\n")
    print(text)


TESTS = {"main": test_main, "summary": test_summary}

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
