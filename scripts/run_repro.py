"""Reproduce the QUANT UCR-142 experiments.

Identical experimental logic to quant/code/run.py (same .ts loader, same resampling
scheme, same Quant transform and ExtraTreesClassifier configuration), with additions
needed for a reproducibility study:
  * per-stage timings (transform train/test, classifier fit/predict), as in the
    published results/accuracy_mean_ucr142.csv;
  * results appended per (dataset, resample) so long runs can resume;
  * configurable classifier threads, torch threads and random seed;
  * choice of resample-index set (Java or Python tsml resampler);
  * each dataset is parsed once rather than once per resample (no effect on results).
"""

import argparse
import csv
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
QUANT_CODE = ROOT / "codebase" / "quant" / "code"
sys.path.insert(0, str(QUANT_CODE))
from quant import Quant  # noqa: E402


def _load_ts_loader():
    # reuse the exact .ts loader from run.py without executing its experiment loop
    source = (QUANT_CODE / "run.py").read_text()
    start = source.index("def _load_header_info")
    end = source.index("# == run ==")
    namespace = {"np": np, "re": __import__("re")}
    exec(compile(source[start:end], "run.py", "exec"), namespace)
    return namespace["load_from_tsfile"]


load_from_tsfile = _load_ts_loader()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input_path", default=str(ROOT / "data" / "ucr142"))
    p.add_argument("-d", "--datasets", default=str(QUANT_CODE / "ucr142.txt"))
    p.add_argument("-o", "--output", required=True, help="csv file (appended to / resumed)")
    p.add_argument("-r", "--num_resamples", type=int, default=30)
    p.add_argument("--resample_start", type=int, default=0)
    p.add_argument("-p", "--resample_indices_path", default=None,
                   help="defaults to data/indices_<indices>")
    p.add_argument("--indices", choices=["java", "python", "mixed"], default="mixed",
                   help="mixed = Java for the original 112 datasets, Python for the 30 new ones "
                        "(the scheme used in the bake-off redux)")
    p.add_argument("-m", "--depth", type=int, default=6)
    p.add_argument("-v", "--div", type=int, default=4)
    p.add_argument("-e", "--num_estimators", type=int, default=200)
    p.add_argument("-f", "--max_features", default="0.1")
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--torch_threads", type=int, default=None)
    p.add_argument("--seed", type=int, default=None, help="ExtraTrees random_state (None = unseeded, as in run.py)")
    return p.parse_args()


NEW30 = set((QUANT_CODE / "ucr142.txt").read_text().split()[112:])


def indices_dir(args, dataset):
    if args.resample_indices_path:
        return Path(args.resample_indices_path)
    kind = args.indices
    if kind == "mixed":
        kind = "python" if dataset in NEW30 else "java"
    return ROOT / "data" / f"indices_{kind}"


def main():
    args = parse_args()
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    max_features = float(args.max_features) if args.max_features not in ["sqrt"] else args.max_features

    datasets = np.loadtxt(args.datasets, "str", ndmin=1)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with out.open() as f:
            done = {(r["dataset"], int(r["resample"])) for r in csv.DictReader(f)}
    fields = ["dataset", "resample", "accuracy", "time_transform_training_seconds",
              "time_transform_test_seconds", "time_fit_classifier_seconds", "time_predict_seconds",
              "num_features", "indices"]
    write_header = not out.exists()

    with out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()

        for dataset in datasets:
            todo = [i for i in range(args.resample_start, args.num_resamples) if (dataset, i) not in done]
            if not todo:
                continue
            _X_TR, _Y_TR = load_from_tsfile(f"{args.input_path}/{dataset}/{dataset}_TRAIN.ts")
            _X_TE, _Y_TE = load_from_tsfile(f"{args.input_path}/{dataset}/{dataset}_TEST.ts")
            _X = np.concatenate((_X_TR, _X_TE))
            _Y = np.concatenate((_Y_TR, _Y_TE))
            idx_dir = indices_dir(args, dataset)

            for i in todo:
                # == identical to run.py ==========================================
                if i == 0:
                    X_TR, X_TE, Y_TR, Y_TE = _X_TR, _X_TE, _Y_TR, _Y_TE
                else:
                    indices = np.arange(_X.shape[0])
                    indices_TR = np.loadtxt(idx_dir / dataset / f"resample{i}Indices_TRAIN.txt", dtype=np.int32)
                    indices_TE = np.setdiff1d(indices, indices_TR)
                    X_TR, X_TE = _X[indices_TR], _X[indices_TE]
                    Y_TR, Y_TE = _Y[indices_TR], _Y[indices_TE]

                X_TR = torch.tensor(X_TR.squeeze().astype(np.float32)).unsqueeze(1)
                X_TE = torch.tensor(X_TE.squeeze().astype(np.float32)).unsqueeze(1)

                le = LabelEncoder()
                Y_TR = le.fit_transform(Y_TR)
                Y_TE = le.transform(Y_TE)

                quant = Quant(depth=args.depth, div=args.div)

                t0 = time.perf_counter()
                Z_TR = quant.fit_transform(X_TR, Y_TR)
                t1 = time.perf_counter()

                classifier = ExtraTreesClassifier(
                    n_estimators=args.num_estimators,
                    max_features=max_features,
                    criterion="entropy",
                    n_jobs=args.n_jobs,
                    random_state=None if args.seed is None else args.seed + i,
                )
                classifier.fit(Z_TR, Y_TR)
                t2 = time.perf_counter()

                Z_TE = quant.transform(X_TE)
                t3 = time.perf_counter()

                accuracy = (classifier.predict(Z_TE) == Y_TE).mean()
                t4 = time.perf_counter()
                # =================================================================

                writer.writerow({
                    "dataset": dataset, "resample": i, "accuracy": round(float(accuracy), 6),
                    "time_transform_training_seconds": round(t1 - t0, 6),
                    "time_transform_test_seconds": round(t3 - t2, 6),
                    "time_fit_classifier_seconds": round(t2 - t1, 6),
                    "time_predict_seconds": round(t4 - t3, 6),
                    "num_features": Z_TR.shape[1],
                    "indices": "original" if i == 0 else idx_dir.name,
                })
                f.flush()
            print(f"{dataset}: mean acc over {len(todo)} resamples done", flush=True)


if __name__ == "__main__":
    main()
