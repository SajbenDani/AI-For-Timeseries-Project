"""Download the UEA multivariate datasets we use and save a table of their sizes.

Run from the repo root:
    python scripts/multivariate_datasets.py

Data goes to data/UEA/ (not in git). The table goes to
results/multivariate/datasets.csv (in git).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from aeon.datasets import load_classification

# Small/medium, equal-length UEA datasets with at most 30 channels.
# Skipped: too big (EigenWorms, FaceDetection, InsectWingbeat, MotorImagery,
# PEMS-SF, DuckDuckGeese, PhonemeSpectra, Heartbeat) or variable length
# (CharacterTrajectories, JapaneseVowels, SpokenArabicDigits).
DATASETS = [
    "ArticularyWordRecognition",
    "AtrialFibrillation",
    "BasicMotions",
    "Cricket",
    "Epilepsy",
    "ERing",
    "EthanolConcentration",
    "FingerMovements",
    "HandMovementDirection",
    "Handwriting",
    "Libras",
    "LSST",
    "NATOPS",
    "PenDigits",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "StandWalkJump",
    "UWaveGestureLibrary",
]

DATA_DIR = Path("data/UEA")
OUT = Path("results/multivariate/datasets.csv")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in DATASETS:
        X_tr, y_tr = load_classification(
            name, split="train", extract_path=str(DATA_DIR)
        )
        X_te, y_te = load_classification(name, split="test", extract_path=str(DATA_DIR))
        X_tr, X_te = np.asarray(X_tr), np.asarray(X_te)
        rows.append(
            {
                "dataset": name,
                "n_train": X_tr.shape[0],
                "n_test": X_te.shape[0],
                "channels": X_tr.shape[1],
                "length": X_tr.shape[2],
                "classes": len(np.unique(y_tr)),
                "has_nan": bool(np.isnan(X_tr).any() or np.isnan(X_te).any()),
            }
        )
        print(f"{name}: {rows[-1]}")
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nSaved {len(rows)} datasets to {OUT}")


if __name__ == "__main__":
    main()
