"""Size, length, classes, class imbalance and domain of the 142 datasets -> results/checks/dataset_stats.csv.

The weakness, idea and round-2 analyses read this table (e.g. to group datasets by number of classes or by
training examples per class). Domains ("type") come from the UCR archive's DataSummary.csv, saved in
results/checks/ucr_DataSummary.csv; it covers only the 112 original datasets, so the 30 new ones have none.

Usage: python scripts/dataset_stats.py      (after scripts/download_data.sh)
"""

from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ucr142"
OUT = ROOT / "results" / "checks"
NAMES = (ROOT / "codebase" / "quant" / "code" / "ucr142.txt").read_text().split()


def series(path):
    """(number of values, label) of every series in a .ts file."""
    in_data = False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not in_data:
            in_data = line.lower().startswith("@data")
        elif line:
            values, label = line.rsplit(":", 1)
            yield values.count(",") + 1, label


def main():
    summary = pd.read_csv(OUT / "ucr_DataSummary.csv", encoding="utf-8-sig")
    summary.columns = summary.columns.str.strip()
    types = summary.set_index("Name")["Type"]
    rows = []
    for i, name in enumerate(NAMES):
        files = {split: DATA / name / f"{name}_{split}.ts" for split in ("TRAIN", "TEST")}
        train, test = list(series(files["TRAIN"])), list(series(files["TEST"]))
        counts = Counter(y for _, y in train)
        rows.append({"dataset": name, "train": len(train), "test": len(test),
                     "length": max(n for n, _ in train + test), "classes": len(counts),
                     "imbalance": max(counts.values()) / min(counts.values()),  # largest / smallest training class
                     "mb": sum(f.stat().st_size for f in files.values()) / 1e6,
                     "group": "ucr112" if i < 112 else "new30", "type": types.get(name) if i < 112 else None})
    pd.DataFrame(rows).to_csv(OUT / "dataset_stats.csv", index=False)
    print(f"wrote {OUT / 'dataset_stats.csv'} ({len(rows)} datasets)")


if __name__ == "__main__":
    main()
