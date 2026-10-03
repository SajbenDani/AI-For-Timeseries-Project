"""Arrange the downloaded UCR archives into the layout expected by quant/code/run.py.

Inputs (extracted into data/raw/ by scripts/download_data.sh):
  data/raw/ucr128/Univariate_ts/<name>/<name>_{TRAIN,TEST}.ts   (original 128 UCR, .ts)
  data/raw/new30/<name>/<name>_{TRAIN,TEST}.ts                   (30 bake-off redux datasets)
  data/raw/idx/{Java,Python}ResampleIndices/<name>/resample<i>Indices_TRAIN.txt

Outputs (keyed by the dataset names used in quant/code/ucr142.txt):
  data/ucr142/<name>/<name>_{TRAIN,TEST}.ts
  data/indices_java/<name>/resample<i>Indices_{TRAIN,TEST}.txt
  data/indices_python/<name>/resample<i>Indices_{TRAIN,TEST}.txt
"""

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DATASETS = ROOT / "codebase" / "quant" / "code" / "ucr142.txt"


def source_name(name):
    # QUANT uses e.g. "AconityMINIPrinterLargeEq", the released files use "AconityMINIPrinterLarge_eq"
    for suffix, replacement in (("Eq", "_eq"), ("Nmv", "_nmv"), ("Discrete", "_disc")):
        if name.endswith(suffix) and not name.endswith("_" + suffix.lower()):
            return re.sub(suffix + "$", replacement, name)
    return name


def main():
    names = DATASETS.read_text().split()
    assert len(names) == 142, len(names)

    for name in names:
        src = source_name(name)
        candidates = [RAW / "ucr128" / "Univariate_ts" / src, RAW / "new30" / src]
        data_dir = next((c for c in candidates if c.exists()), None)
        assert data_dir is not None, f"missing data for {name} ({src})"

        out = ROOT / "data" / "ucr142" / name
        out.mkdir(parents=True, exist_ok=True)
        for split in ("TRAIN", "TEST"):
            shutil.copyfile(data_dir / f"{src}_{split}.ts", out / f"{name}_{split}.ts")

        for kind in ("Java", "Python"):
            idx_src = RAW / "idx" / f"{kind}ResampleIndices" / src
            assert idx_src.exists(), f"missing {kind} indices for {name} ({src})"
            idx_out = ROOT / "data" / f"indices_{kind.lower()}" / name
            if idx_out.exists():
                shutil.rmtree(idx_out)
            shutil.copytree(idx_src, idx_out)

    print(f"prepared {len(names)} datasets in {ROOT / 'data'}")


if __name__ == "__main__":
    main()
