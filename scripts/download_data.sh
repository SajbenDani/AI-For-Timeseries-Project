#!/usr/bin/env bash
# Download the 142 UCR datasets used by QUANT plus the bake-off redux resample indices.
# Sources: https://github.com/time-series-machine-learning/tsml-eval/blob/main/tsml_eval/publications/y2023/tsc_bakeoff/tsc_bakeoff_2023.ipynb
# Usage (from the project root, with the venv's python on PATH or PY set):
#   bash scripts/download_data.sh
set -euo pipefail

PY="${PY:-python}"
cd "$(dirname "$0")/.."
mkdir -p data/raw
cd data/raw

# 128 original UCR datasets (.ts) -- QUANT uses 112 of them
curl -fSL -o Univariate2018_ts.zip "https://www.timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_ts.zip"
# 30 new datasets from the bake-off redux (Google Drive)
"$PY" -m gdown 1vuh6mgNrNKjHr9MMRQP0J0_gGA4dE7E3 -O new30.zip
# resample indices (Java tsml resampler and Python resampler)
"$PY" -m gdown 1FsG7Fp74y_TpaPhJ7U066ot8A07BPhr3 -O indices_java.zip
"$PY" -m gdown 1aLBP_nhnoqz075puKg30zuF3F_QBOXYM -O indices_python.zip

unzip -q -o Univariate2018_ts.zip -d ucr128
unzip -q -o new30.zip -d new30
unzip -q -o indices_java.zip -d idx
unzip -q -o indices_python.zip -d idx

cd ../..
"$PY" scripts/prepare_data.py
