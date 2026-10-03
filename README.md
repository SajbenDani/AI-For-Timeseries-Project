# Reproducing and Improving QUANT

This repository reproduces **QUANT** (Dempster, Schmidt & Webb, *Data Mining and Knowledge Discovery*, 2024), a fast time series classifier that describes each series by the quantiles of fixed windows and classifies them with an ExtraTrees forest. Next, it will try to improve on it. The original code is included unmodified as a git submodule in `codebase/quant`. Our scripts in `scripts/` run the same experiment (142 UCR datasets × 30 train/test resamples) and compare our numbers with the paper's. The report is `Papers/documentation.tex`. Everything runs on a normal CPU; no GPU is needed.

**Setup** (Python 3.11):

```bash
git clone --recursive <this repo> && cd <repo>
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Download the data**: `bash scripts/download_data.sh` fetches the 142 UCR datasets and the official resample indices (~0.7 GB download, 1.4 GB on disk) into `data/`.

**Quick first run** (9 small datasets, a few minutes): `python scripts/check_indices.py`. This shows that the paper used the Java resample indices for the 112 original datasets and the Python ones for the 30 new datasets.

**Full reproduction** (~3-4 h on a 6-core laptop; resumable if interrupted):

```bash
python scripts/run_repro.py --indices mixed --seed 0 -o results/repro/quant_ucr142_30res.csv
python scripts/run_repro.py -r 1 --n_jobs 1 --torch_threads 1 -o results/repro/quant_ucr142_timing_1core.csv  # single-core timing
python scripts/compare_results.py      # ours vs paper, per dataset -> results/repro/comparison_per_dataset.csv
python scripts/compare_methods.py      # paper's comparison tables (Fig. 1 & 6) -> results/repro/mcm_comparison.csv
python scripts/compare_timing.py       # compute time vs paper -> results/repro/summary_timing.csv
```

**Results** (mean accuracy over 30 resamples; "Bake-off" is the independent QUANT run from the bake-off redux paper):

| Datasets | n | Paper | Bake-off | Ours |
|---|---|---|---|---|
| Original UCR | 112 | 0.8671 | 0.8671 | **0.8671** |
| New, unchanged | 26 | 0.8008 | 0.8019 | **0.8017** |
| New, corrected after publication* | 4 | 0.9306 | 0.8662 | **0.8673** |
| All | 142 | 0.8568 | 0.8552 | **0.8551** |
| Compute time, 1 core (all 142) | | 14.9 min | | **18.3 min** |

\*Covid3Month and FloodModeling1-3 were fixed by the archive maintainers after the paper. The paper's numbers use the old versions. On the 138 other datasets, ours and the paper's agree (0.8548 vs 0.8546). Per-dataset results are in `results/repro/comparison_per_dataset.csv`.
