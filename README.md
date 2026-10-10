# Reproducing and Improving QUANT

This repository reproduces **QUANT** (Dempster, Schmidt & Webb, *Data Mining and Knowledge Discovery*, 2024), a fast time series classifier that describes each series by the quantiles of fixed windows and classifies them with an ExtraTrees forest. It then tests ten ideas for improving it, in two rounds (see [report.md](report.md)). The original code is included unmodified as a git submodule in `codebase/quant`. Our scripts in `scripts/` run the same experiment (142 UCR datasets × 30 train/test resamples) and compare our numbers with the paper's. The report is `Papers/documentation.tex`. Everything runs on a normal CPU; no GPU is needed.

**Setup** (Python 3.11):

```bash
git clone --recursive https://github.com/SajbenDani/AI-For-Timeseries-Project.git && cd AI-For-Timeseries-Project
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

## Where QUANT is weak

Controlled tests that change one thing at a time and compare QUANT with [Hydra](https://github.com/angus924/hydra), a convolution method that does not care *where* a pattern occurs. Hydra's original code is in `codebase/hydra`. The tests reuse the resamples and seeds above, so on unchanged data QUANT's numbers are identical to the reproduction. They take about 4 h in total and can be resumed.

```bash
python scripts/weakness_tests.py padding duplicates fewshot manyclass shift normalise   # -> results/weaknesses/
python scripts/weakness_analysis.py    # numbers quoted in the report -> results/weaknesses/summary.txt
                                       # figure -> Papers/figures/weaknesses.pdf
```

| Test | Finding |
|---|---|
| Patterns at random positions (30 datasets) | QUANT loses 5.6 points, Hydra 2.4; Hydra's lead grows from 1.1 to 4.4 points. **Main weakness.** |
| Many classes / ≤5 examples per class (18 datasets) | QUANT 5.6 points behind Hydra (up to 27 on Pig data); a linear classifier on QUANT's features is worse still |
| Fewer training examples (25 datasets, 2-20 per class) | Gap to Hydra grows only from 0.2 to 1.1 points |
| Padded variable-length data (11 datasets) | Padding hurts QUANT and Hydra equally (about 8 points); not QUANT-specific |
| Exact train/test copies (25 datasets) | Inflate accuracy by up to 5.7 points (Worms); 142-dataset mean 0.8551 → 0.8542 without them |
| Class imbalance (bake-off results) | 0.9 points extra loss in balanced accuracy when the largest class is >3x the smallest |
| z-normalising the input (51 datasets) | Costs 3.1 points on average (the level often carries the class); helps only on gesture data (+2-6) |

## Improving QUANT

We picked five ideas from [ideas.md](ideas.md) and added each one *on top of* the unchanged QUANT. Each was tested on the same splits and seeds as the reproduction: 142 datasets × the first 10 resamples, or 11 variable-length datasets × 30. A stress test also moves the patterns to random positions (30 datasets × 5 resamples). The code is in `scripts/quant_extensions.py` (new features) and `scripts/improvement_tests.py` (tests). The full, plain-language write-up is [report.md](report.md).

```bash
python scripts/improvement_tests.py varlen pooled ensemble convq augment shift   # -> results/improvements/ (~10 h, resumable)
python scripts/improvement_tests.py timing        # single-core cost (~3.5 h); run it alone on the machine
python scripts/improvement_analysis.py            # -> results/improvements/summary.txt, Papers/figures/improvements.pdf
# idea 10 checked on 3 fresh variable-length datasets (originals: data/raw/native5, from timeseriesclassification.com)
python scripts/improvement_tests.py varlen --datasets AsphaltObstaclesUniEq AsphaltPavementTypeUniEq AsphaltRegularityUniEq \
    --variants "QUANT (padded, original)" "QUANT (native windows)" "QUANT (padded + native windows)" --out results/improvements/varlen_fresh
```

| Idea | Accuracy (QUANT: 85.36%) | Patterns at random positions (QUANT: 80.3%) | Runtime vs QUANT* | Verdict |
|---|---|---|---|---|
| 3. QUANT + Hydra ensemble | **87.39%** (+2.0; level with HIVE-COTE 2) | **85.3%** | ×5.8 (×4.1 equal weights‡) | Success |
| 1. Position-free pooled quantiles | 85.75% (+0.4) | 82.1% | ×1.4 | Small success |
| 2. Quantiles of random convolutions | 85.76% (+0.4) | 83.1% | ×4.0 | Small success |
| 10. Variable length: padded + native windows | 77.53% vs 75.55% (11 datasets, +2.0); 92.52% vs 92.31% on 3 fresh datasets (+0.2) | n/a | ×9† (slow first implementation) | Success on the 11 (mostly gestures); small on fresh data |
| 5. Shift/stretch augmentation | 84.64% (−0.7) | 82.3% | ×5.8† | Did not help |

\*Features + training + prediction for one split of every dataset on one CPU core, as in the QUANT paper. Plain QUANT takes 15.7 min this way, and the ensemble 91 min. †Measured on all 12 threads (plain QUANT: 5.2 min). ‡With round 2's free calibration of Hydra's scores; our first equal-weights run still cross-validated it (×5.0).

**Round 2: efficient approaches.** [ideas.md](ideas.md) (ideas 14–18) lists five ideas that should raise accuracy while keeping QUANT cheap. They are implemented (new views in `scripts/quant_extensions.py`, tests in `scripts/efficient_tests.py`) and tested; results are in [report.md](report.md). Best: asking Hydra only where QUANT is unsure gives 87.16% (+1.8 points) at 2.1× QUANT's one-core time (with a smaller Hydra: +1.3 points at 1.27×).

```bash
python scripts/idea_checks.py                     # read-only checks behind ideas 14-18 -> results/ideas/ (~5 min)
python scripts/efficient_tests.py main warp shift # ideas 14-18 + stress tests -> results/efficient/ (resumable)
python scripts/efficient_tests.py timing          # single-core cost; run it alone on the machine
python scripts/efficient_analysis.py              # -> results/efficient/summary.txt, Papers/figures/efficient.pdf
python scripts/paper_figures.py                   # -> Papers/figures/main_results.pdf (Fig. 2 of the paper)
```
