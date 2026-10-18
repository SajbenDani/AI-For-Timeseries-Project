# Reproducing and Improving QUANT

UCD *AI for Time Series* group project by Dániel Sajben (26107621) and Tymur Sukhariev (26100566).

**QUANT** (Dempster, Schmidt & Webb, *Data Mining and Knowledge Discovery*, 2024) is a fast time series classifier. It describes each series by the quantiles of fixed windows and classifies these features with an ExtraTrees forest. We:

1. reproduce the paper's main experiment (142 UCR datasets × 30 train/test splits);
2. measure where QUANT is weak;
3. test ten ideas for improving it.

Everything runs on a laptop CPU, with no GPU needed.

- **Paper:** [Papers/documentation.tex](Papers/documentation.tex) (IEEE format; figures in [Papers/figures/](Papers/figures/)).
- **Full write-up of every experiment:** [report.md](report.md). **All ideas, with evidence and verdicts:** [ideas.md](ideas.md).

## Main results

**Reproduction** (mean accuracy over 30 resamples; "Bake-off" is the independent QUANT run from the bake-off redux paper):

| Datasets | n | Paper | Bake-off | Ours |
|---|---|---|---|---|
| Original UCR | 112 | 0.8671 | 0.8671 | **0.8671** |
| New, unchanged | 26 | 0.8008 | 0.8019 | **0.8017** |
| New, corrected after publication* | 4 | 0.9306 | 0.8662 | **0.8673** |
| All | 142 | 0.8568 | 0.8552 | **0.8551** |
| All except the 4 corrected | 138 | 0.8546 | 0.8548 | **0.8548** |
| Compute time, 1 CPU core (one split of every dataset) | | 14.9 min | | **18.3 min** |

\*Covid3Month and FloodModeling1–3 were corrected by the archive maintainers after the paper; only the new versions can be downloaded. Per-dataset results: `results/repro/comparison_per_dataset.csv`.

**Improvements** (142 datasets × first 10 resamples; time is one-core compute relative to QUANT):

| Method | Accuracy | vs QUANT | Time |
|---|---|---|---|
| QUANT | 85.36% | | 1× |
| QUANT + Hydra, asked only when QUANT is unsure (idea 18) | **87.16%** | +1.80 | 2.1× |
| QUANT + Hydra, always averaged (idea 3) | 87.39% | +2.03 | 5.8× (4.1× with equal weights) |
| QUANT + position-free pooled quantiles (idea 1) | 85.75% | +0.39 | 1.4× |
| QUANT + speed-invariant quantiles (idea 14) | 85.50% | +0.14 | 1.3× |
| MultiRocket+Hydra / HIVE-COTE 2 (from the bake-off redux) | 87.06% / 87.53% | | ≈5× / weeks |

Several ideas did not help; they are reported in the paper and in [report.md](report.md).

## Repository layout

```
codebase/quant, codebase/hydra   original QUANT and Hydra code (git submodules, unmodified)
scripts/                         all our code, run in the order given below
results/                         raw results (CSV) and text summaries of every experiment
  repro/                           the reproduction and its comparison with the paper
  checks/                          which resample indices the paper used; table of the 142 datasets
  weaknesses/                      where QUANT is weak
  improvements/                    round 1: ideas 1, 2, 3, 5, 10 (varlen_fresh/: idea 10 on unseen data)
  ideas/                           read-only checks behind ideas 14-18
  efficient/                       round 2: ideas 14-18
  bakeoff_redux/                   published results of other methods, used for comparison
Papers/                          the paper (documentation.tex) and its figures
report.md, ideas.md              detailed write-up and idea catalogue
data/                            created by the download step (not in git, ~4 GB)
```

## Setup

You need Python 3.11, git, and a bash shell with `curl` and `unzip`. On Windows, Git Bash works.

```bash
git clone --recursive https://github.com/SajbenDani/AI-For-Timeseries-Project.git
cd AI-For-Timeseries-Project
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Checking our numbers without re-running anything

The raw results of every run are already in `results/`. These scripts rebuild every summary, table and figure from them in about a minute, with no data download needed:

```bash
python scripts/compare_results.py && python scripts/compare_methods.py && python scripts/compare_timing.py
python scripts/weakness_analysis.py
python scripts/improvement_analysis.py
python scripts/idea_checks.py
python scripts/efficient_analysis.py
python scripts/paper_figures.py
```

| Paper | Script | Output |
|---|---|---|
| Tables I-II, runtime (Section III) | `compare_results.py`, `compare_methods.py`, `compare_timing.py` | `results/repro/summary_*.csv`, `mcm_comparison.csv` |
| Section IV, Fig. 1 | `weakness_analysis.py` | `results/weaknesses/summary.txt`, `Papers/figures/weaknesses.pdf` |
| Section V, Table III (ideas 1, 2, 3, 5, 10) | `improvement_analysis.py` | `results/improvements/summary.txt` |
| Section V, Table III (ideas 14-18) | `idea_checks.py`, `efficient_analysis.py` | `results/ideas/checks.txt`, `results/efficient/summary.txt` |
| Fig. 2 | `paper_figures.py` | `Papers/figures/main_results.pdf` |

## Reproducing everything from scratch

Run these steps in order, because later steps read earlier results. About 30 h of compute in total on our laptop (Ryzen 5 5600H, 6 cores, 16 GB RAM).

- **Resumable:** every run saves after each finished dataset and skips finished datasets when restarted, so it can be stopped at any time. To redo a step, delete its output files first.
- **Same numbers as ours:** all runs use the official resamples and seed = resample index, so accuracies come out exactly as ours. Only timings depend on the machine.
- **Timing runs:** run the one-core timing steps alone on the machine.

```bash
# 1. Data: 142 UCR datasets, official resample indices, 3 unpadded Asphalt sets (~0.9 GB download, ~4 GB on disk)
bash scripts/download_data.sh

# 2. Reproduce QUANT (~4 h, then ~30 min for the one-core timing)
python scripts/run_repro.py --indices mixed --seed 0 -o results/repro/quant_ucr142_30res.csv
python scripts/run_repro.py -r 1 --n_jobs 1 --torch_threads 1 -o results/repro/quant_ucr142_timing_1core.csv
python scripts/check_indices.py    # shows the paper used Java indices for the 112 original datasets, Python for the 30 new
python scripts/compare_results.py && python scripts/compare_methods.py && python scripts/compare_timing.py

# 3. Where QUANT is weak (~4 h)
python scripts/weakness_tests.py padding duplicates fewshot manyclass shift normalise
python scripts/weakness_analysis.py

# 4. Round 1: ideas 1, 2, 3, 5, 10 (~10 h, then ~45 min and ~3.5 h)
python scripts/improvement_tests.py varlen pooled ensemble convq augment shift
python scripts/improvement_tests.py varlen --datasets AsphaltObstaclesUniEq AsphaltPavementTypeUniEq AsphaltRegularityUniEq \
    --variants "QUANT (padded, original)" "QUANT (native windows)" "QUANT (padded + native windows)" \
    --out results/improvements/varlen_fresh      # idea 10 checked on 3 datasets it was not designed on
python scripts/improvement_tests.py timing      # one-core timing
python scripts/improvement_analysis.py

# 5. Round 2: ideas 14-18 (~5 min, ~4 h, then ~3 h)
python scripts/idea_checks.py
python scripts/efficient_tests.py main warp shift
python scripts/efficient_tests.py timing        # one-core timing
python scripts/efficient_analysis.py

# 6. The paper's Fig. 2 and the PDF (needs a TeX distribution with IEEEtran, e.g. TeX Live or MiKTeX)
python scripts/paper_figures.py
cd Papers && pdflatex documentation.tex && pdflatex documentation.tex
```

## Credits

- **QUANT and Hydra:** original code by Angus Dempster and co-authors (GPL-3.0). It is included unmodified as submodules.
- **Data:** the UCR archive (Dau et al., 2019), the 30 new datasets and resample indices from the bake-off redux (Middlehurst, Schäfer & Bagnall, 2024), and the original variable-length Asphalt sets from timeseriesclassification.com.
