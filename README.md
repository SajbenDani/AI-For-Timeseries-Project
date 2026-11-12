# Reproducing and Improving QUANT

UCD *AI for Time Series* group project by Dániel Sajben (26107621) and Tymur Sukhariev (26100566).

**QUANT** (Dempster, Schmidt & Webb, *Data Mining and Knowledge Discovery*, 2024) is a fast time series classifier. It describes each series by the quantiles of fixed windows and classifies these features with an ExtraTrees forest. We:

1. reproduce the paper's main experiment (142 UCR datasets × 30 train/test splits);
2. measure where QUANT is weak;
3. test fifteen ideas for improving it, and whether the best ones stack;
4. extend QUANT to multivariate data with views that combine channels (19 UEA datasets).

Everything runs on a laptop CPU, with no GPU needed.

- **Paper:** [Papers/documentation.pdf](Papers/documentation.pdf) (IEEE format; LaTeX source [Papers/documentation.tex](Papers/documentation.tex), figures in [Papers/figures/](Papers/figures/)).
- **All ideas, with the evidence behind them and the result of each test:** [ideas.md](ideas.md).
- **Full numbers of every experiment:** the `summary.txt` files in [results/](results/).

## Who did what

We planned the project, discussed every result and wrote the paper together. The experiments were split by time:

- **Dániel Sajben:** the reproduction, the weakness tests, round 1 (ideas 1, 2, 3, 5, 10) and round 2 (ideas 14–18), the first draft of the paper and report, and the repository setup.
- **Tymur Sukhariev:** round 3 (ideas 7, 11, 12, 13, and multivariate data: ideas 18 and 19 on 19 UEA datasets, with the rotation stress test) and round 4 (idea 9, feature pruning, and the stacking of ideas 1 + 14 + 18), the paper's multivariate section, merging the new results into the paper, and macOS support.

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

**Improvements** (142 datasets × first 10 resamples; time is one-core compute relative to QUANT, unless the row says otherwise):

| Method | Accuracy | vs QUANT | Time |
|---|---|---|---|
| QUANT | 85.36% | | 1× |
| QUANT + pooled + speed quantiles + Hydra when QUANT is unsure (ideas 1 + 14 + 18)* | **87.47%** | **+2.11** | not timed |
| QUANT + Hydra, always averaged (idea 3) | 87.39% | +2.03 | 5.8× (4.1× with equal weights) |
| QUANT + Hydra, asked only when QUANT is unsure (idea 18) | 87.16% | +1.80 | 2.1× |
| QUANT + position-free pooled quantiles (idea 1) | 85.75% | +0.39 | 1.4× |
| QUANT on the top 25% of its features (idea 9) | 85.65% | +0.29 | 1.7× faster training (all cores) |
| QUANT + speed-invariant quantiles (idea 14) | 85.50% | +0.14 | 1.3× |
| MultiRocket+Hydra / HIVE-COTE 2 (from the bake-off redux) | 87.06% / 87.53% | | ≈5× / weeks |

\*With one fixed gate setting (not picked on held-out datasets, unlike the 87.16% row); idea 18 alone at that setting gives 87.22%, so stacking adds +0.25 points (p = 4e-8). The stacked version is statistically level with HIVE-COTE 2 (−0.06 points, p = 0.55).

**Multivariate data** (19 UEA datasets × 5 resamples):

| Method | Accuracy | vs QUANT |
|---|---|---|
| QUANT | 74.18% | |
| QUANT + Hydra, gated (idea 18) | 75.59% | +1.41 |
| QUANT + cross-channel views (idea 19) | 76.80% | +2.62 |
| Ideas 18 + 19 | **77.78%** | **+3.60** |

With the sensors of the 6 accelerometer/gyroscope datasets tilted at random by up to 180°, QUANT loses 18.1 points and idea 19 only 12.3.

**Ideas that did not help on average:** shift/stretch augmentation (idea 5), the cycle view (15), the bispectrum view (16), rank-uniform splits (17), autocorrelation and AR features (7, although they gave +4 to +14 points on the Pig heartbeat recordings), multi-scale differences (11), order-aware counts per window (12), and other classifiers instead of ExtraTrees (13: Random Forest −1.3 and LightGBM −2.7 points). Details are in the paper and in [ideas.md](ideas.md).

## Repository layout

```
codebase/quant, codebase/hydra   original QUANT and Hydra code (git submodules, unmodified)
scripts/                         all our code, run in the order given below
  quant_extensions.py              every new view or feature we add to QUANT (ideas 1, 2, 5, 7, 10-12, 14-16)
  *_tests.py                       the experiments (they need data/); most also have a "summary" step
  *_analysis.py, compare_*.py      turn saved results into the summaries, tables and figures (no data needed)
  analysis_utils.py                helpers shared by the analysis steps (paired test, bake-off results)
results/                         raw results (CSV) and text summaries of every experiment
  repro/                           the reproduction and its comparison with the paper
  checks/                          which resample indices the paper used; table of the 142 datasets;
                                   Holm correction over all comparisons of the paper's Table III
  weaknesses/                      where QUANT is weak
  improvements/                    round 1: ideas 1, 2, 3, 5, 10 (varlen_fresh/: idea 10 on unseen data)
  ideas/                           read-only checks behind ideas 14-18
  efficient/                       round 2: ideas 14-18
  autocorr/, multiscale/,          round 3: ideas 7, 11, 12, 13
    order_counts/, classifiers/
  multivariate/                    round 3: QUANT, Hydra and idea 18 on 19 UEA datasets
  cross_channel/                   round 3: idea 19 and the rotation stress test
  pruning/, stacking/              round 4: idea 9, ideas 1 + 14 + 18 stacked
  bakeoff_redux/                   published results of other methods, used for comparison
Papers/                          the paper (documentation.tex, documentation.pdf) and its figures
ideas.md                         every idea we considered, with evidence and verdict
data/                            created by the download steps (not in git, ~4 GB)
```

## Setup

You need Python 3.11, git, and a bash shell with `curl` and `unzip`. On Windows, Git Bash works.

```bash
git clone --recursive https://github.com/SajbenDani/AI-For-Timeseries-Project.git
cd AI-For-Timeseries-Project
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On macOS, LightGBM (idea 13) also needs `brew install libomp`.

## Checking our numbers without re-running anything

The raw results of every run are already in `results/`. These scripts rebuild every summary, table and figure from them in about a minute, with no data download needed:

```bash
python scripts/compare_results.py && python scripts/compare_methods.py && python scripts/compare_timing.py
python scripts/weakness_analysis.py
python scripts/improvement_analysis.py
python scripts/idea_checks.py
python scripts/efficient_analysis.py
python scripts/paper_figures.py
python scripts/view_tests.py summary && python scripts/classifier_tests.py summary
python scripts/pruning_tests.py summary && python scripts/stacking_tests.py summary
python scripts/multivariate_tests.py summary && python scripts/cross_channel_tests.py summary rotate_summary
python scripts/multiple_testing.py
```

| Paper | Script | Output |
|---|---|---|
| Tables I-II, runtime (Section III) | `compare_results.py`, `compare_methods.py`, `compare_timing.py` | `results/repro/summary_*.csv`, `mcm_comparison.csv` |
| Section IV, Fig. 1 | `weakness_analysis.py` | `results/weaknesses/summary.txt`, `Papers/figures/weaknesses.pdf` |
| Section V, Table III (ideas 1, 2, 3, 5, 10) | `improvement_analysis.py` | `results/improvements/summary.txt` |
| Section V, Table III (ideas 14-18) | `idea_checks.py`, `efficient_analysis.py` | `results/ideas/checks.txt`, `results/efficient/summary.txt` |
| Section V, Table III (ideas 7, 11, 12, 13) | `view_tests.py`, `classifier_tests.py` | `results/autocorr/summary.txt`, `results/multiscale/summary.txt`, `results/order_counts/summary.txt`, `results/classifiers/summary.txt` |
| Section V, Table III (idea 9, stacking 1 + 14 + 18) | `pruning_tests.py`, `stacking_tests.py` | `results/pruning/summary.txt`, `results/stacking/summary.txt` |
| Section VI, Table IV (multivariate, ideas 18 and 19, rotation test) | `multivariate_tests.py`, `cross_channel_tests.py` | `results/multivariate/summary.txt`, `results/cross_channel/summary.txt`, `results/cross_channel/rotate_summary.txt` |
| Section V-A, Holm correction over Table III | `multiple_testing.py` | `results/checks/multiple_testing.txt` |
| Fig. 2 | `paper_figures.py` | `Papers/figures/main_results.pdf` |

## Reproducing everything from scratch

Run these steps in order, because later steps read earlier results. Steps 1–5 take about 30 h of compute in total on our first laptop (Ryzen 5 5600H, 6 cores, 16 GB RAM). Steps 6–8 were run on a second laptop (a MacBook); their times are wall-clock times of our runs, about 13 h in total.

- **Resumable:** every run saves after each finished dataset and skips finished datasets when restarted, so it can be stopped at any time. To redo a step, delete its output files first.
- **Same numbers as ours?** All runs use the official resamples and seed = resample index. On the same computer this reproduces our accuracies exactly. On another computer, single splits can differ: between our two laptops, plain QUANT differed on 722 of 1,420 splits (by up to 4.3 points on one split and 0.9 points on one dataset's mean), while the mean over the 142 datasets was the same (85.36%). The likely cause is that different processors round QUANT's features differently in the last digit, and a forest magnifies such differences. So expect the same means and conclusions, but not every digit. This is why steps 6 and 8 are compared with plain QUANT fitted on the same computer (in the pruning run of step 8). Timings always depend on the machine.
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

# 6. Round 3: the remaining ideas 7, 11, 12, 13 (142 x 10; ideas 7 and 12 ~15 min each, idea 11 ~45 min, idea 13 ~6 h)
python scripts/view_tests.py main               # ideas 7, 11 and 12
python scripts/classifier_tests.py main rf lgbm # idea 13

# 7. Round 3: multivariate data (19 UEA datasets x 5; downloads the UEA files into data/UEA/; ~20 min in total)
python scripts/multivariate_datasets.py
python scripts/multivariate_tests.py main summary
python scripts/cross_channel_tests.py check main summary rotate rotate_summary

# 8. Round 4: feature pruning (~1 h) and stacking (~4.5 h) (142 x 10), then the summaries of step 6
#    (they compare with plain QUANT from the pruning run, on the same computer)
python scripts/pruning_tests.py main summary
python scripts/stacking_tests.py main summary
python scripts/view_tests.py summary && python scripts/classifier_tests.py summary
python scripts/multiple_testing.py              # Holm correction over all comparisons of Table III

# 9. The paper's Fig. 2 and the PDF (needs a TeX distribution with IEEEtran, e.g. TeX Live or MiKTeX)
python scripts/paper_figures.py
cd Papers && pdflatex documentation.tex && pdflatex documentation.tex && cd ..
```

## Credits

- **QUANT and Hydra:** original code by Angus Dempster and co-authors (GPL-3.0). It is included unmodified as submodules.
- **Data:** the UCR archive (Dau et al., 2019), the 30 new datasets and resample indices from the bake-off redux (Middlehurst, Schäfer & Bagnall, 2024), the original variable-length Asphalt sets from timeseriesclassification.com, and the UEA multivariate archive (Bagnall et al., 2018), loaded with aeon.