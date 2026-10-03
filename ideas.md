# Ideas for improving QUANT

The 10 most promising ways to improve QUANT, ranked by expected impact. Each idea is backed by (a) something we **measured** in our weakness tests (report Section V, `results/weaknesses/summary.txt`) and (b) **published work** it builds on. Nothing here is implemented yet.

**The one-sentence diagnosis behind most ideas:** QUANT describes "what values occur in a *fixed window*", so when the same pattern appears at a different time, it produces different features. When patterns moved to random positions, QUANT lost 5.6 accuracy points against Hydra's 2.4 (Hydra is position-free). The same cause most likely explains its large losses on gesture data (−5.1 points) and on many-class data with few examples (up to −27 points).

## Summary

| # | Idea | Fixes which weakness | Effort | Risk | New? |
|---|---|---|---|---|---|
| 1 | Position-free QUANT: sliding-window quantiles, pooled over positions | Patterns at different positions | Medium | Medium | Yes |
| 2 | "QUANT meets ROCKET": quantiles of random-convolution outputs | Position + local shape | Medium | Medium | Yes (no paper pools by quantiles) |
| 3 | QUANT + Hydra ensemble with calibrated, accuracy-weighted votes | Different errors of the two methods | Low | Low | For UCR, yes |
| 4 | QUANT features + tabular foundation model (TabPFN / TabICL) | Few examples, many classes | Medium | Medium (GPU helps) | Yes (only ROCKET-style features tried) |
| 5 | Shift/warp augmentation of training data + test-time augmentation | Patterns at different positions / speeds | Low | Low | For QUANT, yes |
| 6 | Align series before windowing (cross-correlation / phase alignment) | Position; periodic signals | Low-medium | Medium | For QUANT, yes |
| 7 | Phase-invariant representations (autocorrelation, log-spectrum, AR) | Periodic signals with random phase | Low | Low | Partly (rSTSF uses AR) |
| 8 | Oblique / rotation trees for correlated quantile features | Few examples; wasted splits | Medium | Medium (speed) | For QUANT, yes |
| 9 | Prune features + sample big windows more often (authors' own idea) | Training cost on long series | Low-medium | Low | Partly |
| 10 | Native variable-length QUANT (windows relative to each series' length) | Padding (costs ~8 points) | Medium | Low | Yes (aeon can't do it) |

**Suggested order:**
1. Start with **5 and 6**. They are cheap tests of the "position" diagnosis.
2. Then build **1 or 2**, the main research contribution.
3. Use **3** as a strong, safe baseline.
4. Add **4** if we can get a free GPU (Kaggle / Colab).
5. **9** and **10** are good secondary contributions (speed, new capability).

---

## 1. Position-free QUANT: pooled sliding-window quantiles

**What.** QUANT currently takes quantiles in fixed windows (first half, second quarter, ...). Instead, slide a window of each size across the *whole* series, compute the quantiles at every position, and then summarise those values over all positions (maximum, mean, and a few quantiles). A feature then reads "somewhere in the series there is a stretch whose 90th percentile is this high", which no longer depends on *where* it happens. Keep the original fixed-window features too, so location information isn't lost; the forest picks what helps.

**Why it should help.**
- It removes the exact cause of our main measured weakness: QUANT −5.6 vs Hydra −2.4 points under random offsets, and −5.1 points on gesture data.
- It should also help the many-class datasets, where we think position is what the forest can't learn from only 2 examples per class.

**Inspired by:**
- [ROCKET](https://arxiv.org/abs/1910.13051) / [MiniRocket](https://arxiv.org/abs/2012.08791) and [Hydra](https://arxiv.org/abs/2203.13652), which pool over *all* positions. That pooling is why they are position-free.
- TSBF ([Baydoğan et al., TPAMI 2013](https://pyts.readthedocs.io/en/stable/generated/pyts.classification.TSBF.html)), which bags interval features from many locations.
- [Zhang et al. 2025](https://arxiv.org/abs/2503.20264), who show interval methods break under misalignment.

**How to test.**
- Rerun our shift test (`weakness_tests.py shift`). Success means QUANT's drop shrinks towards Hydra's.
- Then run the full 142 × 30 benchmark against plain QUANT: the overall accuracy must not drop.

## 2. "QUANT meets ROCKET": quantile pooling of random convolutions

**What.** Pass each series through a few hundred random convolution kernels, as MiniRocket does. Each kernel is a small "shape detector", and its output is a new series showing how well that shape matches at each time. Then describe each output series with **quantiles over the whole series**, instead of ROCKET's usual summaries (maximum, or the proportion of positive values, "PPV"). It's a natural fusion: the maximum *is* the 100% quantile, and PPV is just the share of values above 0, so quantiles carry both and more.

**Why it should help.**
- The result is shape-aware (the kernels) *and* position-free (quantiles over the whole output), which targets both of QUANT's main losses.
- QUANT's strength, describing distributions with quantiles, is kept and applied to richer inputs.

**Inspired by:**
- [MultiRocket](https://arxiv.org/abs/2102.00457), which showed that adding more pooling statistics improves accuracy.
- [SelF-Rocket](https://arxiv.org/abs/2409.01115), which showed the best pooling statistic depends on the dataset. It tried PPV, mean of positive values, zero crossings and others, but **not quantiles**.
- QUANT itself.

**How to test.** Compare four variants on the 142 datasets: QUANT, QUANT + conv-quantiles, MiniRocket, and conv-quantiles alone. Also run our shift test.

## 3. QUANT + Hydra ensemble (calibrated, accuracy-weighted)

**What.** Train QUANT and Hydra (or MultiRocket+Hydra) separately and average their predicted class probabilities. Each model's vote is weighted by its cross-validated accuracy on the training data, as HIVE-COTE does with its "CAWPE" weighting.
- Ridge-based methods give poor probabilities (MultiRocket+Hydra: log-loss 4.6 vs QUANT's 0.5 in the bake-off). They first need calibrating, e.g. a softmax over their scores with a temperature fitted by cross-validation.

**Why it should help.**
- The two methods fail on different data. QUANT does well on spectrographs and where the value *level* matters (our normalisation test). Hydra wins where *shape and position* matter (shift test, many-class data).
- QUANT already has the second-best probabilities among top methods (AUROC 0.956), so it's a good ensemble partner.

**Inspired by:**
- [HIVE-COTE 2](https://arxiv.org/abs/2104.07551) and its CAWPE weighting (Large, Lines & Bagnall, DAMI 2019).
- The [Meta-Learning Gap](https://arxiv.org/abs/2512.06666) (2025). For Hydra+QUANT on large datasets, a perfect "oracle" combination would gain 4.7%, but simple combinations captured only 11% of that, so *how* we combine matters.

**How to test.**
- Run on 142 × 30 and compare with QUANT, Hydra and MultiRocket+Hydra.
- Compare with HC2 using the bake-off files. Can a two-method ensemble get close to HC2 at a tiny fraction of its cost?

## 4. QUANT features + a tabular foundation model ("QuantPFN")

**What.** Keep QUANT's features but replace the forest with **TabPFN** or **TabICL**. These models are pre-trained on millions of synthetic tables and classify a new dataset in a single forward pass, with no training.
- They accept at most 500–2,000 features, so we reduce QUANT's ~3,000 features first, using idea 9's ranking or random feature groups as RocketPFN does.
- For more classes than the model's built-in limit, use TabPFN's official [many-class extension](https://docs.priorlabs.ai/extensions/many-class), which splits the problem into smaller ones. This covers the 52-class Pig data.

**Why it should help.**
- QUANT is weakest with many classes and few examples (−5.6 points vs Hydra on 18 datasets), and these models are built exactly for small data.
- Our test showed the fix must be a *non-linear* classifier: a linear one on QUANT's features was 11.7 points worse.

**Inspired by:**
- [RocketPFN](https://arxiv.org/abs/2606.21786) (2026): ROCKET features + TabPFN **matched HC2** (0.900 on 92 UCR datasets). Swapping only the classifier gave +1.8 points.
- [TS2TabPFN](https://arxiv.org/abs/2608.04174) (2026), which tried catch22, tsfresh and MultiROCKET features, but **not QUANT**.
- [TabPFN](https://www.nature.com/articles/s41586-024-08328-6) (Nature 2025) and [TabICL](https://arxiv.org/abs/2502.05564) (ICML 2025; faster, larger datasets).

**How to test.**
- Start with the ≤10-class datasets, then the many-class wrapper.
- A free Kaggle/Colab GPU makes the full 142 × 30 run practical. On CPU only, limit it to small datasets.

## 5. Shift/warp augmentation + test-time augmentation

**What.**
- **Training:** add shifted and slightly stretched copies of every training series, so the forest sees each pattern at several positions and speeds.
- **Testing:** classify a few shifted versions of each test series and average the predictions.

**Why it should help.**
- It's the cheapest direct test of our diagnosis: if moved patterns are the problem, showing the forest moved patterns should help.
- It targets the few-example datasets most, where the forest can't learn positions from 2 examples.
- People also gesture at different *speeds*, which stretching covers.

**Inspired by:**
- Window slicing and warping (Le Guennec et al., AALTD 2016).
- The augmentation survey of [Iwana & Uchida 2021](https://arxiv.org/abs/2007.15951).
- Test-time augmentation ([Molchanov et al. 2020](https://arxiv.org/abs/2002.09103), [Shanmugam et al. 2021](https://arxiv.org/abs/2011.11156)).
- [MantisV2](https://arxiv.org/abs/2602.17868), which uses test-time strategies for time series.

**How to test.** Use our shift test and the 18 many-class datasets. Watch the cost: k copies make training about k× slower.

## 6. Align series before windowing

**What.** Shift every series so its patterns line up before QUANT cuts it into windows, then QUANT runs unchanged:
- **General data:** align each series to a reference (e.g. the training mean) using the shift with the highest cross-correlation.
- **Periodic signals** such as heartbeats: rotate each series so its main cycle starts at the same phase. This can be read off the Fourier transform.

**Why it should help.**
- With aligned series, a fixed window means the same thing in every series again.
- The Pig datasets, where QUANT loses up to 27 points, are heartbeat recordings with beats at arbitrary positions.

**Inspired by:**
- k-Shape's shift-invariant cross-correlation distance (Paparrizos & Gravano, SIGMOD 2015).
- [Kaya & Gündüz-Öğüdücü 2015](https://research.itu.edu.tr/tr/publications/a-distance-based-time-series-classification-framework/), who found that "a suitable alignment method enhances classification performance on nearly every dataset".

**How to test.** Use the shift test, the Pig datasets and the gesture datasets. Check it doesn't hurt the already-aligned datasets.

## 7. Phase-invariant representations

**What.** QUANT computes quantiles on four views of each series (raw, 1st and 2nd difference, Fourier magnitude). Add views that ignore *when* things happen:
- the **autocorrelation function** (how similar the series is to itself shifted by k steps),
- the **log-spectrum**,
- **autoregressive (AR) coefficients**.

**Why it should help.**
- The Fourier *magnitude* is already one such view, and the QUANT paper itself found it gave the largest per-dataset gains of all added views.
- The autocorrelation function is the time-domain counterpart and ignores shifts in the same way. It's ideal for periodic data like the Pig recordings.
- This is the smallest code change on the list: one extra line per representation.

**Inspired by:**
- [rSTSF](https://arxiv.org/abs/2105.14876), which uses periodogram and AR representations.
- [DrCIF / HC2](https://arxiv.org/abs/2104.07551) (periodogram).
- [catch22](https://arxiv.org/abs/1901.10200) (autocorrelation features).
- QUANT's own ablation, from its Fig. 10–11.

**How to test.** Run 142 × 30 and check the per-domain effect, especially on periodic signals.

## 8. Oblique / rotation trees for correlated quantile features

**What.** Replace the forest's ordinary splits ("feature 17 > 0.3") with splits on *combinations* of features. Rotation Forest does this by rotating feature subsets first; sparse oblique forests (SPORF) split on random sparse feature combinations.

**Why it should help.**
- QUANT's features are strongly correlated: neighbouring quantiles of the same window move together. Single-feature splits waste effort on that.
- Combined splits can capture "the whole distribution shifted up" in one step, which matters most when there are few examples.

**Inspired by:**
- Bagnall et al., ["Is rotation forest the best classifier for problems with continuous features?"](https://arxiv.org/abs/1809.06705): yes, on average, though slower.
- FreshPRINCE (Middlehurst & Bagnall 2022), which uses Rotation Forest on time series features.
- SPORF (Tomita et al., JMLR 2020).

**How to test.**
- Compare on the 18 many-class datasets and the few-example test, then on 142 × 30.
- Watch the runtime: Rotation Forest has a time-capped "contract" version.

## 9. Prune features and sample big windows more often (speed)

**What.**
1. Fit a small forest, keep only the most useful 10–50% of features, then fit the final forest.
2. The QUANT authors' own unfinished idea: make features from **larger windows more likely** to be tried at each split.

**Why it should help.**
- Training the forest takes about 78% of the time, and that time grows with the number of features (correlation 0.97 in our timing data).
- QUANT makes about 9 features per time step, so long series are slow: the 10 longest datasets take 37% of the total time.
- Fewer, better features could make training 2–4× faster without losing accuracy.

**Inspired by:**
- [Detach-ROCKET](https://arxiv.org/abs/2309.14518): keeping 10% of features gave *higher* accuracy (+0.2%).
- [Enriched random forests](https://www.academia.edu/26032115/Enriched_random_forests): weighted feature sampling helps when few features matter.
- The [QUANT paper](https://arxiv.org/abs/2308.00928) §4.2.1 (their future-work note).
- Tree "pasting", i.e. training each tree on a subsample ([Dempster et al. 2024](https://link.springer.com/chapter/10.1007/978-3-031-77066-1_5)).

**How to test.** Do a single-core timing run plus 142 × 30 accuracy. Success means at least 2× faster training with no significant accuracy loss.

## 10. Native variable-length QUANT

**What.** Let QUANT handle series of different lengths without padding:
- Define windows as *fractions* of each series' own length (its first half, its second quarter, ...).
- Keep the number of quantiles per window fixed so all feature vectors match.
- Add the series length as one extra feature.

**Why it should help.**
- Padding costs *every* method about 8 points on heavily padded series in our test. Some series are 11 real points padded to 500.
- QUANT could avoid padding entirely, just as MiniRocket already has a variable-length version.
- It's the QUANT authors' own stated future work, and aeon's QUANT doesn't support variable lengths.
- The original unpadded versions of the 11 datasets are already on our disk, in the same order, so the existing splits apply.

**Inspired by:**
- The [QUANT paper](https://arxiv.org/abs/2308.00928) (future work).
- The [UCR 2018 archive](https://arxiv.org/abs/1810.07758) (native variable-length data).
- [MiniRocket](https://arxiv.org/abs/2012.08791), whose authors also released a variable-length version.
- [aeon's QUANT](https://github.com/aeon-toolkit/aeon/blob/main/aeon/classification/interval_based/_quant.py) (no unequal-length support).

**How to test.**
- Compare the 11 datasets, native vs padded, on identical splits.
- Sanity check: on equal-length data it must give exactly QUANT's features.

---

## Also considered (smaller or already done elsewhere)

- **Class-weighted trees.** Small fix for the 0.9-point extra loss on imbalanced data.
- **Raw + normalised quantiles side by side.** Normalising costs 3.1 points on average but gains 2–6 points on gesture data.
- **De-duplicated evaluation.** 25 datasets have train/test copies, so results should be reported with and without them. This fixes the *evaluation*, not QUANT.
- **Faster quantile computation.** Already published as [MomentQuant](https://arxiv.org/abs/2609.05136) (2026), and the transform is only ~22% of the runtime.
- **Multivariate QUANT and regression.** Already exist (authors' code and aeon's `QUANTRegressor`), so there's little novelty.
- **Deep foundation models as feature extractors (Mantis, MOMENT).** [RocketPFN](https://arxiv.org/abs/2606.21786) shows they lose to simple random features on UCR.
