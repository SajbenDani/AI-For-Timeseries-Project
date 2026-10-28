# Ideas for improving QUANT

The most promising ways to improve QUANT. Ideas 1–10 are ranked by expected impact. Ideas 11–13 came from the team's suggestions, and the verdict on every suggestion is at the end. Ideas 14–18 are **efficient approaches**: new ideas, added after our experiments, that should raise accuracy while keeping QUANT cheap. Each idea is backed by (a) something we **measured** in our weakness tests (paper Section IV, `results/weaknesses/summary.txt`) and (b) **published work** it builds on.

**Status (7 Oct 2026):** ideas 1, 2, 3, 5 and 10 have been implemented and tested on all 142 datasets, plus a stress test with patterns at random positions (`scripts/improvement_tests.py`). Results and explanations are in [report.md](report.md). The other ideas are not implemented. Ideas 14–18 (efficient approaches) were added on 7 Oct 2026. All five are implemented and tested on the 142 datasets, plus two stress tests (complete, 10 Oct). Results are in [report.md](report.md), "Round 2". The two choices we made after seeing results were then checked on data they had not seen (10 Oct): idea 18's two-level gate held up, while idea 10's "padded + native" version gave only +0.2 pts on three fresh datasets.

**The one-sentence diagnosis behind most ideas:** QUANT describes "what values occur in a *fixed window*", so when the same pattern appears at a different time, it produces different features. When patterns moved to random positions, QUANT lost 5.6 accuracy points against Hydra's 2.4 (Hydra is position-free). The same cause most likely explains its large losses on gesture data (−5.1 points) and on many-class data with few examples (up to −27 points).

## Summary

| # | Idea | Fixes which weakness | Effort | Risk | New? | Status |
|---|---|---|---|---|---|---|
| 1 | Position-free QUANT: sliding-window quantiles, pooled over positions | Patterns at different positions | Medium | Medium | Yes | **Done: success** (+0.4 pts; +1.8 under random shifts) |
| 2 | "QUANT meets ROCKET": quantiles of random-convolution outputs | Position + local shape | Medium | Medium | Yes (no paper pools by quantiles) | **Done: success** (+0.4 pts, slower; +2.8 under random shifts) |
| 3 | QUANT + Hydra ensemble with calibrated, accuracy-weighted votes | Different errors of the two methods | Low | Low | For UCR, yes | **Done: success** (+2.0 pts, but 4–6× slower; +5.0 under random shifts) |
| 4 | QUANT features + tabular foundation model (TabPFN / TabICL) | Few examples, many classes | Medium | Medium (GPU helps) | Yes (only ROCKET-style features tried) | Not done (needs a GPU) |
| 5 | Shift/warp augmentation of training data + test-time augmentation | Patterns at different positions / speeds | Low | Low | For QUANT, yes | **Done: did not help** on average (−0.7 pts; −0.35 with test-time averaging), but +1.1 pts on many-class data and +2.0 under random shifts |
| 6 | Align series before windowing (cross-correlation / phase alignment) | Position; periodic signals | Low-medium | Medium | For QUANT, yes | Not done |
| 7 | Phase-invariant representations (autocorrelation, AR coefficients) | Periodic signals with random phase | Low | Low | Partly (rSTSF uses AR) | **Done: did not help on average** (+0.25 pts, 70/13/59, p = 0.095), but large gains on the periodic Pig datasets (PigCVP +14.5, PigAirwayPressure +5.8, PigArtPressure +4.3); worst PLAIDEq −5.6 |
| 8 | Oblique / rotation trees for correlated quantile features | Few examples; wasted splits | Medium | Medium (speed) | For QUANT, yes | Not done |
| 9 | Prune features + sample big windows more often (authors' own idea) | Training cost on long series | Low-medium | Low | Partly | Not done |
| 10 | Native variable-length QUANT (windows relative to each series' length) | Padding (costs ~8 points) | Medium | Low | Yes (aeon can't do it) | **Done: partial success** (padded + native: +2.0 pts on the 11 mostly-gesture datasets, but only +0.2 on 3 fresh datasets; native alone +0.1) |
| 11 | Multi-scale differences ("wavelet-like" views) | Change at medium speeds, located in time | Low | Low | For QUANT, yes | Not done |
| 12 | Order-aware counts per window (mean crossings, peaks, slope) | Quantiles ignore the order of values inside a window | Low | Low | For QUANT, yes | **Done: did not help** (+0.04 pts, 68/16/58, p = 0.24; all per-dataset changes within ±2.3 pts) |
| 13 | Quick classifier check: Random Forest, LightGBM | Completes the "which classifier?" comparison | Low | Low (expect small gains) | No | Not done |
| 14 | *Efficient approach:* speed-invariant quantiles (values weighted by how much the series moves) | The same movement made at different speeds (gestures, handwriting) | Low-medium | Medium | Yes | **Done: success (small)** (+0.14 pts, p = 0.006; +0.5 where elastic methods win; ×1.33 runtime) |
| 15 | *Efficient approach:* cycle view: fold repeating series into one typical cycle | Position and rate in repeating signals (Pig, appliances) | Medium | Medium | Yes (folding is done by hand in astronomy) | **Done: did not help** (±0.00; but +1.5 pts under random shifts; ×1.19 runtime) |
| 16 | *Efficient approach:* shape-aware Fourier view (a slice of the bispectrum) | QUANT's position-free view is blind to shape | Low | Medium | Yes (bispectra used only as 2-D images for CNNs) | **Done: did not help** (+0.11, p = 0.24; but +1.0 on gestures; ×1.16 runtime) |
| 17 | *Efficient approach:* rank-uniform splits for ExtraTrees | A third of QUANT's features get lopsided random splits | Very low | Low | For ExtraTrees, yes (the transform is known for ordinary trees) | **Done: did not help** (−0.09 pts, p = 0.05; ×1.16 runtime) |
| 18 | *Efficient approach:* second opinion only when needed (gated QUANT + Hydra) | The ensemble's 4–6× cost | Low-medium | Low | For TSC, yes | **Done: success** (87.16%, +1.8 pts: 95% of the ensemble's gain at ×2.1 QUANT's time; smaller Hydra: +1.3 at ×1.27). **Multivariate** (19 UEA datasets × 5 resamples): 75.59% vs QUANT's 74.18%, +1.4 pts, 11/5/3, p = 0.033 |
| 19 | *Multivariate:* cross-channel views (magnitude, channel differences, main PCA direction) | QUANT treats each channel alone: blind to how channels move together and to sensor rotation | Low-medium | Medium | Yes | Not done |

**Suggested order:**
1. Start with **5 and 6**. They are cheap tests of the "position" diagnosis.
2. Then build **1 or 2**, the main research contribution.
3. Use **3** as a strong, safe baseline.
4. Add **4** if we can get a free GPU (Kaggle / Colab).
5. **9** and **10** are good secondary contributions (speed, new capability).
6. **11–13** are cheap add-ons. Run 11 and 12 together with 7 (they are all "new view / new feature" changes), and 13 alongside 4 and 8 (they are all classifier changes).
7. **14–18 (efficient approaches)** keep QUANT cheap. Start with **17** (a few lines of code, no extra runtime). Then do **15 and 16** (new views, run with the same code as ideas 1 and 2), **18** (reuses the ensemble code) and **14**.

---

## 1. Position-free QUANT: pooled sliding-window quantiles

> **Status: done. Success (small).** 85.36% → 85.75% (+0.39 pts) on 142 datasets × 10 splits, better on 84 / worse on 39, p = 2e-6; +1.0 pts on the 18 many-class datasets (PigAirwayPressure +10.8). Cheap: 1.4× QUANT's runtime on one core (21.5 vs 15.7 min). Under random shifts (stress test, 50%) it loses 4.1 pts instead of QUANT's 5.7, and beats QUANT by +1.8 pts. Details in [report.md](report.md).

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

> **Status: done. Success (small).** 85.36% → 85.76% (+0.40 pts), better on 78 / worse on 57, p = 0.004; +1.5 pts on many-class data. Same gain as idea 1 but 4.0× QUANT's runtime on one core (62 vs 16 min), mostly for the convolutions. The convolution quantiles *alone* are 1.7 pts worse than QUANT. Under random shifts (50%) it loses 3.3 pts instead of 5.7 (+2.8 pts vs QUANT).

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

> **Status: done. Success.** 85.36% → 87.39% (+2.03 pts), better on 99 / worse on 33, p = 4e-12; +6.4 pts on many-class data. Statistically level with MultiRocket+Hydra (87.06%) and HIVE-COTE 2 (87.53%) on the same splits. Equal weights give nearly the same: 87.25%. **Cost (measured on one core):** 91 min vs QUANT's 15.7 min (×5.8); ×4.1 with equal weights once Hydra's calibration is free (see idea 18; our first equal-weights run still cross-validated it, which made it ×5.0). Half the time goes to computing Hydra's features; the weighting's cross-validation adds 26 min for only +0.14 pts. Under random shifts (50%) it loses only 2.5 pts, as little as Hydra (2.4), and beats QUANT by +5.0 pts.

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

> **Status: done. Did not help on average.** 85.36% → 84.64% (−0.72 pts, p = 7e-5); with test-time augmentation 85.01% (−0.35 pts, p = 0.03), worse on 73 / better on 59 datasets. But it helps exactly where QUANT is weakest: PigAirwayPressure +33, PigCVP +13, +1.1 pts on the 18 many-class / few-example datasets. It hurts where position *is* the meaning: spectra (Beef −17) and gestures (−2.3). About 6× QUANT's runtime. Under random shifts (50%) it does help (+2.0 pts vs QUANT), but such shifts are rarer in the real datasets than we assumed. Possible follow-up: use it only when there are very few examples per class.

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
> **Status: done. Did not help on average.** 85.36% → 85.61% (+0.25 pts), better on 70 / worse on 59, p = 0.095 (not significant). Large gains where the series repeat: PigCVP +14.5, PigAirwayPressure +5.8, PigArtPressure +4.3, ToeSegmentation1 +3.7. Worst: PLAIDEq −5.6, DodgerLoopDayNmv −4.2. Code: `scripts/autocorr_tests.py`; results: `results/autocorr/summary.txt`.

**What.** QUANT computes quantiles on four views of each series (raw, 1st and 2nd difference, Fourier magnitude). Add views that ignore *when* things happen:
- the **autocorrelation function** (how similar the series is to itself shifted by k steps),
- **autoregressive (AR) coefficients**.

A log-spectrum would add little. Taking the log doesn't change the order of values, and quantile features and ordinary tree splits care only about order. (ExtraTrees is an exception, because it picks its split points at random; see idea 17.)

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

> **Status: done. Partial success: a clear gain on the 11 (mostly gesture) datasets it was designed on, only a small one on fresh data.** On the 11 variable-length datasets × 30 splits:
> - **Native windows instead of padding**, with the series length as a feature: 75.55% → 75.66% (+0.1 pts, p = 0.9); without the length 75.35% (−0.2 pts, p = 0.97). Large effects in both directions: GesturePebble +3.5 to +3.9, GestureMidAir −3.6 to −6.7. Stretching all series to one length gives the same average.
> - **Padded + native features together**, letting the forest choose: 77.53% (+2.0 pts), better on 10 of 11, p = 0.005. That closes 41% of the gap to MultiRocket+Hydra.
>
> - **Fresh-data check** (10 Oct): the combined version was added after seeing the first results, so we ran it unchanged on three variable-length datasets we had never used (the Asphalt road-surface sets, 30 splits). The gain shrinks to **+0.21 pts** (92.31% → 92.52%): +0.63 on AsphaltObstacles (better on 22 of 30 splits, p = 0.001), +0.07 and −0.06 on the other two. Native windows alone were worse on all three (−1.7 pts).
>
> The padding carries useful information (gesture length and speed), so it should be kept. The amount of padding does not explain why the fresh gain is small: the Asphalt series are padded at least as heavily as the gestures. All gains of 2 points or more were on gesture data, so the combined version is worth adding for gesture-like data, but should not be expected to add much elsewhere (it never cost more than 0.2 pts on any of the 14 datasets). Our simple code computes the native windows slowly (117 s vs 8 s for QUANT's features on these 11 datasets).

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

## 11. Multi-scale differences ("wavelet-like" views)

**What.** QUANT's first difference shows step-to-step change, smoothed with one fixed moving average of width 5. Add a small stack of views that show change at several speeds: the difference between moving averages of widths 2, 4, 8 and 16 (this is what Haar wavelets compute), each kept at full length. QUANT then takes its usual window quantiles on each.

**Why it should help.**
- Today QUANT sees change either at the finest scale (the differences) or at all scales but without time (the Fourier view). Wavelet-like views add "how much medium-speed change, and *where*": the time-localised frequency content the Fourier view lacks.
- It also answers the smoothing question. The paper tried only one width (5) for the first difference ("helps a little") and found no consistent gain from smoothing the other views. Trying another single width is just tuning; giving the forest several widths at once lets it pick the right scale for each dataset.

**Inspired by:**
- The [QUANT paper](https://arxiv.org/abs/2308.00928) §3.1 and Fig. 20 (smoothing the first difference helps).
- Haar wavelets for time series (Chan & Fu, ICDE 1999).
- [MiniRocket](https://arxiv.org/abs/2012.08791)'s dilations: looking at several scales is central to ROCKET-style accuracy.

**Watch out.** Each extra view adds about as many features as the series is long, so keep it to 2–3 scales or combine it with idea 9 (pruning).

**How to test.** Run 142 × 30 against QUANT, adding one scale at a time.

## 12. Order-aware counts in each window

> **Status: done. Did not help.** 85.36% → 85.40% (+0.04 pts), better on 68 / worse on 58, p = 0.24. Four counts per window of the raw series (mean crossings, local peaks, longest run above the mean, slope). Largest gains: Rock +1.8, GestureMidAirD2Eq +1.7; largest losses: Covid3MonthDiscrete −2.3, DodgerLoopDayNmv −2.0. QUANT's quantiles at several window sizes already seem to capture enough of the order. Code: `scripts/order_counts_tests.py`; results: `results/order_counts/summary.txt`.

**What.** For every window QUANT already uses, add a few cheap counts:
- how often the series crosses its window mean,
- the number of local peaks,
- the longest run above the mean,
- the slope (trend) of the window.

**Why it should help.**
- Quantiles ignore the *order* of values inside a window: shuffle a window and QUANT's features don't change. Mean crossings and peaks *do* depend on order (a smooth hill vs a zig-zag made of the same values), so they add new information at almost no cost.
- It stays "QUANT": a fixed, cheap feature set plus a forest. We only add what quantiles cannot see.
- Mean, min, max, median and IQR need not be added, because they already *are* quantiles.

**Inspired by:**
- [rSTSF](https://arxiv.org/abs/2105.14876), which counts mean crossings and values above the mean in each window.
- TSF (window slope; Deng et al., 2013).
- [catch22](https://arxiv.org/abs/1901.10200) (longest stretch above the mean).
- [MultiRocket](https://arxiv.org/abs/2102.00457)'s "longest stretch of positive values".

**How to test.** Run 142 × 30. Check that the forest actually uses the counts (feature importances) and how much extra time they cost.

## 13. Quick classifier check: Random Forest and LightGBM

**What.** Keep QUANT's features and swap only the classifier: a Random Forest with the same settings, and LightGBM (gradient boosting).

**Why it's worth a quick look, but only a quick one.**
- It's cheap to run in the same harness as ideas 4 and 8, and it completes the "which classifier?" picture for the report.
- Expect little:
  - ExtraTrees is usually as accurate as Random Forest and faster, which is why QUANT and rSTSF chose it ([Geurts et al. 2006](https://link.springer.com/article/10.1007/s10994-006-6226-1)).
  - On time series features, the [FreshPRINCE authors](https://arxiv.org/abs/2201.12048) found Rotation Forest *significantly* better than XGBoost and ridge. That's one more reason to prefer idea 8.
  - Boosting builds one tree per class per round, which is slow with 50+ classes. It can also overfit when about 200 training series meet thousands of features.
- Its best chance is the few large datasets (thousands of training series), so analyse the results by dataset size.
- If we try boosting, LightGBM is the faster choice. XGBoost's `feature_weights` option is interesting too, because it could implement idea 9's "sample big windows more often".

**How to test.** Start with the 25 + 18 test datasets, then 142 × 30. Compare accuracy and training time with ExtraTrees.

---

## Efficient approaches (ideas 14–18)

*Added 7 Oct 2026, after our five experiments. Each idea should raise accuracy while keeping QUANT cheap: at most about 1.3× its runtime by our estimate. Idea 18 instead makes our best but slowest result cheaper. We searched the literature (Oct 2026) for each one; the closest published work is named under each idea. The supporting numbers come from `scripts/idea_checks.py` (output in `results/ideas/checks.txt`). That script trains no new model; it only measures the data, QUANT's features and results we already have.*

**Why these five.** Three things we learned point the way:
1. **Changing the statistic inside each window seems to have run its course.** Two 2026 follow-ups did exactly that. [AQ-QUANT](https://doi.org/10.1109/ACCESS.2026.3712607) replaced the quantiles with area-based ones and matched QUANT's mean accuracy. [MomentQuant](https://arxiv.org/abs/2609.05136) approximated them faster, with a tiny loss. Neither raised the mean, so none of the ideas below changes the statistic inside a window.
2. **Adding a view of the series helps; replacing one doesn't** (our ideas 1, 2 and 10).
3. **The biggest gain came from a different kind of invariance.** Hydra ignores *where* patterns happen. Averaging it with QUANT gave +2 points, but at 4–6× the cost.

Almost everything QUANT computes depends on *when* things happen. The exceptions are the whole-series quantiles and the Fourier magnitudes: they ignore position, but they also ignore shape. Nothing in QUANT ignores speed. So ideas 14–16 each add one cheap view that ignores something new. Idea 17 makes the forest use QUANT's existing features better, at no cost, and idea 18 makes the +2-point ensemble cheap.

| # | Idea | What it ignores or fixes | Extra runtime vs QUANT (estimate) |
|---|---|---|---|
| 14 | Speed-invariant quantiles | how fast a movement is made | ×1.3 |
| 15 | Cycle view | where a repeating signal starts, and its rate | ×1.1–1.2 |
| 16 | Shape-aware Fourier view | position, while keeping shape | ×1.15 |
| 17 | Rank-uniform splits | wasted random splits in the forest | ×1.0 |
| 18 | Second opinion only when needed | the ensemble's cost | ×1.8–2.5 instead of ×4.1 |

## 14. Speed-invariant quantiles *(efficient approach)*

> **Status: done. Success (small).** 85.36% → 85.50% (+0.14 pts), better on 81 / worse on 52, p = 0.006. On the 23 datasets where an elastic method beats QUANT: +0.51 pts (18 better, 4 worse, p = 0.002); gestures +0.4. In the speed stress test (random time warps) it loses 7.9 pts instead of QUANT's 9.9 and is +2.2 pts ahead at the strongest warp (p = 6e-5). Cost: ×1.33 QUANT's one-core runtime. Details in [report.md](report.md).

**What.** For every window QUANT already uses, add a second set of quantiles in which each moment counts by *how much the series moves* there, not by *how long it lasts*. A stretch where the series stands still adds nothing, and a quick sweep through a range of values counts each value in that range once. Technically, these are quantiles of the values weighted by the size of each step: a level counts as often as the series passes through it.

**Why it should help.**
- **The same movement made faster or slower gives the same features.** QUANT's ordinary quantiles change when a gesture is slower, because the slow parts last longer and so get more weight. These quantiles don't, for any speed-up or slow-down inside a window. This follows from a classical result, Banach's indicatrix (1925).
- **QUANT has nothing like this, and that's where elastic methods win.** Proximity Forest compares series with time-warping distances such as DTW. On average it is 2.7 points worse than QUANT, but it beats QUANT by more than 1 point on 23 datasets (+4.2 points there). The biggest wins are on gestures (AllGestureWiimoteX +9.0, Z +8.0, Y +4.8) and handwritten words (WordSynonyms +7.7, FiftyWords +5.4).
- **It follows the "add, don't replace" rule.** Idea 10 showed that speed is sometimes the class (GestureMidAir) and sometimes noise (GesturePebble). With both kinds of quantiles, the forest can pick per dataset. Flat stretches, such as constant padding, add nothing.

**Why it's new.** In TSC, speed-invariant features exist only as path signatures or iterated sums ([FRUITS](https://arxiv.org/abs/2311.14549), DAMI 2024) and as topological summaries ([persistence curves](https://arxiv.org/abs/2003.06462), 2020). Our weighted quantiles carry much the same information as a topological "Betti curve". But they cost one weighted sort per window, the same as QUANT's own sort, and they fit straight into QUANT's windows. We found no paper that uses them.

**How to build it.**
1. Smooth the series the way QUANT smooths its first difference (moving average of width 5), so noise doesn't dominate.
2. Give each step between neighbouring points a weight equal to its size.
3. In each window, sort the step midpoints and read off the weighted quantiles, as many as QUANT takes there.
4. In a flat window, fall back to ordinary quantiles.

**Cost.** One more view: about ×1.3 if it's added for the raw series only.

**How to test.**
- Run 142 datasets × 10 resamples against QUANT, as for our other ideas. Look separately at the gestures, the handwriting datasets and the 23 datasets that Proximity Forest wins.
- Add a "speed stress test" like our shift test: stretch every series by a random factor (0.7–1.3), then pad or crop it back to its length. QUANT should lose accuracy, and this version much less.

## 15. Cycle view: fold repeating series into one typical cycle *(efficient approach)*

> **Status: done. Did not help on average.** ±0.00 pts (61 better / 65 worse), also none on the 36 repeating datasets. Big gains on a few (ToeSegmentation2 +5.2, PickupGestureWiimoteZ +2.2, PigArtPressure +1.6), losses on shape outlines (Herring −2.8). Under random shifts (50%) it loses 4.2 pts instead of 5.7 (+1.5 vs QUANT, p = 0.003), thanks to the standard starting point rather than the folding. Cost: ×1.19.

**What.** Many series repeat themselves: heartbeats, breathing, engine vibration, appliances switching on and off. For these:
1. Find the length of one cycle from the Fourier magnitudes QUANT already computes (their strongest peak), and check it with the autocorrelation.
2. Cut the series into cycles and take the median cycle. This gives one clean "typical cycle", stretched to a fixed length (e.g. 64 points) and rotated so it always starts at the same point of the cycle, e.g. its lowest value.
3. Run QUANT's windows on this typical cycle, and on the spread between cycles (how much the individual cycles differ at each point).
4. Add two numbers as features: the cycle length, and how strongly the series repeats.

**Why it should help.**
- **QUANT falls furthest behind on repeating data.** On the 36 datasets whose series repeat at least 3 times, QUANT trails HC2 by 3.9 points, against 1.5 on the other 106 (p = 0.0007). Against MultiRocket+Hydra the difference is smaller and not significant (−2.2 vs −1.4). The worst cases:
  - PigCVP: −21 points vs MultiRocket+Hydra, with about 13 cycles per series;
  - PigArtPressure: −9 vs HC2;
  - RefrigerationDevices: −7 vs HC2.
- **The typical cycle doesn't depend on where the recording starts.** That is the weakness we measured most. In the Pig recordings the beats start at arbitrary points, and teaching QUANT this through augmentation was worth +13 to +33 points there (idea 5). The cycle view gets the same invariance with no extra training copies.
- **Averaging removes noise.** The median of 10 cycles is much cleaner than any single cycle.
- **It separates rhythm from shape.** The heart rate goes into one feature (the cycle length), and the beat's shape into the folded view.

**Why it's new.** Folding is standard in astronomy. The creators of UCR's StarLightCurves dataset folded every light curve by its period and aligned the cycles before publishing it ([Rebbapragada et al. 2009](https://arxiv.org/abs/0905.3428)). ECG analysis uses average-beat templates in the same way. But no general-purpose TSC method folds automatically, and none of the methods in the bake-off has a folded view.

**Cost.** Folding takes one pass over the series, and the typical cycle is short. That gives about 600 extra features (+20% for a typical dataset, less for long series), so about ×1.1–1.2.

**Risk.** On series that don't repeat, the typical cycle is meaningless. The "how strongly it repeats" feature lets the forest ignore it. Alternatively, build the view only when the training series repeat.

**How to test.** Run 142 × 10 against QUANT, focusing on the 36 repeating datasets and the Pig data. Variants:
- cut fixed-length cycles, or cut at landmarks such as each cycle's lowest point (this copes better with a changing heart rate);
- for series with fewer than 3 cycles, treat the whole series as one cycle. This gives closed outlines, such as leaf or word shapes, a standard starting point.

## 16. Shape-aware Fourier view: a slice of the bispectrum *(efficient approach)*

> **Status: done. Did not help on average.** +0.11 pts, p = 0.24; nothing on the many-class datasets (−0.02), but +1.0 pts on the gestures (AllGestureWiimoteZ +3.4, GestureMidAirD2 +2.7). Barely more robust to random shifts than QUANT (+0.3). Cost: ×1.16.

**What.** QUANT's Fourier view keeps only the *size* of each frequency, not its *timing* (its phase). That's why it ignores position: moving a pattern changes only the phases. But it also makes the view blind to shape. For example, a wave that rises fast and falls slowly, and its mirror image in time, have identical Fourier magnitudes. One product of Fourier coefficients, X(f)·X(f)·conj(X(2f)), fixes this. When a pattern moves, the phase changes cancel out, but the product still records how frequency f lines up with frequency 2f, and that is shape. Add its real and imaginary parts as one more view (each about a quarter as long as the series), with QUANT's usual windows and quantiles.

**Why it should help.**
- **It targets the gap between QUANT and Hydra.** Hydra knows "which shapes occur, anywhere". QUANT's Fourier view knows "which frequencies occur, anywhere", but not the shape. In our shift test QUANT still lost 5.7 points against Hydra's 2.4, so its position-free Fourier view wasn't enough.
- **Both parts mean something simple.**
  - The real part tracks sharp peaks vs sharp troughs (skewness).
  - The imaginary part tracks a fast rise with a slow decay vs the reverse (time asymmetry), as in heartbeats and pressure waves. Econometrics uses this part to test whether a series looks the same backwards (Hinich & Rothman 1998).

  The Fourier magnitude sees neither.
- **Fourier views are QUANT's best addition.** The QUANT paper found that the Fourier magnitude gave the largest per-dataset gains of all the views it added.

**Why it's new.** The full two-dimensional bispectrum is used in signal processing (EEG, machine faults). It has also been used as an image for deep CNNs ([Parker et al. 2020](https://arxiv.org/abs/2003.02353)), which is powerful but slow. A one-dimensional slice summarised by QUANT's quantiles, next to the magnitudes, has not been tried.

**Cost.** It is computed from the Fourier transform QUANT already has, with one multiplication per frequency. The new view adds about 15% more features: ×1.15.

**How to test.** Run 142 × 10 against QUANT, plus the shift stress test (it should be as robust as the magnitude view) and the many-class datasets. Variants:
- shrink the values with a signed cube root, so they are on the series' own scale;
- add the next slice, X(f)·X(2f)·conj(X(3f)).

## 17. Rank-uniform splits: quantile features deserve quantile thresholds *(efficient approach)*

> **Status: done. Did not help.** −0.09 pts, worse on 76 / better on 56, p = 0.05. The gain grows only slightly with feature skew (correlation +0.17). Lopsided random splits are not what limits QUANT's forest: choosing the best of ~300 candidate splits already makes up for them. Cost: ×1.16.

**What.** Before training the forest, replace each feature value with its rank among the training series, scaled from 0 (smallest) to 1 (largest). Map test values onto the same scale by interpolating between the training values. The features and the forest stay the same.

**Why it should help.**
- **ExtraTrees chooses its split points at random.** It picks them uniformly between the smallest and largest value in the branch ([Geurts et al. 2006](https://link.springer.com/article/10.1007/s10994-006-6226-1); scikit-learn does the same). Often a few series have much larger values than the rest; this is common for differences and Fourier magnitudes. Then most random split points cut off just those few series. Such splits teach the tree little and usually lose to another feature's split, so the skewed feature is mostly wasted.
- **This affects a third of QUANT's features.** For 35% of them, a random split puts on average fewer than 10% of the training series on the smaller side. The share is 43% for the second-difference features and 77% of all features in PigCVP. It is a little higher on the many-class datasets (42% vs 34%). After ranking, every random split cuts off a random share instead: 25% on average for the smaller side.
- **It costs nothing.** Ordinary decision trees try every split point, so ranking makes no difference to them. ExtraTrees tries one random point per feature, and ranking changes which ones it tries, while keeping exactly the same possible splits. More balanced splits also give shallower trees, so training may even get faster.
- It also corrects a remark under idea 7: for ExtraTrees, a log-spectrum *would* change the result, for this same reason.

**Caveats.**
- How skewed a dataset's features are does not line up with where QUANT loses accuracy (correlation −0.09 with its gap to MultiRocket+Hydra, p = 0.3). So this is a general fix, not a cure for a known weakness, and the gain may be small.
- Random split points in value space are good at isolating outliers, which can help with rare classes. Ranking loses some of that.

**Why it's new.** [Galili & Meilijson (2016)](https://arxiv.org/abs/1611.04561) proposed the same transform for ordinary trees and random forests. There it has a smaller effect: it only changes where a split lands between two neighbouring values. For ExtraTrees it decides which splits get tried at all. We found no study of that, and no one has measured how skewed QUANT's features are.

**Cost.** One sort per feature on the training set takes seconds in total. There are no extra features: ×1.0, or less if the trees get shallower.

**How to test.** Run 142 × 10 against QUANT, plus a one-core timing run. If it helps, check whether it adds to the gains of ideas 1, 15 and 16.

## 18. Second opinion only when needed: gated QUANT + Hydra *(efficient approach)*

> **Status: done. Success.** A two-level gate asks Hydra only about test series where QUANT's two most likely classes are close, and only on datasets where at least 10–20% of test series are unsure. Settings were chosen on the other half of the datasets. It gives 87.16% (+1.80 pts, 83/39/20, p = 1e-11), 95% of the full ensemble's +1.89 pts, at ×2.1 QUANT's one-core time instead of ×4.1; +6.1 pts on many-class data. On the same splits it is level with MultiRocket+Hydra (87.06%; +0.10, p = 0.43) and 0.37 pts behind HIVE-COTE 2 (p = 0.045). Checked on unseen data: with its setting chosen on the first 60 datasets (the gate was designed during the run), it kept 97% of the gain on the other 82 at ×2.0. With a smaller Hydra (16 kernel groups): +1.27 pts at ×1.27. Building it showed that idea 3's "equal weights" version still used the cross-validation to calibrate Hydra's scores (×5.0, not ×4.2); a free leave-one-out calibration gives the same 87.25%.

**What.** Keep our best result, the QUANT + Hydra ensemble (+2.0 points), but pay for Hydra only where QUANT is unsure:
1. **Per dataset:** skip Hydra when the training data say QUANT will do well. For example, QUANT's cross-validated accuracy on the training set is very high, or the dataset has few classes and many examples per class, where our weakness tests found QUANT strong.
2. **Per test series:** QUANT predicts first. Only the series where QUANT's top two class probabilities are close get Hydra's features and the averaged vote.
3. Optionally, use a smaller Hydra with fewer kernels.

**Why it should help.**
- **The gain is concentrated; the cost is not.** Looking back at our results, 20 datasets gave 66% of the ensemble's gain while costing only 6% of Hydra's extra time. 40 datasets gave 94% of the gain for 13% of the time. A gate that finds those datasets keeps most of the gain for little cost.
- **Even a crude rule works.** Take "use Hydra only if there are at least 10 classes, or at most 10 training series per class", which matches the weaknesses in paper Section IV. It keeps 66% of the gain: 86.62%, against 85.36% for QUANT alone and 87.25% for the full ensemble. It takes 27.5 min on one core instead of 65.6, so 1.75× QUANT's time instead of 4.2×. We tried three such rules and this was the best, so treat the number as optimistic.
- **Test series are two thirds of all series** and take 59% of Hydra's feature time. So skipping Hydra on test series where QUANT is confident saves much of the remaining cost.
- **QUANT knows when it is unsure.** Its probabilities are among the best of the top methods (bake-off AUROC 0.956; log-loss 0.53, against 4.6 for MultiRocket+Hydra), and a reliable confidence is what a gate needs.

**Why it's new.** The [Meta-Learning Gap](https://arxiv.org/abs/2512.06666) paper (2025) tested six ways to combine Hydra and QUANT: feature concatenation, two kinds of stacking, and weighted averaging. All of them run both models on every series. Cascades that call an expensive model only when a cheap one is unsure are standard in other fields, e.g. Viola–Jones face detection (2001), but we found none in TSC.

**Cost.** The goal is to keep at least 80% of the +2.0 points at about 2× QUANT's time.

**How to test.**
- Use the same 142 × 10 setup as the ensemble.
- Set the thresholds on training data only (cross-validation), never on test results.
- Plot accuracy against one-core time for several thresholds, next to QUANT (×1), the full ensemble (×4.1) and MultiRocket+Hydra (about 1 h 15 min).


## 19. Cross-channel views for multivariate data

**What.** On multivariate data, QUANT computes its features on each channel alone and joins them. Add views that mix channels, next to QUANT's own, with QUANT's usual windows and quantiles on top:
- **Magnitude** √(x² + y² + z²) for each 3-axis sensor: unchanged when the sensor is rotated.
- **Pairwise channel differences**: how channels move relative to each other (with many channels, 15 fixed random pairs).
- **Main direction**: the series projected on its first PCA component, fitted on the training data.

**Why it should help.**
- Most UEA datasets are motion sensors (X/Y/Z axes), where the class depends on how the axes move *together*. QUANT never sees that.
- The same movement recorded with the sensor at a different angle changes every channel. QUANT's features change with it; the magnitude does not.
- It follows our "add a view, don't replace" rule.

**How to test.**
- Run QUANT, Hydra, idea 18 and QUANT + cross-channel views on about 18 small/medium UEA datasets × 5 resamples.
- **Rotation stress test:** randomly rotate the sensor axes of every series. QUANT should lose accuracy, and the version with the cross-channel views much less.

---

## Your suggestions: verdicts

**The rule we used:** add only what quantiles *can't* see (the order inside a window, patterns anywhere in the series, change at several scales) and keep QUANT's recipe of a fixed, cheap feature set plus a forest. Importing whole feature sets from other methods would just rebuild those methods, not improve QUANT.

| Suggestion | Verdict | Why |
|---|---|---|
| XGBoost / LightGBM | Quick check (idea 13) | FreshPRINCE found Rotation Forest significantly better than XGBoost on time series features; slow with many classes; may help the largest datasets |
| Random Forest | Quick check (idea 13) | Usually about as accurate as ExtraTrees but slower; a cheap drop-in test |
| Ridge regression | No | Already measured: 3–5 points worse with few examples and 11.7 points worse on many-class data |
| DeepSurv | No | It's a survival-analysis model: it predicts *how long until an event* (e.g. patient survival) from tabular data. It isn't a classifier and isn't built for time series |
| A new view beside raw, 1st/2nd difference and DFT | Yes | Autocorrelation and AR coefficients (idea 7); multi-scale differences / wavelets (idea 11). QUANT already uses four views: the raw series counts as one |
| Different moving averages | Yes, as a multi-scale stack (idea 11) | The paper tried only width 5 on the first difference; one other width would just be tuning |
| Simple stats (mean, std, slope, min/max, IQR, skewness, kurtosis) | Only the slope (idea 12) | Min, max, median and IQR *are* quantiles; std, skewness and kurtosis are nearly implied by them ([MomentQuant](https://arxiv.org/abs/2609.05136) even goes the other way) |
| catch22 | No, but borrow its ideas | It would turn QUANT into DrCIF, which QUANT already matches; slow. Its useful parts (autocorrelation, order) are in ideas 7 and 12 |
| TSFresh | No | Hundreds of features would make it FreshPRINCE, which QUANT already beats by about 0.9 points; very slow |
| Counts (mean crossings, peaks, values above the mean) | Yes (idea 12) | Order-aware and cheap. "Values above the mean" is mostly already covered by the quantiles |
| Spectral band power, autocorrelation at lags | Autocorrelation yes (idea 7); band power no | The quantiles of QUANT's Fourier view already describe each frequency band |
| Wavelets | Yes (idea 11) | Adds frequency content that is located in time |
| Convolutional features (ROCKET's proportion of positive values) | Yes, QUANT-style (idea 2) | Take quantiles of the convolution output instead of copying ROCKET's statistic |
| Shapelets, dictionary words, DTW distances to prototypes | No, as features | These are separate method families and slow. Combine their *predictions* instead (idea 3) |
| Learned embeddings (CNN / self-supervised) | No | Needs a GPU, and [RocketPFN](https://arxiv.org/abs/2606.21786) found such embeddings lose to random features on UCR. That would be a different research project |

---

## Also considered (smaller or already done elsewhere)

- **Class-weighted trees.** Small fix for the 0.9-point extra loss on imbalanced data.
- **Raw + normalised quantiles side by side.** Normalising costs 3.1 points on average but gains 2–6 points on gesture data.
- **De-duplicated evaluation.** 25 datasets have train/test copies, so results should be reported with and without them. This fixes the *evaluation*, not QUANT.
- **Faster quantile computation.** Already published as [MomentQuant](https://arxiv.org/abs/2609.05136) (2026), and the transform is only ~22% of the runtime.
- **Multivariate QUANT and regression.** Already exist (authors' code and aeon's `QUANTRegressor`), so there's little novelty.
- **Deep foundation models as feature extractors (Mantis, MOMENT).** [RocketPFN](https://arxiv.org/abs/2606.21786) shows they lose to simple random features on UCR.
- **"When" features: where inside a window the area builds up.** Already published as part of [AQ-QUANT](https://doi.org/10.1109/ACCESS.2026.3712607) (IEEE Access 2026). It replaced QUANT's quantiles with area quantiles and matched QUANT's mean accuracy, but did not beat it.
- **A more compact summary per window (e.g. L-moments instead of hundreds of quantiles).** This would make long series cheaper. But like AQ-QUANT and MomentQuant, it only changes the statistic inside each window, and that hasn't raised mean accuracy so far.
