"""Additions to QUANT used by the improvement tests (scripts/improvement_tests.py, ideas from ideas.md).

QUANT itself (codebase/quant) is not changed. Like QUANT's transform, nothing here looks at the labels, so features
can be computed once for all series of a dataset.

  idea 1   pooled_features   quantiles of sliding windows, summarised over all positions (position-free)
  idea 2   conv_quantiles    quantiles of random-convolution outputs over the whole series (shape-aware, position-free)
  idea 5   AUGMENTATIONS     shifted and stretched copies of each series
  idea 10  varlen_quant      QUANT on series of different lengths, without padding
  idea 14  speed_quantiles   window quantiles weighted by how much the series moves (ignores speed)
  idea 15  cycle_view        QUANT on the typical cycle of a repeating series (ignores where the recording starts)
  idea 16  bispectrum_view   a slice of the bispectrum: like the Fourier magnitudes, but it sees shape
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codebase" / "quant" / "code"))
from quant import IntervalModel, Quant, make_intervals  # noqa: E402

REPRESENTATIONS = Quant().representation_functions  # raw, smoothed 1st difference, 2nd difference, |FFT|
LEVELS = torch.linspace(0, 1, 9)  # quantile levels for the pooled and convolution features


def quantiles(Z, levels):
    """Quantiles along the last axis with linear interpolation (torch.quantile's rule, without its size limit)."""
    Z = Z.sort(-1).values
    pos = levels * (Z.shape[-1] - 1)
    lo, hi = pos.floor().long(), pos.ceil().long()
    return Z[..., lo] + (Z[..., hi] - Z[..., lo]) * (pos - lo)


def batched(function, X, batch_size):
    X = torch.as_tensor(X, dtype=torch.float32)
    return torch.cat([function(X[s:s + batch_size]) for s in range(0, len(X), batch_size)]).numpy()


# == idea 1: position-free pooled window quantiles =====================================================

def pooled_features(X, depth=6):
    """For windows of 1/2, 1/4, ... 1/32 of the series, slide each window over the whole series (step = a quarter
    window) and take its quantiles, every second one relative to the window mean as in QUANT. Each quantile is then
    summarised over all positions by its maximum, minimum and mean. "Somewhere there is a stretch whose 90th
    percentile is this high" no longer depends on where that stretch is. The FFT view is skipped: it is already
    position-free. X: (n_series, length) -> (n_series, up to 3 x 5 x 27 = 405) features."""

    def pool(X):
        features = []
        for represent in REPRESENTATIONS[:3]:
            Z = represent(X.unsqueeze(1))[:, 0]
            for j in range(1, depth):
                width = Z.shape[-1] // 2 ** j
                if width < 4:
                    break
                windows = Z.unfold(-1, width, max(1, width // 4))  # (n, positions, width)
                q = quantiles(windows, LEVELS)
                q[..., 1::2] -= windows.mean(-1, keepdim=True)
                features += [q.amax(1), q.amin(1), q.mean(1)]
        return torch.cat(features, -1)

    return batched(pool, X, 512)


# == idea 2: quantiles of random convolutions ("QUANT meets ROCKET") ====================================

# MiniRocket's 84 kernels: length 9, weight 2 at three positions and -1 elsewhere (they sum to zero)
KERNELS = torch.tensor([[2.0 if i in c else -1.0 for i in range(9)] for c in itertools.combinations(range(9), 3)])


def conv_quantiles(X, num_outputs=288, seed=0):
    """Convolve each series (and its first difference) with MiniRocket's kernels at dilations 1, 2, 4, ... and
    describe every output series by 9 quantiles over its whole length. A kernel's output shows how well a small
    shape matches at each time, so the quantiles say how strongly and how often the shape occurs, wherever it is.
    The maximum (ROCKET) is the 100% quantile and MiniRocket's "proportion of positive values" for any bias is a
    point of the same distribution, so quantiles contain both. A fixed random subset of `num_outputs`
    (view, dilation, kernel) combinations keeps the cost near QUANT's: 288 x 9 = 2,592 features at most."""
    length = X.shape[-1]
    dilations = 2 ** np.arange(int(np.log2(max(length - 1, 8) / 8)) + 1)
    combos = [(v, int(d), k) for v in range(2) for d in dilations for k in range(len(KERNELS))]
    chosen = np.random.default_rng(seed).choice(len(combos), min(num_outputs, len(combos)), replace=False)
    groups = {}
    for c in sorted(chosen):
        v, d, k = combos[c]
        groups.setdefault((v, d), []).append(k)

    def convolve(X):
        views = (X.unsqueeze(1), X.diff().unsqueeze(1))
        return torch.cat([quantiles(F.conv1d(views[v], KERNELS[ks].unsqueeze(1), dilation=d, padding=4 * d),
                                    LEVELS).flatten(1) for (v, d), ks in groups.items()], 1)

    return batched(convolve, X, max(1, int(2e7 // (len(chosen) * length))))


# == idea 5: shifted and stretched copies ==============================================================

def shift(X, fraction):
    """Move every series by fraction x its length (later if positive); the gap repeats the edge value."""
    length = X.shape[1]
    return X[:, np.clip(np.arange(length) - int(round(fraction * length)), 0, length - 1)]


def stretch(X, factor):
    """Play every series `factor` times slower (>1) or faster (<1) around its centre, keeping its length;
    linear interpolation, edges repeat."""
    length = X.shape[1]
    t = np.clip((np.arange(length) - (length - 1) / 2) / factor + (length - 1) / 2, 0, length - 1)
    lo = np.floor(t).astype(int)
    hi = np.minimum(lo + 1, length - 1)
    w = (t - lo).astype(X.dtype)
    return X[:, lo] * (1 - w) + X[:, hi] * w


AUGMENTATIONS = {
    "shift +10%": lambda X: shift(X, 0.1),
    "shift -10%": lambda X: shift(X, -0.1),
    "stretch x1.1": lambda X: stretch(X, 1.1),
    "stretch x0.9": lambda X: stretch(X, 0.9),
}


# == idea 10: QUANT for variable-length series ========================================================

def varlen_quant(series, reference_length, depth=6, div=4, min_length=9):
    """QUANT without padding. Each window is the same *fraction* of every series (its first half, its second
    quarter, ...), so it covers the same part of a short and a long gesture. Which windows exist and how many
    quantiles each gets are fixed by `reference_length`, so all series get the same features. FFT magnitudes are
    scaled by reference_length / length so amplitudes are comparable across lengths. Every step is QUANT's own
    code, so for series of length reference_length the features are exactly QUANT's. The few series shorter than
    `min_length` (e.g. 2 points) are first stretched to it by linear interpolation, so every view is non-empty.
    series: list of 1-d arrays -> (n_series, n_features) float32."""
    series = [s if len(s) >= min_length else np.interp(np.linspace(0, len(s) - 1, min_length), np.arange(len(s)), s)
              for s in series]
    plan = []  # per view: its reference length, reference windows and number of quantiles per window
    for represent in REPRESENTATIONS:
        ref = represent(torch.zeros(1, 1, reference_length)).shape[-1]
        windows = make_intervals(ref, depth)
        sizes = (windows[:, 1] - windows[:, 0]).tolist()
        plan.append((ref, windows, [1 + (m - 1) // div for m in sizes]))
    lengths = np.array([len(s) for s in series])
    out = [None] * len(series)
    for length in np.unique(lengths):
        members = np.flatnonzero(lengths == length)
        X = torch.tensor(np.stack([series[i] for i in members]), dtype=torch.float32).unsqueeze(1)
        features = []
        for index, (represent, (ref, windows, counts)) in enumerate(zip(REPRESENTATIONS, plan)):
            Z = represent(X)
            if index == 3:
                Z = Z * (reference_length / length)
            n = Z.shape[-1]
            for (a, b), count in zip(windows.tolist(), counts):
                a = min(a * n // ref, n - 1)
                b = max(b * n // ref, a + 1)
                part = Z[..., a:b]
                if count == 1:
                    features.append(part.quantile(torch.tensor([0.5]), dim=-1).permute(1, 2, 0).flatten(1))
                else:
                    q = part.quantile(torch.linspace(0, 1, count), dim=-1).permute(1, 2, 0)
                    q[..., 1::2] = q[..., 1::2] - part.mean(-1, keepdims=True)
                    features.append(q.flatten(1))
        for row, i in zip(torch.cat(features, -1).numpy(), members):
            out[i] = row
    return np.stack(out)


# == idea 14: speed-invariant quantiles ===============================================================

def interpolate(q, xp, fp):
    """np.interp for every row: q (n, k), xp (n, m) increasing, fp (n, m); values outside xp take the edge."""
    i = torch.searchsorted(xp, q).clamp(1, xp.shape[-1] - 1)
    x0, x1, f0, f1 = xp.gather(-1, i - 1), xp.gather(-1, i), fp.gather(-1, i - 1), fp.gather(-1, i)
    return f0 + ((q - x0) / (x1 - x0).clamp_min(1e-12)).clamp(0, 1) * (f1 - f0)


def speed_quantiles(X, depth=6, div=4):
    """QUANT's windows and quantile counts on the raw series, but each value counts by how much the series moves
    there, not by how long it stays there: the quantiles are those of the midpoints of all steps between
    neighbouring points, each weighted by the size of its step. Playing a window faster or slower changes how long
    each part lasts but not how far the series moves, so these quantiles stay the same (the weighted distribution
    is the series' level-crossing count, Banach's indicatrix). The series is first smoothed as QUANT smooths its
    first difference (moving average of width 5), so that noise does not dominate the weights, and every second
    quantile is taken relative to the weighted window mean, as in QUANT. A window where the series stands still
    falls back to plain quantiles. X: (n_series, length) -> (n_series, about 3 x length) features."""

    def speed(X):
        S = F.avg_pool1d(F.pad(X.unsqueeze(1), (2, 2), "replicate"), 5, 1)[:, 0]
        step = S.diff()
        mid = (S[:, 1:] + S[:, :-1]) / 2
        weight = step.abs()
        weight = weight + 1e-6 * weight.mean(-1, keepdim=True) + 1e-30
        features = []
        for a, b in make_intervals(mid.shape[-1], depth).tolist():
            m, w = mid[:, a:b], weight[:, a:b]
            if b - a == 1:
                features.append(m)
                continue
            count = 1 + (b - a - 1) // div
            m_sorted, order = m.sort(-1)
            w_sorted = w.gather(-1, order)
            total = w_sorted.sum(-1, keepdim=True)
            centre = ((w_sorted.cumsum(-1) - w_sorted / 2) / total).contiguous()  # where each value's weight sits
            levels = torch.tensor([0.5]) if count == 1 else torch.linspace(0, 1, count)
            q = interpolate(levels.expand(len(m), count).contiguous(), centre, m_sorted)
            if count > 1:
                q[:, 1::2] -= (m * w).sum(-1, keepdim=True) / total
            features.append(q)
        return torch.cat(features, -1)

    return batched(speed, X, 256)


# == idea 15: cycle view ===============================================================================

def find_period(X, min_cycles=3, min_strength=0.3):
    """Length of one cycle of every series (in points, with sub-point precision) and how strongly it repeats: the
    first autocorrelation peak that reaches 90% of the highest one, at a lag that fits at least `min_cycles` times
    into the series. Only peaks after the autocorrelation has first turned negative count, as in pitch detection:
    a repeating series swings below its mean half a cycle later, a drifting one (e.g. a random walk) does not.
    Series without such a peak of at least `min_strength` get their whole length as period and strength 0."""
    n, length = X.shape
    size = 1 << int(np.ceil(np.log2(2 * length)))
    acf = np.fft.irfft(np.abs(np.fft.rfft(X - X.mean(1, keepdims=True), size)) ** 2)[:, :length]
    acf = acf / np.maximum(acf[:, :1], 1e-12)
    period, strength = np.full(n, float(length)), np.zeros(n)
    hi = length // min_cycles
    if hi < 4:
        return period, strength
    lags = np.arange(3, hi + 1)
    mid = acf[:, 3:hi + 1]  # lags 3 .. hi
    after_dip = lags[None, :] > (acf < 0).argmax(1)[:, None]  # argmax = first negative lag (0 if never negative)
    after_dip &= (acf < 0).any(1)[:, None]
    heights = np.where((mid > acf[:, 2:hi]) & (mid >= acf[:, 4:hi + 2]) & after_dip, mid, -np.inf)
    best = heights.max(1)
    for j in np.flatnonzero(best >= min_strength):
        lag = 3 + np.flatnonzero(heights[j] >= 0.9 * best[j])[0]
        a, b, c = acf[j, lag - 1:lag + 2]
        period[j] = lag + (np.clip(0.5 * (a - c) / (a - 2 * b + c), -0.5, 0.5) if a - 2 * b + c < 0 else 0.0)
        strength[j] = b
    return period, strength


def fold(x, period, bins):
    """Median and spread (interquartile range) over all whole cycles of one series, each cycle resampled to `bins`
    points. A series without a cycle (period = its length) is one cycle: itself, resampled, with spread 0."""
    length = len(x)
    if period >= length:
        return np.interp(np.linspace(0, length - 1, bins), np.arange(length), x), np.zeros(bins)
    t = np.arange(int(length // period))[:, None] * period + np.arange(bins)[None, :] * (period / bins)
    q25, q50, q75 = np.percentile(np.interp(t, np.arange(length), x), [25, 50, 75], axis=0)
    return q50, q75 - q25


def cycle_view(X, bins=64):
    """Fold every series into its typical cycle: find the cycle length (find_period), take the median of all whole
    cycles, each resampled to `bins` points, and rotate it so the cycle's fundamental (its first Fourier component)
    has phase 0. Every typical cycle then starts at the same point of its shape, wherever the recording started,
    and averaging the cycles removes noise. Features: QUANT's on the typical cycle, QUANT's raw-view quantiles of
    the spread between cycles (rotated the same way), the cycle length as a fraction of the series and how strongly
    the series repeats. Series that do not repeat are their own single cycle. X: (n, length) -> (n, ~850) features."""
    X = np.asarray(X, dtype=np.float64)
    n, length = X.shape
    bins = min(bins, length)
    period, strength = find_period(X)
    cycle, spread = np.empty((n, bins), np.float32), np.empty((n, bins), np.float32)
    for j in range(n):
        c, s = fold(X[j], period[j], bins)
        shift = int(round(np.angle(np.fft.rfft(c)[1]) * bins / (2 * np.pi)))
        cycle[j], spread[j] = np.roll(c, shift), np.roll(s, shift)
    C, S = torch.from_numpy(cycle).unsqueeze(1), torch.from_numpy(spread).unsqueeze(1)
    return np.hstack([Quant().fit_transform(C, None).numpy(), IntervalModel(bins).fit_transform(S, None).numpy(),
                      np.column_stack([period / length, strength]).astype(np.float32)])


# == idea 16: shape-aware Fourier view (a slice of the bispectrum) =====================================

def bispectrum_view(X, depth=6, div=4):
    """QUANT's Fourier view keeps only the size of each frequency, which ignores where a pattern is but is also blind
    to its shape (a series and its mirror image in time have the same magnitudes). The product
    B(f) = X(f) X(f) conj(X(2f)) is unchanged when the pattern moves (the phase shifts cancel), but it records how
    frequency f lines up with 2f, which is shape: its real part follows peaks vs troughs (skewness), its imaginary
    part a fast rise vs a fast fall (time asymmetry). Both parts, shrunk by a signed cube root so they are on the
    series' own scale, become two more views with QUANT's windows and quantiles.
    X: (n_series, length) -> (n_series, about 1.5 x length) features."""

    def bispectrum(X):
        R = torch.fft.rfft(X)
        f = torch.arange(1, (R.shape[-1] - 1) // 2 + 1)
        B = R[:, f] ** 2 * R[:, 2 * f].conj()
        views = [(v.sign() * v.abs().pow(1 / 3)).unsqueeze(1) for v in (B.real, B.imag)]
        return torch.cat([IntervalModel(len(f), depth, div).fit_transform(v, None) for v in views], -1)

    return batched(bispectrum, X, 512)
