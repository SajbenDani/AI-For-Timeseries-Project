"""Additions to QUANT used by the improvement tests (scripts/improvement_tests.py, ideas from ideas.md).

QUANT itself (codebase/quant) is not changed. Like QUANT's transform, nothing here looks at the labels, so features
can be computed once for all series of a dataset.

  idea 1   pooled_features   quantiles of sliding windows, summarised over all positions (position-free)
  idea 2   conv_quantiles    quantiles of random-convolution outputs over the whole series (shape-aware, position-free)
  idea 5   AUGMENTATIONS     shifted and stretched copies of each series
  idea 10  varlen_quant      QUANT on series of different lengths, without padding
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codebase" / "quant" / "code"))
from quant import Quant, make_intervals  # noqa: E402

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
