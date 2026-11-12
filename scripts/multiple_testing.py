"""Holm correction over all the comparisons with plain QUANT in the paper's Table III.

Usage: python scripts/multiple_testing.py      # -> results/checks/multiple_testing.txt

We tested many ideas, and with many tests some come out "significant" by luck. Holm's method corrects for this: the
smallest of m p-values is multiplied by m, the next by m - 1, and so on, and a result counts only if its corrected
p-value is still below 0.05. The p-values are read from the summary files the other analysis steps write (run those
first), so this step trains nothing and needs no data.
"""

import re

from analysis_utils import ROOT

R = ROOT / "results"
# Table III row -> (summary file, text on the line that holds its Wilcoxon p against QUANT). Several texts: go to the
# first line with the first text, from there to the first line with the next one, and so on.
ROWS = {
    "3: + Hydra, weighted": ("improvements/summary.txt", "QUANT + Hydra (weighted)         (142 datasets)"),
    "3: + Hydra, equal weights": ("efficient/summary.txt", "QUANT + Hydra with every test series asked"),
    "18: + Hydra, two-level gate": ("efficient/summary.txt", ("two-level gate, settings chosen on the other half", "vs QUANT")),
    "18: + small Hydra, gate": ("efficient/summary.txt", ("Hydra-16 alone", "two-level gate, settings chosen on the other half",
                                                         "vs QUANT")),
    "1 + 14 + 18: stacked": ("stacking/summary.txt", "Ideas 1 + 14 + 18 (stacked)"),
    "1: + pooled quantiles": ("improvements/summary.txt", "QUANT + pooled quantiles         (142 datasets)"),
    "2: + convolution quantiles": ("improvements/summary.txt", "QUANT + conv quantiles           (142 datasets)"),
    "14: + speed quantiles": ("efficient/summary.txt", "idea 14: QUANT + speed quantiles"),
    "9: keep top 25% of features": ("pruning/summary.txt", "QUANT, 25% of features"),
    "10: padded + native": ("improvements/summary.txt", "QUANT (padded + native windows) vs padded"),
    "5: + augmentation": ("improvements/summary.txt", "QUANT + augmentation             (142 datasets)"),
    "5: + augm. + test-time avg.": ("improvements/summary.txt", "QUANT + augmentation + TTA       (142 datasets)"),
    "10: native windows + length": ("improvements/summary.txt", "QUANT (native windows) + length vs padded"),
    "15: + cycle view": ("efficient/summary.txt", "idea 15: QUANT + cycle view"),
    "16: + bispectrum view": ("efficient/summary.txt", "idea 16: QUANT + bispectrum view"),
    "17: rank-uniform splits": ("efficient/summary.txt", "idea 17: QUANT, rank-uniform splits"),
    "7: + autocorrelation, AR": ("autocorr/summary.txt", "change"),
    "11: + multi-scale differences": ("multiscale/summary.txt", "change"),
    "12: + order-aware counts": ("order_counts/summary.txt", "change"),
    "13: Random Forest": ("classifiers/summary.txt", ("QUANT features + Random Forest", "change")),
    "13: LightGBM": ("classifiers/summary.txt", ("QUANT features + LightGBM", "change")),
}


def p_value(file, where):
    """The first p-value on the line found by following the texts in `where` (see ROWS)."""
    text = (R / file).read_text(encoding="utf-8").splitlines()
    *sections, last = where if isinstance(where, tuple) else (where,)
    for section in sections:
        text = text[next(i for i, line in enumerate(text) if section in line):]
    line = next(line for line in text if last in line and "p=" in line)
    return float(re.search(r"p=([0-9.e-]+)", line).group(1))


def main():
    p = {row: p_value(*source) for row, source in ROWS.items()}
    order = sorted(p, key=p.get)
    m, corrected, largest = len(order), {}, 0.0
    for k, row in enumerate(order):
        largest = max(largest, min(1.0, (m - k) * p[row]))  # Holm: corrected p-values never decrease down the list
        corrected[row] = largest
    lines = [f"Holm correction over the {m} comparisons with plain QUANT in the paper's Table III",
             "(p-values as printed in the summary files, 2 significant digits)", "",
             f"{'Table III row':<32}{'Wilcoxon p':>12}{'Holm p':>10}   still below 0.05?"]
    for row in ROWS:
        lines.append(f"{row:<32}{p[row]:>12.2g}{corrected[row]:>10.2g}   "
                     f"{'yes' if corrected[row] < 0.05 else 'no'}")
    text = "\n".join(lines)
    (R / "checks" / "multiple_testing.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
