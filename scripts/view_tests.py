"""Round 3: three more views of the series, each added next to QUANT's own features (ideas 7, 11 and 12 in ideas.md).

  python scripts/view_tests.py main              # all three ideas on all 142 datasets (first 10 resamples)
  python scripts/view_tests.py main --ideas 7    # only one of them
  python scripts/view_tests.py summary           # -> results/<folder>/summary.txt for each idea

  idea 7   autocorrelation function and AR coefficients   results/autocorr/
  idea 11  multi-scale differences                        results/multiscale/
  idea 12  order-aware counts per window                  results/order_counts/

The views themselves are in quant_extensions.py. Same rules as improvement_tests.py and efficient_tests.py: QUANT
itself is never changed, the forest is the paper's, every run uses the official resamples and seed = resample index,
and finished datasets are skipped when the test is re-run. A dataset that fails is reported and skipped, so one error
never stops the run.

Plain QUANT is not re-run here. The summary compares each idea with plain QUANT fitted on the same computer
(analysis_utils.quant_second_machine): these runs were made on a second computer, where QUANT has the same mean
accuracy as on the first but differs on about half of the single splits.
"""

import argparse
import time
import traceback
import warnings

import numpy as np
import pandas as pd

from analysis_utils import KEY, paired, quant_second_machine
from quant_extensions import autocorr_view, multiscale_view, order_counts
from weakness_tests import NAMES, ROOT, Dataset, append, finished, quant_features
from improvement_tests import evaluate, timed

warnings.filterwarnings("ignore")

# idea number -> (results folder and file name, method name in the results, the view)
IDEAS = {
    "7": ("autocorr", "QUANT + autocorrelation view + AR (idea 7)", autocorr_view),
    "11": ("multiscale", "QUANT + multi-scale differences (idea 11)", multiscale_view),
    "12": ("order_counts", "QUANT + order-aware counts (idea 12)", order_counts),
}


def test_main(args):
    for idea in args.ideas:
        folder, method, view = IDEAS[idea]
        out = ROOT / "results" / folder / f"{folder}.csv"
        done = finished(out)
        for name in args.datasets or NAMES:
            if name in done:
                continue
            try:
                t_start = time.perf_counter()
                d = Dataset(name)
                V, t = timed(view, d.X)
                F = np.hstack([quant_features(d.X), V])
                rows = [{"dataset": name, "resample": i, **evaluate(method, F, d.y, *d.split(i), i),
                         "transform_seconds": t} for i in range(args.resamples or 10)]
                append(rows, out)
                print(f"idea {idea} {name}: done in {time.perf_counter() - t_start:.0f} s", flush=True)
            except Exception:
                print(f"idea {idea} {name}: FAILED, skipped\n{traceback.format_exc()}", flush=True)


def test_summary(args):
    quant = quant_second_machine().accuracy
    for idea in args.ideas:
        folder, method, _ = IDEAS[idea]
        path = ROOT / "results" / folder / f"{folder}.csv"
        if not path.exists():
            continue
        new = pd.read_csv(path).set_index(KEY).accuracy
        both = pd.DataFrame({"new": new, "quant": quant}).dropna()
        per = both.groupby(level="dataset").mean()
        diff, w, d, l_, p = paired(per.new, per.quant)
        change = per.new.round(4) - per.quant.round(4)
        lines = [
            f"Idea {idea}: {method}",
            f"{len(per)} datasets x {both.index.get_level_values('resample').nunique()} resamples, same splits and "
            f"seeds; plain QUANT fitted on the same computer (results/pruning/pruning.csv)",
            "",
            f"{'QUANT':<16}{100 * per.quant.mean():.2f}%",
            f"{'QUANT + idea ' + idea:<16}{100 * per.new.mean():.2f}%",
            f"{'change':<16}{100 * diff:+.2f} pts   W/D/L {w}/{d}/{l_}   Wilcoxon p={p:.2g}",
            "",
            "Largest gains (pts):",
            (100 * change.sort_values(ascending=False).head(5)).round(2).to_string(),
            "",
            "Largest losses (pts):",
            (100 * change.sort_values().head(5)).round(2).to_string(),
        ]
        text = "\n".join(lines)
        (path.parent / "summary.txt").write_text(text + "\n", encoding="utf-8")
        print(text + "\n")


TESTS = {"main": test_main, "summary": test_summary}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("tests", nargs="+", choices=list(TESTS))
    p.add_argument("--ideas", nargs="*", choices=list(IDEAS), default=list(IDEAS), help="default: all three")
    p.add_argument("--datasets", nargs="*", help="only these datasets (default: all 142)")
    p.add_argument("--resamples", type=int, help="override the number of resamples (default 10)")
    args = p.parse_args()
    for t in args.tests:
        TESTS[t](args)
