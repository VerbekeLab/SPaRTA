# Extra test metrics -- F1 at the 0.5 cut-off, best-F1 and precision@k -- computed OFFLINE from the saved per-row test
# scores in results/predictions/*.parquet (written by every experiment script, or backfilled with
# SPARTA_RESCORE=1; see slurm/README.md). Nothing is retrained. Writes one tidy table with the
# same key columns as results/experiments/summary.csv, so the two merge on
# (dataset, run_tag, K, task, model, split):
#   results/experiments/extra_metrics.csv
#
# F1@0.5 is the F1 of the fixed rule score >= 0.5 (meaningful for the supervised models'
# probabilities; arbitrary for IsolationForest's anomaly scores, which are not on that scale).
# F1_best is the MAXIMUM F1 over thresholds on the test band ("F1_best", with its threshold /
# precision / recall): an oracle operating point, optimistic, and not a deployable F1 -- see
# evaluation.best_f1. precision@k defaults to k=100 only; --ks / --fracs (fractions of the test
# rows) / --r-precision (k = #positives) add more. Counts above the test size are NaN.
#
# Run from the repo root: python scripts/extra_metrics.py [--ks 100,500] [--fracs 0.01] [--r-precision]
import os
import sys
DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import argparse
import glob

import pandas as pd

from scripts.collect_results import DYNAMIC_RE, EXP_DIR
from src.methods import evaluation

OUT_PATH = os.path.join(EXP_DIR, "extra_metrics.csv")
STATIC_MODEL = {"NN": "NeuralNetwork"}      # same renaming as collect_results.py
KEYS = ["dataset", "run_tag", "K", "task", "model", "split"]


def describe(basename):
    """(dataset, run_tag, K, task, model-prefix) for a predictions file, or None. The dynamic
    names are the metrics dumps' with .parquet for .txt, so collect_results' regex applies."""
    m = DYNAMIC_RE.match(basename[:-len(".parquet")] + ".txt")
    if m:
        return m["type"], m["tag"], int(m["K"]), m["task"], "Local " if m["kind"] == "local" else ""
    for suffix in ("_features.parquet", "_CNN.parquet"):
        if basename.endswith(suffix):
            return basename[:-len(suffix)], "static", None, None, ""
    return None


def main(ks, fracs, r_precision):
    rows, skipped = [], []
    for path in sorted(glob.glob(os.path.join(evaluation.PRED_DIR, "*.parquet"))):
        base = os.path.basename(path)
        info = describe(base)
        if info is None:
            skipped.append(base)
            continue
        dataset, tag, K, task, prefix = info
        y, scores, _ = evaluation.load_predictions(path)
        for model, s in scores.items():
            rows.append({"dataset": dataset, "run_tag": tag, "K": K, "task": task,
                         "model": prefix + STATIC_MODEL.get(model, model), "split": "test",
                         "n_test": len(y), "n_pos": int(y.sum()),
                         **evaluation.extra_metrics(y, s, ks=ks, fracs=fracs,
                                                       r_precision=r_precision)})
    df = pd.DataFrame(rows)
    if len(df):
        # a per-model fan-out and a combined run of one cell give the same row twice
        df = df.drop_duplicates(subset=KEYS, keep="last")
        df = df.sort_values(["dataset", "run_tag", "task", "K", "model"], na_position="first")
    df.reset_index(drop=True).to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows -> {OUT_PATH}")
    if skipped:
        print(f"[warn] skipped (unrecognised name): {skipped}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("--ks", default="100", help="comma-separated counts for precision@k")
    ap.add_argument("--fracs", default="",
                    help="comma-separated fractions of the test rows for precision@k")
    ap.add_argument("--r-precision", action="store_true", help="also report P@R (k = number of positives)")
    a = ap.parse_args()
    main(ks=[int(x) for x in a.ks.split(",") if x], fracs=[float(x) for x in a.fracs.split(",") if x],
         r_precision=a.r_precision)
