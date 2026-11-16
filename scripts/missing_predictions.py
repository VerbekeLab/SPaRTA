# Which finished cells still lack a saved test-prediction table? Compares the metrics dumps in
# results/experiments/*.txt with results/predictions/*.parquet (same basename: the experiment
# scripts write both from one stem, see evaluation.save_predictions) and prints, per model
# family, how many are covered and which are missing. Backfill a missing cell by re-running its
# job with SPARTA_RESCORE=1 (reload the .pt / refit from the stored best params; no tuning):
#   SPARTA_RESCORE=1 sbatch --time=01:00:00 slurm/train_lstm.slurm AMLWorld lstm
# A per-model fan-out dump (..._baselines_XGBoost-IsolationForest.txt) must be rescored with the
# same SPARTA_BASELINE_MODELS / SPARTA_SEQ_MODELS / SPARTA_CNN_SEQ_MODELS selection, because the
# predictions file takes the same suffix.
#
# Run from the repo root: python scripts/missing_predictions.py [--list]
import os
import sys
DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import glob
import re
from collections import defaultdict

EXP_DIR, PRED_DIR = "results/experiments", "results/predictions"
FAMILIES = [  # (label, regex on the metrics basename)
    ("LSTM/Transformer", r"_(?:nowcast|forecast)_timeseries"),
    ("CNN-LSTM/CNN-Transformer", r"_(?:nowcast|forecast)_cnnseq"),
    ("window baselines", r"_(?:nowcast|forecast)_baselines"),
    ("local baselines", r"_(?:nowcast|forecast)_local"),
    ("GraphSAGE", r"_(?:nowcast|forecast)_gnn"),
    ("static features", r"_features_(?:LogisticRegression|XGBoost|NN)\.txt$"),
    ("static CNN", r"_CNN\.txt$"),
]


def pred_name(txt):
    """Predictions basename for a metrics dump (static dumps share one table per dataset)."""
    base = os.path.basename(txt)
    m = re.match(r"^(.+_features)_(?:LogisticRegression|XGBoost|NN)\.txt$", base)
    return (m.group(1) if m else base[:-len(".txt")]) + ".parquet"


if __name__ == "__main__":
    show = "--list" in sys.argv
    have = {os.path.basename(p) for p in glob.glob(f"{PRED_DIR}/*.parquet")}
    missing = defaultdict(set)
    total = defaultdict(set)
    for txt in sorted(glob.glob(f"{EXP_DIR}/*.txt")):
        base = os.path.basename(txt)
        if "_HPT_" in base:
            continue
        for label, pat in FAMILIES:
            if re.search(pat, base):
                total[label].add(pred_name(txt))
                if pred_name(txt) not in have:
                    missing[label].add(pred_name(txt))
                break
    for label, _ in FAMILIES:
        n, miss = len(total[label]), missing[label]
        print(f"{label:26s} {n - len(miss):4d}/{n:<4d} have predictions")
        if show:
            for name in sorted(miss):
                print(f"    missing: {name}")
    if not show and any(missing.values()):
        print("\n(--list shows the missing cells)")
