# Aggregate every model's metrics into one tidy long-format table for the
# "effect of timing parameters" analysis. Walks results/experiments/ (metric dumps from
# experiment_LSTM.py / experiment_baseline.py / experiment_features.py) and results/tuning/
# (the CNN's tuning JSON, which is HPO-only — see below) and writes
# results/experiments/summary.csv with one row per (dataset, run_tag, K, task, model).
#
# Run from the repo root: python scripts/collect_results.py
#
# Filename conventions parsed (set by the experiment scripts):
#   dynamic:  {type}_{tag}_K{K}_{task}_{timeseries|baselines}.txt   (write_metrics blocks)
#   static:   {type}_features_{LogisticRegression|XGBoost|NN}.txt   (single-model dump)
#   CNN:      results/tuning/CNN_{type}_best_params.json            (val AUC-PR only)
# where tag matches d<step>_(echo<days>|w<width>), e.g. d1_echo3 / d1_w1.
import os
import sys
DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import re
import glob
import json
import pandas as pd

EXP_DIR = "results/experiments"
TUNING_DIR = "results/tuning"
OUT_PATH = os.path.join(EXP_DIR, "summary.csv")

# {type}_{tag}_K{K}_{task}_{kind}.txt — tag has the fixed d<step>_(echo<d>|w<w>) shape, so
# the non-greedy {type} can't swallow it. type may contain hyphens (HI-Small) but no '_'.
DYNAMIC_RE = re.compile(
    r"^(?P<type>.+?)_(?P<tag>d\d+_(?:echo\d+|w\d+))_K(?P<K>\d+)_"
    r"(?P<task>nowcast|forecast)_(?P<kind>timeseries|baselines)\.txt$"
)
STATIC_RE = re.compile(
    r"^(?P<type>.+)_features_(?P<model>LogisticRegression|XGBoost|NN)\.txt$"
)
_PR = re.compile(r"AUC PR:\s*([0-9.eE+\-]+)")
_ROC = re.compile(r"AUC ROC:\s*([0-9.eE+\-]+)")


def _f(m):
    """Float from a regex match, or None."""
    return float(m.group(1)) if m else None


def parse_write_metrics(text):
    """Per-model (AUC_PR, AUC_ROC) from an evaluation.write_metrics dump — one block per
    'Model:' line. Returns {model_name: (aucpr, aucroc)}."""
    out = {}
    blocks = re.split(r"^Model:\s*", text, flags=re.MULTILINE)
    for blk in blocks:
        blk = blk.strip()
        if not blk:
            continue
        name = blk.splitlines()[0].strip()
        out[name] = (_f(_PR.search(blk)), _f(_ROC.search(blk)))
    return out


def collect():
    rows = []
    unparsed = []

    for path in sorted(glob.glob(os.path.join(EXP_DIR, "*.txt"))):
        fname = os.path.basename(path)
        # experiment_features.py writes BOTH {type}_features_XGBoost.txt (test metrics) and
        # {type}_HPT_features_XGBoost.txt (grid-search best params, no AUC PR/ROC line). Skip
        # the latter so its greedy {type} match ('{type}_HPT') doesn't fabricate a row.
        if "_HPT_" in fname:
            continue
        with open(path) as fh:
            text = fh.read()

        m = DYNAMIC_RE.match(fname)
        if m:
            # LSTM/baseline: multi-model write_metrics dump, scored on the test band.
            for model, (pr, roc) in parse_write_metrics(text).items():
                rows.append(dict(dataset=m["type"], run_tag=m["tag"], K=int(m["K"]),
                                 task=m["task"], model=model, split="test",
                                 AUC_PR=pr, AUC_ROC=roc))
            continue

        s = STATIC_RE.match(fname)
        if s:
            # experiment_features: one model per file, single AUC PR / AUC ROC, test split.
            rows.append(dict(dataset=s["type"], run_tag="static", K=None, task=None,
                             model={"NN": "NeuralNetwork"}.get(s["model"], s["model"]),
                             split="test", AUC_PR=_f(_PR.search(text)),
                             AUC_ROC=_f(_ROC.search(text))))
            continue

        # *_HPT_features_XGBoost.txt and anything else are not metric dumps — skip quietly
        # only if they clearly aren't (no AUC PR line); otherwise flag for the user.
        if "HPT" not in fname and _PR.search(text):
            unparsed.append(fname)

    # CNN: HPO-only (experiment_CNN.py reports the best VAL AUC-PR, never trains+scores
    # the test band), so its result lives in the tuning JSON, recorded as split=val.
    for path in sorted(glob.glob(os.path.join(TUNING_DIR, "CNN_*_best_params.json"))):
        try:
            d = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            unparsed.append(os.path.basename(path))
            continue
        rows.append(dict(dataset=d.get("dataset"), run_tag="static", K=None, task=None,
                         model="CNN", split="val",
                         AUC_PR=d.get("best_value_AUCPR"), AUC_ROC=None))

    df = pd.DataFrame(rows, columns=["dataset", "run_tag", "K", "task", "model",
                                     "split", "AUC_PR", "AUC_ROC"])
    df = df.sort_values(["dataset", "run_tag", "task", "K", "model"],
                        na_position="first").reset_index(drop=True)
    return df, unparsed


if __name__ == "__main__":
    os.makedirs(EXP_DIR, exist_ok=True)
    df, unparsed = collect()
    df.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(df)} rows -> {OUT_PATH}")
    if len(df):
        # Coverage at a glance: which dynamic (dataset, tag, task) combos produced results,
        # so gaps (e.g. infeasible large-K cells that raised in temporal_split) are visible.
        dyn = df[df["run_tag"] != "static"]
        if len(dyn):
            print("\nDynamic combos present (dataset | run_tag | task | K values):")
            for (ds, tag, task), g in dyn.groupby(["dataset", "run_tag", "task"]):
                ks = sorted(int(k) for k in g["K"].dropna().unique())
                print(f"  {ds:10s} | {tag:9s} | {task:8s} | K={ks}")
        print("\nTop AUC-PR per dataset:")
        for ds, g in df.dropna(subset=["AUC_PR"]).groupby("dataset"):
            best = g.loc[g["AUC_PR"].idxmax()]
            print(f"  {ds:10s} | {best['model']:16s} | {best['run_tag']}"
                  f" K={best['K']} {best['task']} | AUC-PR={best['AUC_PR']:.4f}")
    if unparsed:
        print(f"\n[warn] {len(unparsed)} result file(s) had an AUC line but didn't match a "
              f"known naming pattern (skipped): {unparsed}")
