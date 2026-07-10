# Aggregate every model's metrics into one tidy long-format table for the
# "effect of timing parameters" analysis. Walks results/experiments/ for metric dumps from
# experiment_LSTM.py / experiment_baseline.py / experiment_features.py / experiment_CNN.py
# and writes results/experiments/summary.csv with one row per (dataset, run_tag, K, task, model).
#
# Run from the repo root: python scripts/collect_results.py
#
# Filename conventions parsed (set by the experiment scripts):
#   dynamic:  {type}_{tag}_K{K}_{task}_{timeseries|baselines}.txt   (write_metrics blocks)
#   static:   {type}_features_{LogisticRegression|XGBoost|NN}.txt   (single-model dump)
#   CNN:      {type}_CNN.txt                                        (write_metrics block, test split)
# where tag matches <g><step>_(echo<days>|w<width>) with g the time_type initial
# (d=days, h=hours, ...), e.g. d1_echo3 / d2_w3 / h6_echo1.
import os
import sys
DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import re
import glob
import pandas as pd

EXP_DIR = "results/experiments"
OUT_PATH = os.path.join(EXP_DIR, "summary.csv")

# {type}_{tag}_K{K}_{task}_{kind}[_<models>].txt — tag has the fixed <g><step>_(echo<d>|w<w>)
# shape (g = run_tag's time_type initial: d, h, ...), so the non-greedy {type} can't swallow it.
# type may contain hyphens (HI-Small) but no '_'. The optional trailing _<models> group is the
# per-model/per-arch fan-out suffix (experiment_baseline.py with SPARTA_BASELINE_MODELS, or
# experiment_LSTM.py with SPARTA_SEQ_MODELS): per-model/per-arch tasks write e.g.
# ..._baselines_XGBoost.txt / ..._baselines_NeuralNetwork-IsolationForest.txt /
# ..._timeseries_LSTM.txt. It is purely for filename uniqueness; the actual model names come
# from the 'Model:' blocks inside.
DYNAMIC_RE = re.compile(
    r"^(?P<type>.+?)_(?P<tag>[a-z]\d+_(?:echo\d+|w\d+))_K(?P<K>\d+)_"
    r"(?P<task>nowcast|forecast)_(?P<kind>timeseries|baselines)"
    r"(?:_(?P<models>[A-Za-z][A-Za-z-]*))?\.txt$"
)
STATIC_RE = re.compile(
    r"^(?P<type>.+)_features_(?P<model>LogisticRegression|XGBoost|NN)\.txt$"
)
CNN_RE = re.compile(r"^(?P<type>.+)_CNN\.txt$")
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

        c = CNN_RE.match(fname)
        if c:
            # experiment_CNN: single-model write_metrics dump, scored on the test band.
            for model, (pr, roc) in parse_write_metrics(text).items():
                rows.append(dict(dataset=c["type"], run_tag="static", K=None, task=None,
                                 model=model, split="test", AUC_PR=pr, AUC_ROC=roc))
            continue

        # *_HPT_features_XGBoost.txt and anything else are not metric dumps — skip quietly
        # only if they clearly aren't (no AUC PR line); otherwise flag for the user.
        if "HPT" not in fname and _PR.search(text):
            unparsed.append(fname)

    df = pd.DataFrame(rows, columns=["dataset", "run_tag", "K", "task", "model",
                                     "split", "AUC_PR", "AUC_ROC"])
    # A per-model baseline fan-out and a combined all-models run of the same combo produce the
    # same (dataset, run_tag, K, task, model) metrics in two files; keep one so it isn't counted
    # twice. keep="last" is arbitrary — they are identical (same seed, same code).
    df = df.drop_duplicates(subset=["dataset", "run_tag", "K", "task", "model", "split"],
                            keep="last")
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
