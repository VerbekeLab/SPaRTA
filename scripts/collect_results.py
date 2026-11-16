# Aggregate every model's metrics into one tidy long-format table for the
# "effect of timing parameters" analysis. Walks results/experiments/ for metric dumps from
# experiment_LSTM.py / experiment_CNN_LSTM.py / experiment_baseline.py / experiment_GNN.py /
# experiment_features.py / experiment_CNN.py
# and writes results/experiments/summary.csv with one row per (dataset, run_tag, K, task, model).
#
# Run from the repo root: python scripts/collect_results.py
#
# Filename conventions parsed (set by the experiment scripts):
#   dynamic:  {type}_{tag}_K{K}_{task}_{timeseries|cnnseq|baselines|local|gnn}.txt   (write_metrics
#             blocks; cnnseq = the CNN->LSTM/Transformer models of experiment_CNN_LSTM.py;
#             local = experiment_baseline.py's SPARTA_BASELINE_VIEW=local, whose models are
#             prefixed "Local " so they don't collide with the window baselines'; gnn = the
#             GraphSAGE-1L / GraphSAGE-2L models of experiment_GNN.py)
#   static:   {type}_features_{LogisticRegression|XGBoost|NN}.txt   (single-model dump)
#   CNN:      {type}_CNN.txt                                        (write_metrics block, test split)
# where tag matches <g><step>_(echo<days>|w<width>)[_o<order>] with g the time_type initial
# (d=days, h=hours, ...), e.g. d1_echo3 / d2_w3 / h6_echo1.
#
# NEIGHBOURHOOD ORDER. The first-order baseline (SPARTA_ORDER=1) carries an `_o1` infix: in the tag
# (d1_echo3_o1, before _K) and on the static type (HI-Small_o1_features_XGBoost.txt). Order-2 names
# have no infix, so every existing file parses exactly as before. The `neighbourhood_order` column
# (appended LAST, so positional readers are unaffected) is 2 or 1; `run_tag` keeps the full tag
# incl. `_o1` ("static_o1" for static) so a consumer that ignores the new column sees first-order
# rows as separate tags instead of silently mixing them into the order-2 groups; the static
# `dataset` is stripped of `_o1`, so the two orders pair on (dataset, base tag, K, task, model).
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
# The tag may end in the neighbourhood-order infix _o<order> (order 1 only; see run_tag). It sits
# BEFORE _K, so it can never be mistaken for the trailing _<models> fan-out group.
DYNAMIC_RE = re.compile(
    r"^(?P<type>.+?)_(?P<tag>[a-z]\d+_(?:echo\d+|w\d+)(?:_o(?P<order>\d+))?)_K(?P<K>\d+)_"
    r"(?P<task>nowcast|forecast)_(?P<kind>timeseries|cnnseq|baselines|local|gnn)"
    r"(?:_(?P<models>[A-Za-z][A-Za-z-]*))?\.txt$"
)
STATIC_RE = re.compile(
    r"^(?P<type>.+)_features_(?P<model>LogisticRegression|XGBoost|NN)\.txt$"
)
CNN_RE = re.compile(r"^(?P<type>.+)_CNN\.txt$")
_PR = re.compile(r"AUC PR:\s*([0-9.eE+\-]+)")
_ROC = re.compile(r"AUC ROC:\s*([0-9.eE+\-]+)")
# First entry of the dumped Precision array = precision at the lowest threshold (everything predicted
# positive) = positive rate of the test set = the AUC-PR of a no-skill classifier. numpy truncates the
# middle of the array when printing, never its first element.
_PREC0 = re.compile(r"^Precision:\s*\[\s*([0-9.eE+\-]+)", re.MULTILINE)


def _f(m):
    """Float from a regex match, or None."""
    return float(m.group(1)) if m else None


def peel_order(name):
    """Split the neighbourhood-order infix off a static type: 'HI-Small_o1' -> ('HI-Small', 1),
    'HI-Small' -> ('HI-Small', 2). (Dataset types hold hyphens but no '_', so a trailing
    _o<digits> is unambiguous.)"""
    m = re.fullmatch(r"(.+?)_o(\d+)", name)
    return (m.group(1), int(m.group(2))) if m else (name, 2)


def static_run_tag(order):
    """run_tag of a static row: 'static' for order 2, 'static_o1' for order 1."""
    return "static" if order == 2 else f"static_o{order}"


def parse_write_metrics(text):
    """Per-model (AUC_PR, AUC_ROC, prevalence) from an evaluation.write_metrics dump — one block per
    'Model:' line. Returns {model_name: (aucpr, aucroc, prevalence)}; prevalence is the test positive
    rate (the no-skill AUC-PR), None if the dump has no Precision line."""
    out = {}
    blocks = re.split(r"^Model:\s*", text, flags=re.MULTILINE)
    for blk in blocks:
        blk = blk.strip()
        if not blk:
            continue
        name = blk.splitlines()[0].strip()
        out[name] = (_f(_PR.search(blk)), _f(_ROC.search(blk)), _f(_PREC0.search(blk)))
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
            prefix = "Local " if m["kind"] == "local" else ""
            order = int(m["order"]) if m["order"] else 2
            for model, (pr, roc, prev) in parse_write_metrics(text).items():
                rows.append(dict(dataset=m["type"], run_tag=m["tag"], K=int(m["K"]),
                                 task=m["task"], model=prefix + model, split="test",
                                 AUC_PR=pr, AUC_ROC=roc, PREVALENCE=prev,
                                 neighbourhood_order=order))
            continue

        s = STATIC_RE.match(fname)
        if s:
            # experiment_features: one model per file, single AUC PR / AUC ROC, test split.
            dataset, order = peel_order(s["type"])
            rows.append(dict(dataset=dataset, run_tag=static_run_tag(order), K=None, task=None,
                             model={"NN": "NeuralNetwork"}.get(s["model"], s["model"]),
                             split="test", AUC_PR=_f(_PR.search(text)),
                             AUC_ROC=_f(_ROC.search(text)), PREVALENCE=_f(_PREC0.search(text)),
                             neighbourhood_order=order))
            continue

        c = CNN_RE.match(fname)
        if c:
            # experiment_CNN: single-model write_metrics dump, scored on the test band.
            dataset, order = peel_order(c["type"])
            for model, (pr, roc, prev) in parse_write_metrics(text).items():
                rows.append(dict(dataset=dataset, run_tag=static_run_tag(order), K=None, task=None,
                                 model=model, split="test", AUC_PR=pr, AUC_ROC=roc, PREVALENCE=prev,
                                 neighbourhood_order=order))
            continue

        # *_HPT_features_XGBoost.txt and anything else are not metric dumps — skip quietly
        # only if they clearly aren't (no AUC PR line); otherwise flag for the user.
        if "HPT" not in fname and _PR.search(text):
            unparsed.append(fname)

    df = pd.DataFrame(rows, columns=["dataset", "run_tag", "K", "task", "model",
                                     "split", "AUC_PR", "AUC_ROC", "PREVALENCE",
                                     "neighbourhood_order"])
    # A per-model baseline fan-out and a combined all-models run of the same combo produce the
    # same (dataset, run_tag, K, task, model) metrics in two files; keep one so it isn't counted
    # twice. keep="last" is arbitrary — they are identical (same seed, same code). The order is in
    # the key so a first-order cell can never displace its order-2 counterpart.
    df = df.drop_duplicates(subset=["dataset", "run_tag", "K", "task", "model", "split",
                                    "neighbourhood_order"], keep="last")
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
        dyn = df[df["task"].notna()]          # static rows (any order) have no task / K
        if len(dyn):
            print("\nDynamic combos present (dataset | run_tag | task | K values):")
            for (ds, tag, task), g in dyn.groupby(["dataset", "run_tag", "task"]):
                ks = sorted(int(k) for k in g["K"].dropna().unique())
                print(f"  {ds:10s} | {tag:12s} | {task:8s} | K={ks}")
        print("\nTop AUC-PR per dataset (per neighbourhood order):")
        for (ds, order), g in df.dropna(subset=["AUC_PR"]).groupby(["dataset", "neighbourhood_order"]):
            best = g.loc[g["AUC_PR"].idxmax()]
            print(f"  {ds:10s} | order {order} | {best['model']:16s} | {best['run_tag']}"
                  f" K={best['K']} {best['task']} | AUC-PR={best['AUC_PR']:.4f}")
    if unparsed:
        print(f"\n[warn] {len(unparsed)} result file(s) had an AUC line but didn't match a "
              f"known naming pattern (skipped): {unparsed}")
