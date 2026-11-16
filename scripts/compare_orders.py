# Is the 2nd-order neighbourhood actually needed? Pairs every first-order (SPARTA_ORDER=1,
# `_o1`) result with its order-2 counterpart, cell by cell, and reports the gain of the second
# hop:  d = metric(order 2) - metric(order 1)   (positive = the 2-hop features help).
#
# Pairing key: (dataset, base tag, K, task, model, split) — the same cell trained on the same
# rows (same graph, same node set, same temporal split; the static runs are aligned to a common
# split in experiment_features.data_prep_features). A model with only one of the two orders is
# listed as unpaired, not silently dropped. GraphSAGE and the CNN-based models have no first-order
# counterpart and so show up as unpaired by design.
#
# Writes (results/experiments/):
#   order_comparison.csv            one row per paired cell: AUC-PR / AUC-ROC for both orders + d
#   order_comparison_bootstrap.csv  only with --bootstrap B: paired bootstrap of d(AUC-PR) over the
#                                   shared test rows (needs results/predictions/*.parquet)
#
# The bootstrap resamples TEST ROWS only, jointly for both orders (the same indices), so it captures
# test-set sampling noise but not seed / training variance (one run per cell), and it treats rows of
# one account across test anchors as independent. It first checks that both orders scored exactly the
# same (anchor, node) rows with the same labels and refuses to pair them otherwise.
#
# Run from the repo root (after scripts/collect_results.py has been run for both orders):
#   python scripts/compare_orders.py [--bootstrap 200] [--only AMLSim,Tide]
import os
import sys
DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import argparse
import glob
import re

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from scripts.collect_results import EXP_DIR, collect
from scripts.extra_metrics import describe
from src.methods import evaluation

SUMMARY_PATH = os.path.join(EXP_DIR, "summary.csv")
OUT_PATH = os.path.join(EXP_DIR, "order_comparison.csv")
BOOT_PATH = os.path.join(EXP_DIR, "order_comparison_bootstrap.csv")
KEYS = ["dataset", "base_tag", "K", "task", "model", "split"]
_ORDER_TAG = re.compile(r"_o\d+$")


def base_tag(run_tag):
    """'d1_echo3_o1' -> 'd1_echo3'; 'static_o1' -> 'static'; order-2 tags pass through."""
    return _ORDER_TAG.sub("", run_tag)


def _keyed(df):
    d = df.copy()
    d["base_tag"] = d["run_tag"].map(base_tag)
    d["K"] = d["K"].fillna(-1).astype(int)             # static rows carry no K / task
    d["task"] = d["task"].fillna("static")
    return d


def paired_table(summary):
    """(paired, unpaired): paired has one row per cell present in BOTH orders."""
    d = _keyed(summary)
    o2, o1 = d[d["neighbourhood_order"] == 2], d[d["neighbourhood_order"] == 1]
    pairs = o2.merge(o1, on=KEYS, how="outer", suffixes=("_o2", "_o1"), indicator=True)
    both = pairs[pairs["_merge"] == "both"].copy()
    both["dAUC_PR"] = both["AUC_PR_o2"] - both["AUC_PR_o1"]
    both["dAUC_ROC"] = both["AUC_ROC_o2"] - both["AUC_ROC_o1"]
    keep = KEYS + ["AUC_PR_o2", "AUC_PR_o1", "dAUC_PR", "AUC_ROC_o2", "AUC_ROC_o1", "dAUC_ROC",
                   "PREVALENCE_o2", "PREVALENCE_o1"]
    both = both[keep].sort_values(["dataset", "base_tag", "task", "K", "model"]).reset_index(drop=True)
    lone = pairs[pairs["_merge"] != "both"]
    unpaired = (lone.assign(only_in=lone["_merge"].map({"left_only": "order 2", "right_only": "order 1"}))
                    [["dataset", "base_tag", "K", "task", "model", "only_in"]]
                    .sort_values(["only_in", "dataset", "model", "base_tag", "task", "K"]).reset_index(drop=True))
    return both, unpaired


def print_summary(paired, unpaired):
    if paired.empty:
        print("No cell has both an order-2 and an order-1 result yet.")
    else:
        g = paired.groupby(["dataset", "model"])
        agg = pd.DataFrame({
            "cells": g.size(),
            "mean_dPR": g["dAUC_PR"].mean(),
            "median_dPR": g["dAUC_PR"].median(),
            "share_2hop_better": g["dAUC_PR"].apply(lambda s: float((s > 0).mean())),
            "mean_dROC": g["dAUC_ROC"].mean(),
        }).round(4)
        pd.set_option("display.width", 200)
        print("Gain of the 2nd-order neighbourhood, d = metric(order 2) - metric(order 1):")
        print(agg.to_string())
        print(f"\nOverall: {len(paired)} paired cells | mean dAUC-PR = {paired['dAUC_PR'].mean():+.4f} | "
              f"2-hop better in {100 * (paired['dAUC_PR'] > 0).mean():.0f}% of cells")
    if len(unpaired):
        only = unpaired.groupby(["only_in", "model"]).size()
        print("\nUnpaired (present in one order only):")
        print(only.to_string())


# ------------------------------------------------------------------------------ paired bootstrap
def _order2_name(order1_base):
    """Predictions basename of the order-2 twin of an order-1 file (the `_o1` infix removed)."""
    return order1_base.replace("_o1_K", "_K", 1) if "_o1_K" in order1_base \
        else order1_base.replace("_o1_features", "_features", 1)


def _row_keys(df):
    node = df["node"].astype(str)
    return (df["anchor"].astype(str) + "|" + node) if "anchor" in df.columns else node


def bootstrap_cells(n_boot, only=None, seed=1997):
    rng = np.random.default_rng(seed)
    rows, skipped = [], []
    for path1 in sorted(glob.glob(os.path.join(evaluation.PRED_DIR, "*.parquet"))):
        base = os.path.basename(path1)
        info = describe(base)
        if info is None or info[5] != 1:
            continue
        dataset, tag, K, task, prefix, _ = info
        if only and dataset not in only:
            continue
        path2 = os.path.join(evaluation.PRED_DIR, _order2_name(base))
        if not os.path.exists(path2):
            skipped.append((base, "no order-2 predictions file"))
            continue
        y1, s1, df1 = evaluation.load_predictions(path1)
        y2, s2, df2 = evaluation.load_predictions(path2)
        if "node" not in df1.columns or "node" not in df2.columns:
            skipped.append((base, "predictions carry no node ids (cannot verify the rows match)"))
            continue
        k1, k2 = _row_keys(df1), _row_keys(df2)
        if len(k1) != len(k2) or k1.duplicated().any() or k2.duplicated().any() or set(k1) != set(k2):
            skipped.append((base, f"different test rows ({len(k1)} vs {len(k2)}): pairing would be invalid"))
            continue
        pos = pd.Series(np.arange(len(k1)), index=k1.to_numpy()).loc[k2.to_numpy()].to_numpy()
        y1a = y1[pos]
        if not np.array_equal(y1a, y2):
            skipped.append((base, "labels differ on the shared rows"))
            continue
        for model in sorted(set(s1) & set(s2)):
            a2, a1 = s2[model], s1[model][pos]
            ap2, ap1 = average_precision_score(y2, a2), average_precision_score(y2, a1)
            diffs = np.empty(n_boot)
            for b in range(n_boot):
                ix = rng.integers(0, len(y2), len(y2))
                diffs[b] = average_precision_score(y2[ix], a2[ix]) - average_precision_score(y2[ix], a1[ix])
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            rows.append(dict(dataset=dataset, base_tag=base_tag(tag), K=-1 if K is None else K,
                             task="static" if task is None else task, model=prefix + model,
                             n_test=len(y2), n_pos=int(y2.sum()), AP_o2=ap2, AP_o1=ap1,
                             dAP=ap2 - ap1, ci_lo=lo, ci_hi=hi,
                             p_2hop_not_better=float((diffs <= 0).mean())))
    return pd.DataFrame(rows), skipped


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bootstrap", type=int, default=0, metavar="B",
                    help="paired bootstrap replicates over the shared test rows (0 = off; slow on AMLWorld)")
    ap.add_argument("--only", default="", help="comma-separated datasets to bootstrap (default: all)")
    a = ap.parse_args()

    if os.path.exists(SUMMARY_PATH):
        summary = pd.read_csv(SUMMARY_PATH)
    else:
        summary, _ = collect()
    if "neighbourhood_order" not in summary.columns:
        raise SystemExit(f"{SUMMARY_PATH} predates the first-order baseline: rerun scripts/collect_results.py")

    paired, unpaired = paired_table(summary)
    os.makedirs(EXP_DIR, exist_ok=True)
    paired.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(paired)} paired cells -> {OUT_PATH}\n")
    print_summary(paired, unpaired)

    if a.bootstrap > 0:
        only = {x for x in a.only.split(",") if x}
        boot, skipped = bootstrap_cells(a.bootstrap, only=only or None)
        boot.to_csv(BOOT_PATH, index=False)
        print(f"\nWrote {len(boot)} bootstrapped cells (B={a.bootstrap}) -> {BOOT_PATH}")
        if len(boot):
            sig = (boot["ci_lo"] > 0).mean()
            print(f"  2-hop significantly better (95% CI of dAP above 0): {100 * sig:.0f}% of cells")
        for name, why in skipped:
            print(f"  [skip] {name}: {why}")
