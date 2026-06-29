# Non-sequential baselines on the same K-snapshot windows the sequence models use
# (scripts/experiment_LSTM.py): a tuned XGBoost classifier and an unsupervised
# Isolation Forest anomaly detector. Kept separate from the time-series models so
# baselines and sequence experiments can evolve independently and so new (non-baseline,
# non-sequential) methods can slot in beside them. Minimalist style; self-contained.
import os
import sys
DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import xgboost as xgb            # imported (unused here but kept for parity / OpenMP order)
import numpy as np

import json
import optuna
from sklearn.metrics import average_precision_score

from src.utils.setup import load_config, resolve_dataset, suggest_param
from src.data.sequence_data import build_sequence_dataset, temporal_split
from src.methods import evaluation

SEED = 1997


def run_xgb_baseline(Xtr2d, ytr, Xval2d, yval, Xte2d, search_space, seed=SEED, n_trials=20):
    """XGBoost on already-2D feature arrays. Minimal Optuna tuning of AUC-PR on the
    (Xval2d, yval) band only (no CV shuffle). Returns (probs_te, best_params).
    Imbalance via scale_pos_weight (mirrors the bce pos_weight in the sequence models)."""
    from xgboost import XGBClassifier

    spw = (ytr == 0).sum() / max(1, (ytr == 1).sum())

    def objective(trial):
        params = {name: suggest_param(trial, name, spec) for name, spec in search_space.items()}
        clf = XGBClassifier(eval_metric="aucpr", scale_pos_weight=spw,
                            random_state=seed, **params)
        clf.fit(Xtr2d, ytr)
        pv = clf.predict_proba(Xval2d)[:, 1]
        return average_precision_score(yval, pv) if yval.sum() else 0.0

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params

    clf = XGBClassifier(eval_metric="aucpr", scale_pos_weight=spw,
                        random_state=seed, **best_params)
    clf.fit(Xtr2d, ytr)
    return clf.predict_proba(Xte2d)[:, 1], best_params


def run_iforest_baseline(Xtr2d, Xte2d, params=None, seed=SEED):
    """Unsupervised Isolation Forest anomaly baseline on the SAME 2D feature arrays as the
    XGBoost baseline. Ignores the labels: fits on train and ranks the test set by the anomaly
    score -score_samples (higher = more anomalous = more suspected laundering). Fixed params
    (not tuned) — a label-free reference point. Returns the test anomaly scores."""
    from sklearn.ensemble import IsolationForest

    clf = IsolationForest(random_state=seed, **(params or {}))
    clf.fit(Xtr2d)
    return -clf.score_samples(Xte2d)


def _xgb_arrays(X, K, task):
    """Per-task 2D feature arrays from the RAW (unscaled) windows.
    nowcast  -> anchor step (K-1) only, 90 feats.
    forecast -> ALL K window steps flattened (K*90). The forecast window [a-K..a-1]
    already excludes the anchor (builder offsets base=a-K), so every step is history;
    the baselines must see the same K-step history the sequence models do — slicing it
    shorter would silently handicap them and confound the sequential-vs-not comparison."""
    if task == "nowcast":
        return X[:, K - 1, :]
    return X.reshape(X.shape[0], -1)


if __name__ == "__main__":
    np.random.seed(SEED)

    data_config = load_config("config/data/config.yaml")
    method_config = load_config("config/methods/config.yaml")

    dataset = resolve_dataset(data_config)
    type_dataset = data_config[dataset]["type_dataset"]

    net_cfg = data_config[dataset]["network_construction"]
    echo = net_cfg["echo"]
    seq_cfg = data_config[dataset]["sequence"]
    K = seq_cfg["K"]
    task = seq_cfg["task"]
    n_test_anchors = seq_cfg["n_test_anchors"]
    n_val_anchors = seq_cfg["n_val_anchors"]

    bl_cfg = method_config[dataset]["baselines"]
    xgb_search_space = bl_cfg["xgb_search_space"]
    n_trials = bl_cfg["n_trials"]            # XGBoost Optuna budget
    iforest_params = bl_cfg.get("iforest", {})

    suffix = "_echo" if echo else ""

    # --- Build the SAME windowed dataset the sequence models use. The baselines need
    # only the raw per-task 2D arrays (no scaling — trees and isolation splits are
    # scale-invariant per feature).
    X, mask, y, anchors, nodes = build_sequence_dataset(
        dataset, type_dataset, echo, K, task,
        features_dir=bl_cfg.get("data_directory", "results/features"),
        time_step=net_cfg["time_step"], time_width=net_cfg["time_width"],
        time_type=net_cfg["time_type"])

    train_idx, val_idx, test_idx = temporal_split(
        anchors, y, n_test_anchors=n_test_anchors, n_val_anchors=n_val_anchors,
        dataset_label=type_dataset)

    # Drop all-masked rows per band (forecast no-history rows; nowcast anchor is
    # always valid). Mirrors experiment_LSTM.py so both consume the same node sets.
    def _band(idx):
        return idx[mask[idx].any(1)]
    train_idx, val_idx, test_idx = _band(train_idx), _band(val_idx), _band(test_idx)

    ytr = y[train_idx]
    yval = y[val_idx]
    y_test = y[test_idx]

    Xtr2d = _xgb_arrays(X[train_idx], K, task)
    Xval2d = _xgb_arrays(X[val_idx], K, task)
    Xte2d = _xgb_arrays(X[test_idx], K, task)
    print(f"Baselines | task={task} K={K} echo={echo} | "
          f"train {len(train_idx)} val {len(val_idx)} test {len(test_idx)} "
          f"| 2D feature dim = {Xtr2d.shape[1]}")

    os.makedirs("results/tuning", exist_ok=True)
    os.makedirs("results/experiments", exist_ok=True)

    scores = {}
    best_params_all = {}

    # --- XGBoost (supervised, tuned on the val band).
    scores["XGBoost"], xgb_best = run_xgb_baseline(
        Xtr2d, ytr, Xval2d, yval, Xte2d, xgb_search_space, seed=SEED, n_trials=n_trials)
    best_params_all["XGBoost"] = {"best_params": xgb_best}

    # --- Isolation Forest (unsupervised, fixed params — labels are not used).
    scores["IsolationForest"] = run_iforest_baseline(Xtr2d, Xte2d, params=iforest_params, seed=SEED)
    best_params_all["IsolationForest"] = {"params": iforest_params}

    # --- Persist best params, metrics, and the ROC/PR comparison figure.
    with open(f"results/tuning/baselines_{type_dataset}_{task}{suffix}_best_params.json", "w") as f:
        json.dump({"dataset": type_dataset, "task": task, "K": K, "echo": echo,
                   "models": best_params_all}, f, indent=2)

    metrics_path = f"results/experiments/{type_dataset}_baselines_{task}{suffix}.txt"
    evaluation.write_metrics(metrics_path, scores, y_test)
    evaluation.plot_curves(
        scores, y_test,
        f"Baselines ({task}) — {type_dataset} (n={len(y_test)}, {int(y_test.sum())} positive)",
        save_path=f"results/experiments/{type_dataset}_baselines_{task}{suffix}.png")
    print(f"Wrote metrics -> {metrics_path}")
