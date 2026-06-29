# Time-series experiment: LSTM + Transformer over K-snapshot windows, with an
# XGBoost baseline on the same features. Mirrors the notebooks/test_timeseries.ipynb
# flow, ported to the temporal (anchor-band) split. Per architecture we run an
# Optuna study (maximise val-band AUC-PR), train the best config on train with
# early-stop on the val band, then score the held-out test band. Minimalist style;
# heavy work lives under __main__ so the module functions stay importable.
import os
import sys
DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import xgboost as xgb            # import xgboost BEFORE torch (OpenMP segfault otherwise)
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

import json
import optuna
from sklearn.metrics import average_precision_score

from src.utils.setup import load_config, resolve_dataset, suggest_param
from src.data.sequence_data import (build_sequence_dataset, temporal_split,
                                     fit_scaler, apply_scaler)
from src.methods.models import LSTMClassifier, TransformerClassifier
from src.methods.losses import make_loss
from src.methods import evaluation

SEED = 1997


def make_model(arch, params, n_features):
    """Build an LSTM or Transformer from a params dict (Optuna-suggested or fixed).
    lstm reads 'hidden'/'num_layers'/'dropout'; transformer reads
    'head_dim'/'nhead'/'num_layers'/'dropout' (d_model = head_dim * nhead)."""
    if arch == "lstm":
        return LSTMClassifier(n_features=n_features, hidden=params["hidden"],
                              num_layers=params["num_layers"], dropout=params["dropout"])
    if arch == "transformer":
        return TransformerClassifier(n_features=n_features, head_dim=params["head_dim"],
                                     nhead=params["nhead"], num_layers=params["num_layers"],
                                     dropout=params["dropout"])
    raise ValueError(f"unknown arch: {arch!r}")


def predict(model, X_t, mask_t, device, batch_size=4096):
    """Return P(positive) for the given (already-scaled) tensors as a numpy array.
    Chunked over batch_size so the full-population test/val band can't OOM on GPU/MPS
    (the notebook ran a ~150-node cohort in one pass; real datasets are far larger)."""
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            logits = model(X_t[i:i + batch_size].to(device), mask_t[i:i + batch_size].to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs) if probs else np.zeros(0, dtype=np.float32)


def train_model(model, train_tensors, val_tensors, device, loss="bce", lr=1e-3,
                batch_size=16, max_epochs=300, patience=30, alpha=0.75, gamma=2.0,
                pos_weight=None):
    """Mini-batch Adam, early-stop on the GIVEN val tensors by AUC-PR, restore best.

    train_tensors = (X[N,K,F] f32, mask[N,K] bool, y[N] f32); val_tensors = (Xv,mv,yv).
    Returns (best_val_aucpr, best_state). Adapts the notebook's train_eval to the
    explicit temporal val band (no random carve); shuffling seeded with 1997."""
    Xa, ma, ya = train_tensors
    Xv, mv, yv = val_tensors
    Xa, ma, ya = Xa.to(device), ma.to(device), ya.to(device)
    Xv, mv = Xv.to(device), mv.to(device)
    yv_np = yv.cpu().numpy()

    loader = DataLoader(TensorDataset(Xa, ma, ya), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))

    model.to(device)
    pw = pos_weight.to(device) if pos_weight is not None else None
    criterion = make_loss(loss, pos_weight=pw, alpha=alpha, gamma=gamma)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_ap, best_state, wait = -1.0, None, 0
    for epoch in range(max_epochs):
        model.train()
        for xb, mb, yb in loader:
            opt.zero_grad()
            criterion(model(xb, mb), yb).backward()
            opt.step()
        pv = predict(model, Xv, mv, device)
        ap = average_precision_score(yv_np, pv) if yv_np.sum() else 0.0
        if ap > best_ap:
            best_ap, wait = ap, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_ap, best_state


def run_xgb_baseline(Xtr2d, ytr, Xval2d, yval, Xte2d, search_space, seed=1997, n_trials=20):
    """XGBoost on already-2D feature arrays. Minimal Optuna tuning of AUC-PR on the
    (Xval2d, yval) band only (no CV shuffle). Returns (probs_te, best_params).
    Imbalance via scale_pos_weight (mirrors the bce pos_weight). n_trials is the
    baseline's own (deliberately small) budget, separate from the sequence-model studies."""
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


def _xgb_arrays(X, K, task):
    """Per-task 2D feature arrays from the RAW (unscaled) windows.
    nowcast  -> anchor step (K-1) only, 90 feats.
    forecast -> ALL K window steps flattened (K*90). The forecast window [a-K..a-1]
    already excludes the anchor (the builder offsets base=a-K), so every step is
    history; XGBoost must see the same K-step history the sequence models do — slicing
    it shorter would silently handicap the baseline and confound the sequential-vs-not
    comparison this task exists to make."""
    if task == "nowcast":
        return X[:, K - 1, :]
    return X.reshape(X.shape[0], -1)


def _tune_arch(arch, study_name, storage, n_trials, search_space, n_features,
               train_tensors, val_tensors, device, pos_weight):
    """One Optuna study for an architecture: maximise val-band AUC-PR via
    train_model. Returns (best_params, best_value)."""
    def objective(trial):
        params = {name: suggest_param(trial, name, spec) for name, spec in search_space.items()}
        model = make_model(arch, params, n_features)
        # 'epochs' is the max_epochs upper bound; early stopping (patience) usually
        # ends training sooner, so it mostly caps very-long runs rather than fixing length.
        best_ap, _ = train_model(
            model, train_tensors, val_tensors, device,
            loss=params["loss"], lr=params["lr"], batch_size=params["batch_size"],
            max_epochs=params["epochs"], alpha=params["alpha"], gamma=params["gamma"],
            pos_weight=pos_weight)
        return best_ap

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name=study_name, storage=storage,
                                load_if_exists=True)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on: {device}")

    torch.manual_seed(SEED)
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

    ts_cfg = method_config[dataset]["timeseries"]
    search_space = ts_cfg["search_space"]
    xgb_search_space = ts_cfg["xgb_search_space"]
    n_trials = ts_cfg["n_trials"]
    xgb_n_trials = ts_cfg.get("xgb_n_trials", 20)   # baseline's own small, separate budget
    storage = ts_cfg.get("storage")
    if storage:
        storage = storage.format(dataset=type_dataset)

    suffix = "_echo" if echo else ""

    # --- Build the windowed dataset (T derived from the dataset's snapshot grid).
    X, mask, y, anchors, nodes = build_sequence_dataset(
        dataset, type_dataset, echo, K, task,
        features_dir=ts_cfg.get("data_directory", "results/features"),
        time_step=net_cfg["time_step"], time_width=net_cfg["time_width"],
        time_type=net_cfg["time_type"])

    # --- Temporal split + mask-aware scaler (fit on TRAIN only).
    train_idx, val_idx, test_idx = temporal_split(
        anchors, y, n_test_anchors=n_test_anchors, n_val_anchors=n_val_anchors,
        dataset_label=type_dataset)
    mu, sd = fit_scaler(X, mask, nodes, anchors, K, task, train_idx)
    Xs = apply_scaler(X, mask, mu, sd)

    # Drop all-masked rows per band (forecast no-history rows; nowcast anchor is
    # always valid, so this is a no-op there).
    def _band(idx):
        return idx[mask[idx].any(1)]
    train_idx, val_idx, test_idx = _band(train_idx), _band(val_idx), _band(test_idx)

    to_t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt)
    train_tensors = (to_t(Xs[train_idx]), to_t(mask[train_idx], torch.bool), to_t(y[train_idx]))
    val_tensors = (to_t(Xs[val_idx]), to_t(mask[val_idx], torch.bool), to_t(y[val_idx]))
    test_tensors = (to_t(Xs[test_idx]), to_t(mask[test_idx], torch.bool), to_t(y[test_idx]))
    y_test = y[test_idx]

    # Empirical pos_weight (neg/pos) on the train band — used for the bce loss.
    ytr = y[train_idx]
    pos_weight = to_t([(ytr == 0).sum() / max(1, (ytr == 1).sum())])

    n_features = X.shape[-1]
    print(f"X{X.shape} | train {len(train_idx)} val {len(val_idx)} test {len(test_idx)} "
          f"| task={task} K={K} echo={echo}")

    os.makedirs("results/tuning", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/experiments", exist_ok=True)

    scores = {}
    best_params_all = {}

    # --- Sequence models: one Optuna study per architecture, then final train + test.
    # `task` is in every study/output name so nowcast and forecast runs don't collide
    # (overwriting saved models/params or resuming an incompatible SQLite study).
    archs = [("lstm", f"LSTM_{type_dataset}_{task}", "LSTM"),
             ("transformer", f"Transformer_{type_dataset}_{task}", "Transformer")]
    for arch, study_name, label in archs:
        best_params, best_value = _tune_arch(
            arch, study_name, storage, n_trials, search_space, n_features,
            train_tensors, val_tensors, device, pos_weight)
        best_params_all[label] = {"best_params": best_params, "best_value_AUCPR": best_value}

        torch.manual_seed(SEED)
        model = make_model(arch, best_params, n_features)
        train_model(model, train_tensors, val_tensors, device,
                    loss=best_params["loss"], lr=best_params["lr"],
                    batch_size=best_params["batch_size"], max_epochs=best_params["epochs"],
                    alpha=best_params["alpha"], gamma=best_params["gamma"],
                    pos_weight=pos_weight)
        scores[label] = predict(model, test_tensors[0], test_tensors[1], device)

        torch.save(
            {"state_dict": model.state_dict(), "mu": mu, "sd": sd,
             "arch": arch, "best_params": best_params, "K": K, "task": task},
            f"results/models/{study_name}{suffix}.pt")

    # --- XGBoost baseline on the RAW (unscaled) per-task 2D arrays. _band kept any
    # node with >=1 valid step across the K-window, and _xgb_arrays spans all K steps
    # (forecast) / the anchor step (nowcast), so every kept node carries real features.
    Xtr2d = _xgb_arrays(X[train_idx], K, task)
    Xval2d = _xgb_arrays(X[val_idx], K, task)
    Xte2d = _xgb_arrays(X[test_idx], K, task)
    xgb_probs, xgb_best = run_xgb_baseline(
        Xtr2d, ytr, Xval2d, y[val_idx], Xte2d, xgb_search_space, seed=SEED, n_trials=xgb_n_trials)
    scores["XGBoost"] = xgb_probs
    best_params_all["XGBoost"] = {"best_params": xgb_best}

    # --- Persist best params, metrics, and the ROC/PR comparison figure.
    with open(f"results/tuning/timeseries_{type_dataset}_{task}{suffix}_best_params.json", "w") as f:
        json.dump({"dataset": type_dataset, "task": task, "K": K, "echo": echo,
                   "models": best_params_all}, f, indent=2)

    metrics_path = f"results/experiments/{type_dataset}_timeseries_{task}{suffix}.txt"
    evaluation.write_metrics(metrics_path, scores, y_test)
    evaluation.plot_curves(
        scores, y_test,
        f"Time-series ({task}) — {type_dataset} (n={len(y_test)}, {int(y_test.sum())} positive)",
        save_path=f"results/experiments/{type_dataset}_timeseries_{task}{suffix}.png")
    print(f"Wrote metrics -> {metrics_path}")
