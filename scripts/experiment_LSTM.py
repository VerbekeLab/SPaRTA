# Time-series experiment: LSTM + Transformer over K-snapshot windows. Per architecture
# we run an Optuna study (maximise val-band AUC-PR), train the best config on train with
# early-stop on the val band, then score the held-out test band. The non-sequential
# baselines (XGBoost, feed-forward MLP, IsolationForest) live in scripts/experiment_baseline.py, so this
# script focuses on sequence models only — run that script for the baselines on the
# same windowed dataset. Minimalist style; heavy work lives under __main__.
import os
import sys
DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

import json
import optuna
from sklearn.metrics import average_precision_score

from src.utils.setup import (load_config, resolve_dataset, resolve_timing,
                             resolve_sequence, run_tag, suggest_param,
                             resolve_storage, make_pruner, remaining_trials)
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
                pos_weight=None, weight_decay=1e-5, trial=None):
    """Mini-batch Adam, early-stop on the GIVEN val tensors by AUC-PR, restore best.

    train_tensors = (X[N,K,F] f32, mask[N,K] bool, y[N] f32); val_tensors = (Xv,mv,yv).
    Returns (best_val_aucpr, best_state). Adapts the notebook's train_eval to the
    explicit temporal val band (no random carve); shuffling seeded with 1997. weight_decay is
    tuned (defaults to the old fixed 1e-5). When `trial` is given (Optuna tuning), the
    best-so-far val AUC-PR is reported each epoch and the trial is pruned if it tracks below
    the running median of completed trials."""
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
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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
        # Report best-so-far (monotone) so the pruner compares each trial's best at matching
        # epochs; reported before the early-stop break so the final epoch is always recorded.
        if trial is not None:
            trial.report(best_ap, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if wait >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_ap, best_state


def _tune_arch(arch, study_name, storage, n_trials, search_space, n_features,
               train_tensors, val_tensors, device, pos_weight, pruner=None):
    """One Optuna study for an architecture: maximise val-band AUC-PR via train_model.
    Returns (best_params, best_value). MedianPruner (via `pruner`) stops trials tracking below
    the median early; per-study `storage` makes it resumable and n_trials tops up on resume."""
    def objective(trial):
        params = {name: suggest_param(trial, name, spec) for name, spec in search_space.items()}
        model = make_model(arch, params, n_features)
        # 'epochs' is the max_epochs upper bound; early stopping (patience) / pruning usually
        # end training sooner, so it mostly caps very-long runs rather than fixing length.
        best_ap, _ = train_model(
            model, train_tensors, val_tensors, device,
            loss=params["loss"], lr=params["lr"], batch_size=params["batch_size"],
            max_epochs=params["epochs"], alpha=params["alpha"], gamma=params["gamma"],
            weight_decay=params.get("weight_decay", 1e-5), pos_weight=pos_weight, trial=trial)
        return best_ap

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name=study_name, storage=storage,
                                load_if_exists=True,
                                pruner=(pruner or optuna.pruners.NopPruner()))
    study.optimize(objective, n_trials=remaining_trials(study, n_trials))
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

    # resolve_timing/resolve_sequence apply SPARTA_* env overrides so one Slurm array task
    # picks its (timing, K, task) combo without editing the YAML; `tag` namespaces this
    # combo's features dir, .npz cache, and every output file so a sweep's combos coexist.
    net_cfg = resolve_timing(data_config, dataset)
    echo = net_cfg["echo"]
    tag = run_tag(net_cfg)
    seq_cfg = resolve_sequence(data_config, dataset)
    K = seq_cfg["K"]
    task = seq_cfg["task"]
    n_test_anchors = seq_cfg["n_test_anchors"]
    n_val_anchors = seq_cfg["n_val_anchors"]

    ts_cfg = method_config[dataset]["timeseries"]
    search_space = ts_cfg["search_space"]
    n_trials = ts_cfg["n_trials"]
    # storage is templated PER STUDY (per architecture) inside the loop via resolve_storage,
    # so the LSTM and Transformer studies get separate resumable SQLite files.

    # Per-combo stem for all outputs (tag already encodes echo + snapshot grid).
    stem = f"{type_dataset}_{tag}_K{K}_{task}"

    # --- Build the windowed dataset (T derived from the dataset's snapshot grid).
    # features_dir + cache_dir are namespaced by `tag` so each timing combo reads its own
    # snapshots and writes its own .npz (the tag encodes width/days_echo — the field the
    # old cache key omitted — so a same-T combo can no longer hit a stale cache).
    X, mask, y, anchors, nodes = build_sequence_dataset(
        dataset, type_dataset, echo, K, task,
        features_dir=os.path.join(ts_cfg.get("data_directory", "results/features"), tag),
        cache_dir=os.path.join("results/timeseries", tag),
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
    archs = [("lstm", f"LSTM_{stem}", "LSTM"),
             ("transformer", f"Transformer_{stem}", "Transformer")]
    for arch, study_name, label in archs:
        best_params, best_value = _tune_arch(
            arch, study_name, resolve_storage(ts_cfg, study_name=study_name), n_trials,
            search_space, n_features, train_tensors, val_tensors, device, pos_weight,
            pruner=make_pruner(ts_cfg))
        best_params_all[label] = {"best_params": best_params, "best_value_AUCPR": best_value}

        torch.manual_seed(SEED)
        model = make_model(arch, best_params, n_features)
        train_model(model, train_tensors, val_tensors, device,
                    loss=best_params["loss"], lr=best_params["lr"],
                    batch_size=best_params["batch_size"], max_epochs=best_params["epochs"],
                    alpha=best_params["alpha"], gamma=best_params["gamma"],
                    weight_decay=best_params.get("weight_decay", 1e-5), pos_weight=pos_weight)
        scores[label] = predict(model, test_tensors[0], test_tensors[1], device)

        torch.save(
            {"state_dict": model.state_dict(), "mu": mu, "sd": sd,
             "arch": arch, "best_params": best_params, "K": K, "task": task},
            f"results/models/{study_name}.pt")

    # --- Persist best params, metrics, and the ROC/PR comparison figure (per-combo stem).
    with open(f"results/tuning/timeseries_{stem}_best_params.json", "w") as f:
        json.dump({"dataset": type_dataset, "run_tag": tag, "task": task, "K": K, "echo": echo,
                   "models": best_params_all}, f, indent=2)

    metrics_path = f"results/experiments/{stem}_timeseries.txt"
    evaluation.write_metrics(metrics_path, scores, y_test)
    evaluation.plot_curves(
        scores, y_test,
        f"Time-series ({task}, {tag}) — {type_dataset} (n={len(y_test)}, {int(y_test.sum())} positive)",
        save_path=f"results/experiments/{stem}_timeseries.png")
    print(f"Wrote metrics -> {metrics_path}")
