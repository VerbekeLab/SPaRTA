# SPaRTA architecture experiment: a CNN embeds each 3x3xF snapshot frame, and an LSTM or
# Transformer reads the K-frame embedding sequence (see the "Architecture" paragraph of the
# paper). Same windows, temporal split, scaler, losses and Optuna protocol as
# scripts/experiment_LSTM.py — the flat-input LSTM/Transformer there is the "no CNN embedding"
# ablation, so the two are directly comparable. Per architecture we run an Optuna study
# (maximise val-band AUC-PR), train the best config on train with early-stop on the val band,
# then score the held-out test band. Shared training machinery lives in
# src/methods/sequence_training.py. Minimalist style; heavy work lives under __main__.
import os
import sys
DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import numpy as np
import torch

import json

from src.utils.setup import (load_config, resolve_dataset, resolve_timing,
                             resolve_sequence, run_tag,
                             resolve_storage, make_pruner,
                             resolve_ts_cache_dir,
                             resolve_window_store_dir, require_prebuilt_windows)
from src.data.sequence_data import (build_sequence_windows, temporal_split,
                                     fit_scaler_compact, apply_scaler_compact)
from src.methods.models import (LSTMClassifier, TransformerClassifier, FrameEncoder,
                                CNNSequenceClassifier)
from src.methods.sequence_training import predict, train_model, tune_arch
from src.methods import evaluation

SEED = 1997


def make_model(arch, params, n_features):
    """Build a CNN->LSTM or CNN->Transformer from a params dict (Optuna-suggested or fixed).
    Encoder reads 'cnn_channels'/'cnn_layers'/'kernel_size'/'emb_dim'; the sequence body reads
    the same keys as experiment_LSTM.make_model, with n_features = emb_dim."""
    emb_dim = params["emb_dim"]
    encoder = FrameEncoder(n_features=n_features, hidden_channels=params["cnn_channels"],
                           num_layers=params["cnn_layers"], kernel_size=params["kernel_size"],
                           emb_dim=emb_dim)
    if arch == "cnn_lstm":
        body = LSTMClassifier(n_features=emb_dim, hidden=params["hidden"],
                              num_layers=params["num_layers"], dropout=params["dropout"])
    elif arch == "cnn_transformer":
        body = TransformerClassifier(n_features=emb_dim, head_dim=params["head_dim"],
                                     nhead=params["nhead"], num_layers=params["num_layers"],
                                     dropout=params["dropout"])
    else:
        raise ValueError(f"unknown arch: {arch!r}")
    return CNNSequenceClassifier(encoder, body, emb_dim)


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
    # picks its (timing, K, task) combo without editing the YAML (see experiment_LSTM.py).
    net_cfg = resolve_timing(data_config, dataset)
    echo = net_cfg["echo"]
    tag = run_tag(net_cfg)
    seq_cfg = resolve_sequence(data_config, dataset)
    K = seq_cfg["K"]
    task = seq_cfg["task"]
    n_test_anchors = seq_cfg["n_test_anchors"]
    n_val_anchors = seq_cfg["n_val_anchors"]

    cs_cfg = method_config[dataset]["cnn_seq"]
    search_space = cs_cfg["search_space"]
    n_trials = cs_cfg["n_trials"]
    patience = cs_cfg["patience"]

    stem = f"{type_dataset}_{tag}_K{K}_{task}"

    # Which architectures this job runs. SPARTA_CNN_SEQ_MODELS (comma-separated) fans the two
    # architectures into separate Slurm tasks so their Optuna studies run concurrently.
    # Unset / "all" -> both, with the unsuffixed output names. Mirrors SPARTA_SEQ_MODELS.
    ALL_ARCHS = ["CNN-LSTM", "CNN-Transformer"]
    _ARCH_ALIASES = {"cnn-lstm": "CNN-LSTM", "cnn_lstm": "CNN-LSTM",
                     "cnn-transformer": "CNN-Transformer", "cnn_transformer": "CNN-Transformer",
                     "lstm": "CNN-LSTM", "transformer": "CNN-Transformer"}
    raw_sel = os.environ.get("SPARTA_CNN_SEQ_MODELS", "all").strip().lower()
    if raw_sel in ("", "all"):
        selected = list(ALL_ARCHS)
    else:
        chosen = set()
        for tok in raw_sel.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if tok not in _ARCH_ALIASES:
                raise SystemExit(
                    f"Unknown model {tok!r} in SPARTA_CNN_SEQ_MODELS; "
                    f"known aliases: {sorted(_ARCH_ALIASES)}")
            chosen.add(_ARCH_ALIASES[tok])
        selected = [m for m in ALL_ARCHS if m in chosen]
    model_suffix = "" if selected == ALL_ARCHS else "_" + "-".join(selected)

    # --- Windowed dataset: identical to experiment_LSTM.py (store -> .npz cache -> assembly).
    values, idx, y, anchors, nodes = build_sequence_windows(
        dataset, type_dataset, echo, K, task,
        features_dir=os.path.join(cs_cfg.get("data_directory", "results/features"), tag),
        cache_dir=resolve_ts_cache_dir(tag),
        store_dir=resolve_window_store_dir(tag),
        require_prebuilt=require_prebuilt_windows(),
        time_step=net_cfg["time_step"], time_width=net_cfg["time_width"],
        time_type=net_cfg["time_type"])

    # Temporal split + mask-aware scaler (fit on TRAIN only), on the compact form.
    train_idx, val_idx, test_idx = temporal_split(
        anchors, y, n_test_anchors=n_test_anchors, n_val_anchors=n_val_anchors,
        dataset_label=type_dataset)
    mu, sd = fit_scaler_compact(values, idx, train_idx)
    values = apply_scaler_compact(values, mu, sd)
    n_features = values.shape[1]

    def _band(band):
        return band[(idx[band] >= 0).any(1)]
    train_idx, val_idx, test_idx = _band(train_idx), _band(val_idx), _band(test_idx)

    values_t = torch.as_tensor(values, dtype=torch.float32).to(device)
    idx_shape = idx.shape
    n_values = len(values)
    del values
    to_idx = lambda b: torch.as_tensor(idx[b], dtype=torch.int32).to(device)
    to_y = lambda b: torch.as_tensor(y[b], dtype=torch.float32).to(device)
    train_bands = (to_idx(train_idx), to_y(train_idx))
    val_bands = (to_idx(val_idx), to_y(val_idx))
    test_bands = (to_idx(test_idx), to_y(test_idx))
    y_test = y[test_idx]
    del idx

    ytr = y[train_idx]
    pos_weight = torch.as_tensor([(ytr == 0).sum() / max(1, (ytr == 1).sum())],
                                 dtype=torch.float32)

    print(f"values[{n_values:,}, {n_features}] + idx{tuple(idx_shape)} on {device} "
          f"| train {len(train_idx)} val {len(val_idx)} test {len(test_idx)} "
          f"| task={task} K={K} echo={echo}")

    os.makedirs("results/tuning", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/experiments", exist_ok=True)

    scores = {}
    best_params_all = {}

    archs = [("cnn_lstm", f"CNN-LSTM_{stem}", "CNN-LSTM"),
             ("cnn_transformer", f"CNN-Transformer_{stem}", "CNN-Transformer")]
    archs = [a for a in archs if a[2] in selected]
    for arch, study_name, label in archs:
        # The Transformer body has no `hidden`, the LSTM body no head_dim/nhead: drop the
        # keys the architecture does not read so its study doesn't tune dead parameters.
        space = {k: v for k, v in search_space.items()
                 if not (arch == "cnn_lstm" and k in ("head_dim", "nhead"))
                 and not (arch == "cnn_transformer" and k == "hidden")}
        best_params, best_value = tune_arch(
            make_model, arch, study_name, resolve_storage(cs_cfg, study_name=study_name),
            n_trials, space, n_features, values_t, train_bands, val_bands, device, pos_weight,
            patience, pruner=make_pruner(cs_cfg))
        best_params_all[label] = {"best_params": best_params, "best_value_AUCPR": best_value}

        torch.manual_seed(SEED)
        model = make_model(arch, best_params, n_features)
        train_model(model, values_t, train_bands, val_bands, device,
                    loss=best_params["loss"], lr=best_params["lr"],
                    batch_size=best_params["batch_size"], max_epochs=best_params["epochs"],
                    patience=patience, alpha=best_params["alpha"], gamma=best_params["gamma"],
                    weight_decay=best_params.get("weight_decay", 1e-5), pos_weight=pos_weight)
        scores[label] = predict(model, values_t, test_bands[0])

        torch.save(
            {"state_dict": model.state_dict(), "mu": mu, "sd": sd,
             "arch": arch, "best_params": best_params, "K": K, "task": task},
            f"results/models/{study_name}.pt")

    with open(f"results/tuning/cnnseq_{stem}{model_suffix}_best_params.json", "w") as f:
        json.dump({"dataset": type_dataset, "run_tag": tag, "task": task, "K": K, "echo": echo,
                   "models": best_params_all}, f, indent=2)

    metrics_path = f"results/experiments/{stem}_cnnseq{model_suffix}.txt"
    evaluation.write_metrics(metrics_path, scores, y_test)
    print(f"Wrote metrics -> {metrics_path}")
