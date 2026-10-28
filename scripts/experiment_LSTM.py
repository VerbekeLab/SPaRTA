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

import json

from src.utils.setup import (load_config, resolve_dataset, resolve_timing,
                             resolve_sequence, run_tag,
                             resolve_storage, make_pruner,
                             resolve_ts_cache_dir,
                             resolve_window_store_dir, require_prebuilt_windows)
from src.data.sequence_data import (build_sequence_windows, temporal_split,
                                     fit_scaler_compact, apply_scaler_compact)
from src.methods.models import LSTMClassifier, TransformerClassifier
from src.methods.sequence_training import (gather_windows, predict, train_model,
                                           tune_arch)
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
    patience = ts_cfg["patience"]
    # storage is templated PER STUDY (per architecture) inside the loop via resolve_storage,
    # so the LSTM and Transformer studies get separate resumable SQLite files.

    # Per-combo stem for all outputs (tag already encodes echo + snapshot grid).
    stem = f"{type_dataset}_{tag}_K{K}_{task}"

    # Which architectures this job runs. SPARTA_SEQ_MODELS (comma-separated) lets a sweep fan
    # LSTM and Transformer out into SEPARATE Slurm tasks so their Optuna studies run
    # CONCURRENTLY (wall-clock ~= the slower architecture, not LSTM+Transformer summed under
    # one time budget) rather than sequentially in one job. Unset / "all" -> both, i.e. the
    # original single-job behaviour with the original unsuffixed output names. Mirrors
    # experiment_baseline.py's SPARTA_BASELINE_MODELS.
    ALL_ARCHS = ["LSTM", "Transformer"]
    _ARCH_ALIASES = {"lstm": "LSTM", "transformer": "Transformer"}
    raw_sel = os.environ.get("SPARTA_SEQ_MODELS", "all").strip().lower()
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
                    f"Unknown seq model {tok!r} in SPARTA_SEQ_MODELS; "
                    f"known aliases: {sorted(_ARCH_ALIASES)}")
            chosen.add(_ARCH_ALIASES[tok])
        selected = [m for m in ALL_ARCHS if m in chosen]

    # When a strict subset of architectures runs (fan-out mode), suffix the outputs by the
    # selected archs so concurrent per-arch tasks never overwrite each other; collect_results.py's
    # DYNAMIC_RE already parses both the combined and per-arch names (same as the baselines).
    model_suffix = "" if selected == ALL_ARCHS else "_" + "-".join(selected)

    # --- Build the windowed dataset (T derived from the dataset's snapshot grid).
    # features_dir + cache_dir are namespaced by `tag` so each timing combo reads its own
    # snapshots and writes its own .npz (the tag encodes width/days_echo — the field the
    # old cache key omitted — so a same-T combo can no longer hit a stale cache).
    # resolve_ts_cache_dir sends the cache to $VSC_SCRATCH on the cluster, or disables
    # it under SPARTA_TS_CACHE=off. store_dir is the Stage-1.5 prebuilt window store
    # (slurm/build_windows.slurm), tried before both of those — it is why this GPU job can
    # start training seconds after allocation instead of re-assembling windows for ~40 min.
    values, idx, y, anchors, nodes = build_sequence_windows(
        dataset, type_dataset, echo, K, task,
        features_dir=os.path.join(ts_cfg.get("data_directory", "results/features"), tag),
        cache_dir=resolve_ts_cache_dir(tag),
        store_dir=resolve_window_store_dir(tag),
        require_prebuilt=require_prebuilt_windows(),
        time_step=net_cfg["time_step"], time_width=net_cfg["time_width"],
        time_type=net_cfg["time_type"])

    # --- Temporal split + mask-aware scaler (fit on TRAIN only). Both work on the COMPACT
    # (values, idx) form: the dense [N, K, F] window tensor is never built, on host or device.
    # It is ~K times `values`, and the old flow held the raw copy, the scaled copy AND the
    # per-band tensor copies at once (~3x one array) — that product, not the model, is what
    # set this job's --mem and its GPU residency.
    train_idx, val_idx, test_idx = temporal_split(
        anchors, y, n_test_anchors=n_test_anchors, n_val_anchors=n_val_anchors,
        dataset_label=type_dataset)
    mu, sd = fit_scaler_compact(values, idx, train_idx)
    values = apply_scaler_compact(values, mu, sd)
    n_features = values.shape[1]

    # Drop all-masked rows per band (forecast no-history rows; nowcast anchor is
    # always valid, so this is a no-op there). mask is just idx >= 0 here.
    def _band(band):
        return band[(idx[band] >= 0).any(1)]
    train_idx, val_idx, test_idx = _band(train_idx), _band(val_idx), _band(test_idx)

    # ONE shared feature block on the device, plus a small int32 index per band (torch indexes
    # fine with int32, so the index stays half the size of an int64 one). Windows are gathered
    # from this per batch by gather_windows.
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

    # Empirical pos_weight (neg/pos) on the train band — used for the bce loss.
    ytr = y[train_idx]
    pos_weight = torch.as_tensor([(ytr == 0).sum() / max(1, (ytr == 1).sum())],
                                 dtype=torch.float32)

    print(f"values[{n_values:,}, {n_features}] + idx{tuple(idx_shape)} on {device} "
          f"(dense windows would be {idx_shape[0] * idx_shape[1] * n_features * 4 / 1e9:.1f} GB) "
          f"| train {len(train_idx)} val {len(val_idx)} test {len(test_idx)} "
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
    archs = [a for a in archs if a[2] in selected]
    for arch, study_name, label in archs:
        best_params, best_value = tune_arch(
            make_model, arch, study_name, resolve_storage(ts_cfg, study_name=study_name), n_trials,
            search_space, n_features, values_t, train_bands, val_bands, device, pos_weight,
            patience, pruner=make_pruner(ts_cfg))
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

    # --- Persist best params, metrics, and the ROC/PR comparison figure (per-combo stem).
    # model_suffix is "" for an all-archs run (original filenames) and "_<archs>" for a
    # per-arch fan-out task.
    with open(f"results/tuning/timeseries_{stem}{model_suffix}_best_params.json", "w") as f:
        json.dump({"dataset": type_dataset, "run_tag": tag, "task": task, "K": K, "echo": echo,
                   "models": best_params_all}, f, indent=2)

    metrics_path = f"results/experiments/{stem}_timeseries{model_suffix}.txt"
    evaluation.write_metrics(metrics_path, scores, y_test)
    # Curve PNGs disabled: they only compare the models within this run (not across all
    # experiments) and the per-combo figures add up on VSC storage. The .txt metrics keep
    # the ROC/PR arrays, so curves can still be re-plotted offline if ever needed.
    # evaluation.plot_curves(
    #     scores, y_test,
    #     f"Time-series ({task}, {tag}) — {type_dataset} (n={len(y_test)}, {int(y_test.sum())} positive)",
    #     save_path=f"results/experiments/{stem}_timeseries{model_suffix}.png")
    print(f"Wrote metrics -> {metrics_path}")

    # The per-study SQLites in results/tuning/ are deliberately KEPT — see optimize_study.
    # They are the on-disk record of which cells finished their n_trials budget, and they make
    # an accidental resubmission cheap (it skips tuning) instead of a full re-tune.
