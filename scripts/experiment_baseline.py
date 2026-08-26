# Non-sequential baselines on the same K-snapshot windows the sequence models use
# (scripts/experiment_LSTM.py): a tuned XGBoost classifier, a tuned feed-forward MLP, and
# an unsupervised Isolation Forest anomaly detector. Kept separate from the time-series
# models so baselines and sequence experiments can evolve independently and so new
# (non-baseline, non-sequential) methods can slot in beside them. Minimalist style;
# self-contained.
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

from src.utils.setup import (load_config, resolve_dataset, resolve_timing,
                             resolve_sequence, run_tag, suggest_param,
                             resolve_storage, make_pruner, optimize_study,
                             resolve_ts_cache_dir,
                             resolve_window_store_dir, require_prebuilt_windows)
from src.data.sequence_data import build_sequence_windows, temporal_split
from src.methods import evaluation

SEED = 1997


def run_xgb_baseline(Xtr2d, ytr, Xval2d, yval, Xte2d, search_space, seed=SEED, n_trials=20,
                     study_name=None, storage=None, early_stopping_rounds=None):
    """XGBoost on already-2D feature arrays. Optuna tuning of AUC-PR on the (Xval2d, yval)
    band only (no CV shuffle). Returns (probs_te, best_params, best_value). Imbalance via
    scale_pos_weight (mirrors the bce pos_weight in the sequence models).

    early_stopping_rounds (native XGBoost, on the val band) bounds each trial's tree count, so
    the tuned n_estimators is an UPPER cap and predict_proba uses the best iteration. XGBoost
    trials are single fits with no per-epoch report, so the study runs with a NopPruner (no
    Optuna pruning); a per-study SQLite `storage` makes it resumable after a Slurm timeout."""
    from xgboost import XGBClassifier

    spw = (ytr == 0).sum() / max(1, (ytr == 1).sum())

    def _fit(**params):
        # eval_metric + early_stopping_rounds live on the constructor in xgboost >= 2.0; the
        # eval_set drives both early stopping and the val-band score used by the objective.
        kw = dict(eval_metric="aucpr", scale_pos_weight=spw, random_state=seed, **params)
        if early_stopping_rounds:
            kw["early_stopping_rounds"] = early_stopping_rounds
        clf = XGBClassifier(**kw)
        clf.fit(Xtr2d, ytr, eval_set=[(Xval2d, yval)], verbose=False)
        return clf

    def objective(trial):
        params = {name: suggest_param(trial, name, spec) for name, spec in search_space.items()}
        clf = _fit(**params)
        pv = clf.predict_proba(Xval2d)[:, 1]
        return average_precision_score(yval, pv) if yval.sum() else 0.0

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=study_name,
                                storage=storage, load_if_exists=True,
                                pruner=optuna.pruners.NopPruner())
    # Runs only the trials still missing from the budget, under a wall-time timeout so the
    # final refit and metrics still get written; the resumed study tops back up to n_trials
    # on resubmit, and one already at n_trials skips tuning entirely.
    optimize_study(study, objective, n_trials, label=study_name)
    best_params = study.best_params

    return _fit(**best_params).predict_proba(Xte2d)[:, 1], best_params, study.best_value


def run_nn_baseline(Xtr2d, ytr, Xval2d, yval, Xte2d, search_space, seed=SEED,
                    n_trials=20, patience=20, study_name=None, storage=None, pruner=None):
    """Feed-forward MLP on the already-2D feature arrays. Light Optuna tuning of AUC-PR on
    the (Xval2d, yval) band only — mirrors run_xgb_baseline. Two deliberate deviations from
    the tree baselines: (1) features are standardised (StandardScaler fit on TRAIN only),
    since an MLP — unlike tree/isolation splits — is scale-sensitive; (2) training is
    mini-batch Adam with early stopping on val-band AUC-PR (mirrors experiment_LSTM.py).
    Imbalance via BCEWithLogitsLoss pos_weight (= neg/pos, the same rule the sequence
    models use). torch is imported lazily here (after the module-level xgboost import) to
    keep the tree-only path light and preserve the xgboost-before-torch OpenMP order.
    Returns (probs_te, best_params, best_value)."""
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    from src.methods.models import NeuralNetwork

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")

    # Standardise (fit on TRAIN only); the trees did not need this, the MLP does. Tensors
    # are moved to the device once, so the training loop only ever gathers from device memory.
    scaler = StandardScaler().fit(Xtr2d)
    Xtr = torch.as_tensor(scaler.transform(Xtr2d), dtype=torch.float32).to(device)
    Xval = torch.as_tensor(scaler.transform(Xval2d), dtype=torch.float32).to(device)
    Xte = torch.as_tensor(scaler.transform(Xte2d), dtype=torch.float32).to(device)
    ytr_np = np.asarray(ytr)
    ytr_t = torch.as_tensor(ytr_np, dtype=torch.float32).to(device)
    yval_np = np.asarray(yval)

    input_size = Xtr.shape[1]
    spw = (ytr_np == 0).sum() / max(1, (ytr_np == 1).sum())
    pos_weight = torch.tensor([spw], dtype=torch.float32, device=device)

    def _fit_predict(params, X_eval, trial=None):
        """Train an MLP (mini-batch Adam, early-stop on val-band AUC-PR, restore the best
        state) and return (best_val_aucpr, P(positive) for X_eval). Seeded per call so
        trials differ only by hyperparameters, not by weight init / batch order. dropout and
        weight_decay are tuned (default to the old fixed 0.0 / 1e-5 when absent). When `trial`
        is given (tuning), the best-so-far val AUC-PR is reported each epoch and the trial is
        pruned if it falls below the running median of completed trials."""
        torch.manual_seed(seed)
        model = NeuralNetwork(num_layers=params["num_layers"], input_size=input_size,
                              hidden_size=params["hidden_size"], output_size=1,
                              dropout=params.get("dropout", 0.0)).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(model.parameters(), lr=params["lr"],
                               weight_decay=params.get("weight_decay", 1e-5))
        # Mini-batches are sliced out of a per-epoch permutation rather than drawn from a
        # DataLoader(TensorDataset(...)). A DataLoader fetches ONE ROW AT A TIME and collates
        # them, so on device-resident tensors it spends a couple of kernel launches PER SAMPLE:
        # measured ~10 us/sample, i.e. minutes of pure launch overhead per epoch at these row
        # counts (millions) against milliseconds of actual GEMM — which is why the A100 sat at
        # 3-4% utilisation. That cost is proportional to batch_size, so raising batch_size could
        # not fix it. One gather per batch makes batch assembly ~free and lets batch_size cut the
        # step count as intended. NOTE: this changes the shuffle stream (a DataLoader draws from
        # its generator in its own way), so MLP baseline numbers shift within run-to-run noise —
        # re-run a combo's MLP rather than comparing across the change.
        n_train, bs = Xtr.shape[0], params["batch_size"]
        gen = torch.Generator(device=device).manual_seed(seed)

        best_ap, best_state, wait = -1.0, None, 0
        for epoch in range(params["epochs"]):    # 'epochs' is the max_epochs cap; patience/pruning end sooner
            model.train()
            perm = torch.randperm(n_train, device=device, generator=gen)
            for i in range(0, n_train, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                criterion(model(Xtr[idx]).squeeze(-1), ytr_t[idx]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                pv = torch.sigmoid(model(Xval).squeeze(-1)).cpu().numpy()
            ap = average_precision_score(yval_np, pv) if yval_np.sum() else 0.0
            if ap > best_ap:
                best_ap, wait = ap, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
            # Report best-so-far (monotone) so the pruner compares each trial's best at matching
            # epochs; a dip in a single noisy epoch can't prune a trial that already found a good
            # config. Report before the early-stop break so the last value is always recorded.
            if trial is not None:
                trial.report(best_ap, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            if wait >= patience:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_eval).squeeze(-1)).cpu().numpy()
        return best_ap, probs

    def objective(trial):
        params = {name: suggest_param(trial, name, spec) for name, spec in search_space.items()}
        best_ap, _ = _fit_predict(params, Xval, trial=trial)
        return best_ap

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=study_name,
                                storage=storage, load_if_exists=True,
                                pruner=(pruner or optuna.pruners.NopPruner()))
    # Wall-time budget recomputed per call: when this runs after the XGBoost study in one
    # job it gets whatever wall clock that study left over (see walltime_budget's docstring).
    optimize_study(study, objective, n_trials, label=study_name)
    best_params = study.best_params

    # Refit the best config on train (early-stopped on val, as in run_xgb_baseline) and score test.
    _, probs_te = _fit_predict(best_params, Xte)
    return probs_te, best_params, study.best_value


def run_iforest_baseline(Xtr2d, Xte2d, params=None, seed=SEED):
    """Unsupervised Isolation Forest anomaly baseline on the SAME 2D feature arrays as the
    XGBoost baseline. Ignores the labels: fits on train and ranks the test set by the anomaly
    score -score_samples (higher = more anomalous = more suspected laundering). Fixed params
    (not tuned) — a label-free reference point. Returns the test anomaly scores."""
    from sklearn.ensemble import IsolationForest

    clf = IsolationForest(random_state=seed, **(params or {}))
    clf.fit(Xtr2d)
    return -clf.score_samples(Xte2d)


def _xgb_arrays(values, band_idx, K, task):
    """Per-task 2D feature array for ONE band, gathered straight from the compact
    (values, idx) form — the dense [N, K, 90] window tensor is never materialised.
    nowcast  -> anchor step (K-1) only, 90 feats.
    forecast -> ALL K window steps flattened (K*90). The forecast window [a-K..a-1]
    already excludes the anchor (builder offsets base=a-K), so every step is history;
    the baselines must see the same K-step history the sequence models do — slicing it
    shorter would silently handicap them and confound the sequential-vs-not comparison.

    `band_idx` is this band's [n, K] slice of idx. Absent steps (-1) read as 0.0, matching
    _expand_windows. Nowcast gathers ONE column, so it allocates n x 90 instead of the
    n x K x 90 it used to build and immediately throw 1/K'th of away. Forecast still needs a
    real dense 2D matrix (XGBoost's DMatrix, sklearn), but only for this band — the old flow
    built the full-population tensor first and copied each band out of it, roughly doubling
    the peak."""
    if task == "nowcast":
        col = band_idx[:, K - 1]
        out = values[np.maximum(col, 0)]
        out[col < 0] = 0.0
        return out
    win = values[np.maximum(band_idx, 0)]
    win[band_idx < 0] = 0.0
    return win.reshape(win.shape[0], -1)


if __name__ == "__main__":
    np.random.seed(SEED)

    data_config = load_config("config/data/config.yaml")
    method_config = load_config("config/methods/config.yaml")

    dataset = resolve_dataset(data_config)
    type_dataset = data_config[dataset]["type_dataset"]

    # resolve_timing/resolve_sequence apply SPARTA_* env overrides so one Slurm array task
    # picks its (timing, K, task) combo without editing the YAML; `tag` namespaces this
    # combo's features dir, .npz cache, and outputs — and must match experiment_LSTM.py so
    # baselines and sequence models read the SAME windows for the per-combo comparison.
    net_cfg = resolve_timing(data_config, dataset)
    echo = net_cfg["echo"]
    tag = run_tag(net_cfg)
    seq_cfg = resolve_sequence(data_config, dataset)
    K = seq_cfg["K"]
    task = seq_cfg["task"]
    n_test_anchors = seq_cfg["n_test_anchors"]
    n_val_anchors = seq_cfg["n_val_anchors"]

    bl_cfg = method_config[dataset]["baselines"]
    xgb_search_space = bl_cfg["xgb_search_space"]
    nn_search_space = bl_cfg["nn_search_space"]
    n_trials = bl_cfg["n_trials"]            # XGBoost's Optuna budget
    nn_n_trials = bl_cfg.get("nn_n_trials", n_trials)  # MLP's own (smaller) budget — see config
    patience = bl_cfg["patience"]            # MLP early-stop patience (XGBoost uses its own below)
    iforest_params = bl_cfg.get("iforest", {})
    # Resumable per-study SQLite storage + MedianPruner (MLP) + XGBoost native early stopping.
    xgb_esr = bl_cfg.get("xgb_early_stopping_rounds")

    # Which baselines to run this job. SPARTA_BASELINE_MODELS (comma-separated) lets a sweep
    # fan the three models out into SEPARATE Slurm tasks so their tuning runs concurrently
    # (wall-clock ~= the slowest single model, not XGBoost+MLP summed under one time budget)
    # rather than all in one run. Unset / "all" -> all three, i.e. the original single-job
    # behaviour with the original unsuffixed output names. Accepts friendly aliases.
    ALL_MODELS = ["XGBoost", "NeuralNetwork", "IsolationForest"]
    _MODEL_ALIASES = {
        "xgboost": "XGBoost", "xgb": "XGBoost",
        "neuralnetwork": "NeuralNetwork", "nn": "NeuralNetwork", "mlp": "NeuralNetwork",
        "isolationforest": "IsolationForest", "iforest": "IsolationForest", "if": "IsolationForest",
    }
    raw_sel = os.environ.get("SPARTA_BASELINE_MODELS", "all").strip().lower()
    if raw_sel in ("", "all"):
        selected = list(ALL_MODELS)
    else:
        chosen = set()
        for tok in raw_sel.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if tok not in _MODEL_ALIASES:
                raise SystemExit(
                    f"Unknown baseline model {tok!r} in SPARTA_BASELINE_MODELS; "
                    f"known aliases: {sorted(_MODEL_ALIASES)}")
            chosen.add(_MODEL_ALIASES[tok])
        # Keep canonical order regardless of how the user listed them.
        selected = [m for m in ALL_MODELS if m in chosen]

    # Per-combo stem for all outputs (tag already encodes echo + snapshot grid). When a strict
    # subset of models runs (fan-out mode), suffix the outputs by the selected models so the
    # concurrent per-model tasks never overwrite each other; collect_results.py parses both the
    # combined (all-models) dump and these suffixed per-model dumps.
    stem = f"{type_dataset}_{tag}_K{K}_{task}"
    model_suffix = "" if selected == ALL_MODELS else "_" + "-".join(selected)

    # --- Build the SAME windowed dataset the sequence models use (same tag-namespaced
    # features_dir + cache_dir + store_dir as experiment_LSTM.py). Sources in order: the
    # prebuilt per-tag window store (Stage 1.5, seconds), this combo's .npz cache, then a full
    # re-assembly from the feature files. resolve_ts_cache_dir / resolve_window_store_dir both
    # send their artefacts to $VSC_SCRATCH on the cluster.
    # The baselines need only the raw per-task 2D arrays (no scaling — trees and
    # isolation splits are scale-invariant).
    values, idx, y, anchors, nodes = build_sequence_windows(
        dataset, type_dataset, echo, K, task,
        features_dir=os.path.join(bl_cfg.get("data_directory", "results/features"), tag),
        cache_dir=resolve_ts_cache_dir(tag),
        store_dir=resolve_window_store_dir(tag),
        require_prebuilt=require_prebuilt_windows(),
        time_step=net_cfg["time_step"], time_width=net_cfg["time_width"],
        time_type=net_cfg["time_type"])

    train_idx, val_idx, test_idx = temporal_split(
        anchors, y, n_test_anchors=n_test_anchors, n_val_anchors=n_val_anchors,
        dataset_label=type_dataset)

    # Drop all-masked rows per band (forecast no-history rows; nowcast anchor is
    # always valid). Mirrors experiment_LSTM.py so both consume the same node sets.
    # mask is just idx >= 0 in the compact form, so no dense mask array is needed.
    def _band(band):
        return band[(idx[band] >= 0).any(1)]
    train_idx, val_idx, test_idx = _band(train_idx), _band(val_idx), _band(test_idx)

    ytr = y[train_idx]
    yval = y[val_idx]
    y_test = y[test_idx]

    Xtr2d = _xgb_arrays(values, idx[train_idx], K, task)
    Xval2d = _xgb_arrays(values, idx[val_idx], K, task)
    Xte2d = _xgb_arrays(values, idx[test_idx], K, task)
    n_rows_values = len(values)
    del values, idx    # the models read only the 2D band arrays above
    print(f"Baselines {selected} | task={task} K={K} echo={echo} | "
          f"train {len(train_idx)} val {len(val_idx)} test {len(test_idx)} "
          f"| 2D feature dim = {Xtr2d.shape[1]} "
          f"| gathered from {n_rows_values:,} compact rows (no dense [N, K, F] tensor)")

    os.makedirs("results/tuning", exist_ok=True)
    os.makedirs("results/experiments", exist_ok=True)

    scores = {}
    best_params_all = {}

    # --- XGBoost (supervised, tuned on the val band; native early stopping, resumable study).
    if "XGBoost" in selected:
        xgb_study = f"baseline_xgb_{stem}"
        scores["XGBoost"], xgb_best, xgb_value = run_xgb_baseline(
            Xtr2d, ytr, Xval2d, yval, Xte2d, xgb_search_space, seed=SEED, n_trials=n_trials,
            study_name=xgb_study, storage=resolve_storage(bl_cfg, study_name=xgb_study),
            early_stopping_rounds=xgb_esr)
        best_params_all["XGBoost"] = {"best_params": xgb_best, "best_value_AUCPR": xgb_value}

    # --- Feed-forward MLP (supervised, tuned on the val band; standardised inputs; pruned + resumable).
    if "NeuralNetwork" in selected:
        nn_study = f"baseline_nn_{stem}"
        scores["NeuralNetwork"], nn_best, nn_value = run_nn_baseline(
            Xtr2d, ytr, Xval2d, yval, Xte2d, nn_search_space, seed=SEED, n_trials=nn_n_trials,
            patience=patience, study_name=nn_study, storage=resolve_storage(bl_cfg, study_name=nn_study),
            pruner=make_pruner(bl_cfg))
        best_params_all["NeuralNetwork"] = {"best_params": nn_best, "best_value_AUCPR": nn_value}

    # --- Isolation Forest (unsupervised, fixed params — labels are not used).
    if "IsolationForest" in selected:
        scores["IsolationForest"] = run_iforest_baseline(Xtr2d, Xte2d, params=iforest_params, seed=SEED)
        best_params_all["IsolationForest"] = {"params": iforest_params}

    # --- Persist best params, metrics, and the ROC/PR comparison figure. model_suffix is ""
    # for an all-models run (original filenames) and "_<models>" for a per-model fan-out task.
    with open(f"results/tuning/baselines_{stem}{model_suffix}_best_params.json", "w") as f:
        json.dump({"dataset": type_dataset, "run_tag": tag, "task": task, "K": K, "echo": echo,
                   "models": best_params_all}, f, indent=2)

    metrics_path = f"results/experiments/{stem}_baselines{model_suffix}.txt"
    evaluation.write_metrics(metrics_path, scores, y_test)
    # Curve PNGs disabled: they only compare the models within this run (not across all
    # experiments) and the per-combo figures add up on VSC storage. The .txt metrics keep
    # the ROC/PR arrays, so curves can still be re-plotted offline if ever needed.
    # evaluation.plot_curves(
    #     scores, y_test,
    #     f"Baselines ({task}, {tag}) — {type_dataset} (n={len(y_test)}, {int(y_test.sum())} positive)",
    #     save_path=f"results/experiments/{stem}_baselines{model_suffix}.png")
    print(f"Wrote metrics -> {metrics_path}")

    # The per-study SQLites in results/tuning/ are deliberately KEPT — see optimize_study.
    # They are the on-disk record of which cells finished their trial budget, and they make an
    # accidental resubmission cheap (it skips tuning) instead of a full re-tune.
