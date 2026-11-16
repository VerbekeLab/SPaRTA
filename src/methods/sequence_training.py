# Shared training machinery for the sequence experiments (scripts/experiment_LSTM.py and
# scripts/experiment_CNN_LSTM.py): per-batch window gather from the compact (values, idx) form,
# chunked prediction, early-stopped mini-batch training on a temporal val band, and one Optuna
# study per architecture. Moved verbatim out of experiment_LSTM.py so both entry points share a
# single implementation; the only change is that tune_arch takes the model factory as an argument.
import os

import numpy as np
import optuna
import torch
from sklearn.metrics import average_precision_score

from src.methods.losses import make_loss
from src.utils.setup import suggest_param, optimize_study

SEED = 1997


def gather_windows(values_t, idx_b):
    """([B, K, F] windows, [B, K] mask) gathered from the shared compact block.

    This is the device-side twin of sequence_data._expand_windows, done PER BATCH instead of
    once for the whole population: the dense window tensor is ~K times `values` (2.6 GB -> 18 GB
    at AMLWorld K=7, ~31 GB at K=17), so materialising it was what set this job's memory
    request. Absent steps (idx == -1) are zeroed, exactly as _expand_windows does."""
    m = idx_b >= 0
    x = values_t[idx_b.clamp(min=0)]
    x[~m] = 0.0
    return x, m


def predict(model, values_t, idx_t, batch_size=4096):
    """Return P(positive) for the samples in `idx_t` as a numpy array. Chunked over
    batch_size so the full-population test/val band can't OOM on GPU/MPS: only batch_size
    windows exist at a time, on top of the one shared `values_t` block."""
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(idx_t), batch_size):
            x, m = gather_windows(values_t, idx_t[i:i + batch_size])
            probs.append(torch.sigmoid(model(x, m)).cpu().numpy())
    return np.concatenate(probs) if probs else np.zeros(0, dtype=np.float32)


def train_model(model, values_t, train_bands, val_bands, device, loss="bce", lr=1e-3,
                batch_size=16, max_epochs=300, patience=30, alpha=0.75, gamma=2.0,
                pos_weight=None, weight_decay=1e-5, trial=None):
    """Mini-batch Adam, early-stop on the GIVEN val band by AUC-PR, restore best.

    values_t     [M, F] float32, already on `device` — the SHARED scaled feature block. One
                 allocation for the whole job; windows are gathered from it per batch.
    train_bands  (idx [n, K] int32, y [n] f32), both already on `device`; val_bands likewise.
                 `mask` is idx >= 0, so no separate mask tensor is stored or uploaded.
    Returns (best_val_aucpr, best_state). Adapts the notebook's train_eval to the
    explicit temporal val band (no random carve); shuffling seeded with 1997. weight_decay is
    tuned (defaults to the old fixed 1e-5). When `trial` is given (Optuna tuning), the
    best-so-far val AUC-PR is reported each epoch and the trial is pruned if it tracks below
    the running median of completed trials. If training diverges (non-finite val predictions),
    it stops and returns the best-so-far AP floored at 0.0 rather than raising.

    Callers upload once, so trials no longer re-copy their bands to the device each time."""
    idx_a, ya = train_bands
    idx_v, yv = val_bands
    yv_np = yv.cpu().numpy()

    # Mini-batches are sliced out of a per-epoch permutation, not drawn from a
    # DataLoader(TensorDataset(...)) — see the same change in scripts/experiment_baseline.py.
    # A DataLoader fetches one row at a time and collates, costing a few kernel launches per
    # SAMPLE; at these window counts that overhead, not the recurrent/attention math, set the
    # epoch time and left the GPU mostly idle. This changes the shuffle stream, so
    # sequence-model numbers shift within run-to-run noise.
    n_train = idx_a.shape[0]
    gen = torch.Generator(device=device).manual_seed(SEED)

    model.to(device)
    pw = pos_weight.to(device) if pos_weight is not None else None
    criterion = make_loss(loss, pos_weight=pw, alpha=alpha, gamma=gamma)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_ap, best_state, wait = -1.0, None, 0
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_train, device=device, generator=gen)
        for i in range(0, n_train, batch_size):
            b = perm[i:i + batch_size]
            xb, mb = gather_windows(values_t, idx_a[b])
            opt.zero_grad()
            criterion(model(xb, mb), ya[b]).backward()
            opt.step()
        pv = predict(model, values_t, idx_v)
        # Diverged (NaN/inf weights, e.g. high lr + pos-weighted bce): unrecoverable, so stop
        # here — the trial COMPLETES with best-so-far (floored at 0.0) instead of crashing the
        # study on average_precision_score, and best_state keeps the last pre-divergence weights.
        if not np.isfinite(pv).all():
            best_ap = max(best_ap, 0.0)
            break
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


def save_best_trial_state(path, state, params, value):
    """Atomically write the weights of a study's best trial (temp file + os.replace, so a
    wall-time kill mid-write leaves the previous cache intact, never a truncated one).

    The cache is keyed by WHAT it holds — the trial's `params` and validation `value` — not by
    a trial number: it stays valid exactly while the study's best trial has these params and
    this value, and a study deleted and re-run under the same name can't pick up weights that
    belong to a different search."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in state.items()},
                "params": dict(params), "value": float(value)}, tmp)
    os.replace(tmp, path)


def load_best_trial_state(path, params, value, tol=1e-12):
    """The cached state_dict (on CPU) if `path` holds the weights for exactly this best
    (`params`, `value`), else None. None covers every reason the cache can't be trusted: no
    file yet, an unreadable file, or weights that belong to a different best trial (a later
    trial overtook them, or the study predates the cache) — the caller then refits."""
    if not os.path.exists(path):
        return None
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        print(f"[cache] unreadable best-trial cache {path} ({type(exc).__name__}) -- ignoring it",
              flush=True)
        return None
    if ck.get("params") != dict(params) or abs(ck.get("value", float("nan")) - value) > tol:
        return None
    return ck["state_dict"]


def _beats_study_best(study, value):
    """True when `value` exceeds every COMPLETE trial already in `study` (the trial being
    evaluated is still RUNNING, so it is not in that set). Mirrors what `study.best_value`
    will be once this trial is recorded, so only genuine new bests are written to disk."""
    done = [t.value for t in study.get_trials(deepcopy=False,
                                              states=(optuna.trial.TrialState.COMPLETE,))
            if t.value is not None]
    return not done or value > max(done)


def _refit_reserve_s(study, ckpt_path):
    """Seconds the end-of-job refit will cost if no trial in THIS run overtakes the study's
    best: zero when the best trial's weights are cached (nothing to refit) or the study has
    no finished trial yet; otherwise that trial's own duration — a refit is the same
    computation. Passed to `walltime_budget` so the refit gets wall time instead of being
    killed mid-way, which is how a job that tuned for 12 h ended up writing nothing."""
    try:
        best = study.best_trial
    except ValueError:                       # no COMPLETE trial yet: the first one is cached
        return 0.0
    if load_best_trial_state(ckpt_path, best.params, best.value) is not None:
        return 0.0
    return best.duration.total_seconds() if best.duration is not None else 0.0


def tune_arch(make_model, arch, study_name, storage, n_trials, search_space, n_features,
               values_t, train_bands, val_bands, device, pos_weight, patience, pruner=None,
               ckpt_path=None):
    """One Optuna study for an architecture: maximise val-band AUC-PR via train_model.
    `make_model(arch, params, n_features)` builds the (untrained) model for a params dict.
    Returns (best_params, best_value). MedianPruner (via `pruner`) stops trials tracking below
    the median early; per-study `storage` makes it resumable and n_trials tops up on resume.

    `ckpt_path` (optional) turns on the best-trial cache. The final refit on the best params
    is the same computation as the best trial itself (same seed, same bands, same params), so
    it only repeats hours of training the study already did. With `ckpt_path`, every trial that
    sets a new study best saves its early-stopped weights there, and the caller can score the
    test band from `load_best_trial_state(ckpt_path, best_params, best_value)` instead of
    refitting. If the study's CURRENT best has no cached weights (a study from before the
    cache, or a lost file), the refit still has to happen, so its expected cost is held back
    from the tuning time budget. Without `ckpt_path` nothing changes."""
    def objective(trial):
        params = {name: suggest_param(trial, name, spec) for name, spec in search_space.items()}
        # Reseed before each trial (mirrors experiment_baseline.py's _fit_predict) so trials
        # differ only by hyperparameters, not by weight init / batch order.
        torch.manual_seed(SEED)
        model = make_model(arch, params, n_features)
        # 'epochs' is the max_epochs upper bound; early stopping (patience) / pruning usually
        # end training sooner, so it mostly caps very-long runs rather than fixing length.
        best_ap, best_state = train_model(
            model, values_t, train_bands, val_bands, device,
            loss=params["loss"], lr=params["lr"], batch_size=params["batch_size"],
            max_epochs=params["epochs"], patience=patience, alpha=params["alpha"], gamma=params["gamma"],
            weight_decay=params.get("weight_decay", 1e-5), pos_weight=pos_weight, trial=trial)
        # Keep the weights only when this trial becomes the study's best (a few MB per write,
        # a handful of writes per study). Written BEFORE returning, so a kill after the trial
        # ends can't lose them; a pruned trial raised above and never gets here.
        if ckpt_path is not None and best_state is not None and _beats_study_best(trial.study, best_ap):
            save_best_trial_state(ckpt_path, best_state, params, best_ap)
            print(f"[cache] {study_name}: trial {trial.number} is the new best "
                  f"(val AUC-PR {best_ap:.4f}) -- weights saved to {ckpt_path}", flush=True)
        return best_ap

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name=study_name, storage=storage,
                                load_if_exists=True,
                                pruner=(pruner or optuna.pruners.NopPruner()))
    # optimize_study runs only the trials still missing from the budget, under a wall-time
    # timeout so the final refit and metrics still get written. The budget is recomputed per
    # call — the second architecture's study gets whatever wall clock the first left over;
    # both top up on resubmit, and a study already at n_trials runs nothing at all.
    reserve_s = _refit_reserve_s(study, ckpt_path) if ckpt_path is not None else 0.0
    if reserve_s:
        print(f"[cache] {study_name}: the study's best trial has no cached weights -- holding "
              f"back {reserve_s / 3600:.2f} h of wall time for the end-of-job refit in case no "
              f"new trial beats it", flush=True)
    optimize_study(study, objective, n_trials, label=study_name, reserve_s=reserve_s)
    return study.best_params, study.best_value
