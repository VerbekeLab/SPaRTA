# CNN over the static 3x3 measure "pictures" -- one snapshot per node, no time windows
# (see experiment_LSTM.py for the K-snapshot sequence models). Optuna study maximises
# val-band AUC-PR; the best config is retrained on train (best-epoch weights restored)
# and scored on the held-out test band.
import os
import sys
DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import json
import optuna
from sklearn.metrics import average_precision_score

from src.data.feature_data import ImageDataset
from src.methods.models import CNN
from src.methods import evaluation
from src.utils.setup import (load_config, resolve_dataset, suggest_param,
                             resolve_storage, make_pruner, remaining_trials)
from src.utils.feature_io import load_table

SEED = 1997


def load_data(features_stem, labels_stem):
    # Stems without extension: load_table prefers .parquet, falls back to legacy .csv.
    features_df = load_table(features_stem)
    labels_df = load_table(labels_stem)
    data = pd.merge(features_df, labels_df, left_on='node', right_on='Account')
    data.drop(columns=['node', 'Account'], inplace=True)
    return data


def fit_channel_transform(X, n_channels):
    """Per-channel mean/std Normalize, fit on X (train only)."""
    X_tensor = torch.tensor(X, dtype=torch.float32).view(-1, n_channels, 3, 3)
    return transforms.Normalize(X_tensor.mean([0, 2, 3]), X_tensor.std([0, 2, 3]))


def predict(model, loader, device):
    """P(positive) over a loader, as a numpy array."""
    model.eval()
    probs = []
    with torch.no_grad():
        for images, _ in loader:
            probs.extend(torch.sigmoid(model(images.to(device))).cpu().numpy().tolist())
    return np.array(probs)


def train_cnn(params, num_channels, loader_train, loader_val, y_train, y_val, device, trial=None):
    """Train a CNN, tracking the best val-AUC-PR epoch and restoring those weights before
    returning. num_epochs is itself tuned (no separate early-stop cap). Drives both the
    Optuna objective (trial given) and the final fit (trial=None). Returns (model, best_val_aucpr)."""
    model = CNN(
        num_channels=num_channels,
        hidden_channels=params['hidden_channels'],
        num_layers=params['num_layers'],
        kernel_size=params['kernel_size'],
        max_pool=params['max_pool'],
        half_final_layer=params['half_final_layer'],
    ).to(device)

    if params['scale_labels'] > 0:
        train_weight = round((y_train == 0).sum() / (y_train == 1).sum()) * params['scale_labels'] / 100
        pos_weight = torch.tensor(train_weight, device=device)
    else:
        pos_weight = None

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'],
                                 weight_decay=params.get('weight_decay', 1e-5))

    best_ap, best_state = -1.0, None
    for epoch in range(params['num_epochs']):
        model.train()
        for images, labels in loader_train:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels.float())
            loss.backward()
            optimizer.step()

        probs = predict(model, loader_val, device)
        ap = average_precision_score(y_val, probs) if y_val.sum() else 0.0
        if ap > best_ap:
            best_ap, best_state = ap, {k: v.detach().clone() for k, v in model.state_dict().items()}

        # Report best-so-far (monotone) so a trial tracking below the running median of
        # completed trials gets pruned.
        if trial is not None:
            trial.report(best_ap, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ap


def _tune_cnn(study_name, storage, n_trials, search_space, num_channels,
             loader_train, loader_val, y_train, y_val, device, pruner):
    """One Optuna study maximising val-band AUC-PR via train_cnn. Returns (best_params, best_value)."""
    def objective(trial):
        params = {name: suggest_param(trial, name, spec) for name, spec in search_space.items()}
        _, best_ap = train_cnn(params, num_channels, loader_train, loader_val, y_train, y_val, device, trial=trial)
        return best_ap

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
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
    type_dataset = data_config[dataset]['type_dataset']
    num_channels = data_config[dataset]['n_channels']

    cnn_cfg = method_config[dataset]['features_CNN']
    batch_size = cnn_cfg['batch_size']
    search_space = cnn_cfg['search_space']
    data_directory = cnn_cfg['data_directory']

    data = load_data(os.path.join(data_directory, f"{type_dataset}_static_features"),
                     os.path.join(data_directory, f"{type_dataset}_static_labels"))
    X = data.drop(columns=['is_laundering'])
    y = data['is_laundering']

    # No time axis on the static network (one snapshot per node), so a stratified random
    # split stands in for the sequence models' temporal split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=SEED)
    y_train_np, y_val_np, y_test_np = y_train.to_numpy(), y_val.to_numpy(), y_test.to_numpy()

    transform = fit_channel_transform(X_train.values, num_channels)
    dataset_train = ImageDataset(X_train.values, y_train, n_channels=num_channels, transform=transform)
    dataset_val = ImageDataset(X_val.values, y_val, n_channels=num_channels, transform=transform)
    dataset_test = ImageDataset(X_test.values, y_test, n_channels=num_channels, transform=transform)

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    print(f"CNN | train {len(dataset_train)} val {len(dataset_val)} test {len(dataset_test)} | channels={num_channels}")

    os.makedirs("results/tuning", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/experiments", exist_ok=True)

    # Per-study SQLite (resolve_storage templates the URL on study_name) so the study
    # resumes after a Slurm timeout; remaining_trials tops it up to n_trials on resume.
    study_name = f"CNN_{type_dataset}"
    best_params, best_value = _tune_cnn(
        study_name, resolve_storage(cnn_cfg, study_name=study_name), cnn_cfg['n_trials'],
        search_space, num_channels, loader_train, loader_val, y_train_np, y_val_np, device,
        make_pruner(cnn_cfg))

    # Final fit on train (same train_cnn as the objective, best-epoch restored), scored on test.
    torch.manual_seed(SEED)
    model, _ = train_cnn(best_params, num_channels, loader_train, loader_val, y_train_np, y_val_np, device)
    probs_test = predict(model, loader_test, device)

    with open(f"results/tuning/{study_name}_best_params.json", "w") as f:
        json.dump({"dataset": type_dataset, "best_params": best_params,
                   "best_value_AUCPR": best_value}, f, indent=2)

    torch.save({"state_dict": model.state_dict(), "mean": transform.mean, "std": transform.std,
               "num_channels": num_channels, "best_params": best_params},
              f"results/models/{study_name}.pt")

    scores = {"CNN": probs_test}
    metrics_path = f"results/experiments/{type_dataset}_CNN.txt"
    evaluation.write_metrics(metrics_path, scores, y_test_np)
    evaluation.plot_curves(
        scores, y_test_np,
        f"CNN — {type_dataset} (n={len(y_test_np)}, {int(y_test_np.sum())} positive)",
        save_path=f"results/experiments/{type_dataset}_CNN.png")
    print(f"Wrote metrics -> {metrics_path}")
