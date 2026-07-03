import os
import sys
DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import torch 
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.feature_data import ImageDataset
from src.methods.models import CNN
from src.utils.setup import load_config, resolve_dataset, suggest_param
from src.utils.feature_io import load_table

from sklearn.metrics import roc_auc_score, average_precision_score

import json
import optuna

def load_data(features_stem, labels_stem):
    # Stems without extension: load_table prefers .parquet, falls back to legacy .csv.
    features_df = load_table(features_stem)
    labels_df = load_table(labels_stem)
    data = pd.merge(features_df, labels_df, left_on='node', right_on='Account')
    data.drop(columns=['node', 'Account'], inplace=True)
    return data

def transforms_channels(X, n_channels):
    X_tensor = torch.tensor(X, dtype=torch.float32).view(-1, n_channels, 3, 3)
    X_mean = torch.mean(X_tensor, dim = [0,2,3])
    X_std = torch.std(X_tensor, dim = [0,2,3])
    return transforms.Normalize(X_mean, X_std)

def objective(trial):
    # Search space is driven by config (features_CNN.search_space); suggest_param
    # maps each entry to the right optuna suggest_* call. NUM_CHANNELS and
    # SEARCH_SPACE are set as module globals in __main__.
    params = {name: suggest_param(trial, name, spec) for name, spec in SEARCH_SPACE.items()}

    num_epochs = params['num_epochs']
    scale_labels = params['scale_labels']
    learning_rate = params['learning_rate']

    model = CNN(
        num_channels=NUM_CHANNELS,
        hidden_channels=params['hidden_channels'],
        num_layers=params['num_layers'],
        kernel_size=params['kernel_size'],
        max_pool=params['max_pool'],
        half_final_layer=params['half_final_layer']
    )
    model.to(device)

    y_train = dataset_train.y.numpy()
    if scale_labels > 0:
        train_weight = round((y_train == 0).sum() / (y_train == 1).sum())*scale_labels/100; print(f"Positive class weight: {train_weight}")
        pos_weight=torch.tensor(train_weight)
    else:
        pos_weight = None; print("No positive class weight applied")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    for epoch in range(num_epochs):
        model.train()
        for images, labels in loader_train:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for images, labels in loader_val:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            y_pred.extend(torch.sigmoid(outputs).cpu().numpy().tolist())
            y_true.extend(labels.float().cpu().numpy().tolist())
    roc_loss = roc_auc_score(y_true, y_pred)
    ap_loss = average_precision_score(y_true, y_pred)
    print(f'Validation errors, AUC-ROC: {roc_loss:.4f}, AUC-PR: {ap_loss:.4f}')
    return average_precision_score(y_true, y_pred)


if __name__ == "__main__":
    ### To Do ###
    # Transform input image data

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"); print(f"Training on: {device}")

    data_config = load_config("config/data/config.yaml")
    method_config = load_config("config/methods/config.yaml")

    dataset = resolve_dataset(data_config)
    type_dataset = data_config[dataset]['type_dataset']

    num_channels = data_config[dataset]['n_channels']

    cnn_cfg = method_config[dataset]['features_CNN']
    batch_size = cnn_cfg['batch_size']
    # Globals consumed by objective().
    SEARCH_SPACE = cnn_cfg['search_space']
    NUM_CHANNELS = num_channels

    data = load_data(f"results/features/{type_dataset}_static_features",
                   f"results/features/{type_dataset}_static_labels")
    
    X = data.drop(columns=['is_laundering'])
    y = data['is_laundering']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1997)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=1997)

    transform = transforms_channels(X_train.values, n_channels=num_channels)

    dataset_train = ImageDataset(X_train.values, y_train, n_channels=num_channels, transform=transform)
    dataset_val = ImageDataset(X_val.values, y_val, n_channels=num_channels, transform=transform)
    dataset_test = ImageDataset(X_test.values, y_test, n_channels=num_channels, transform=transform)

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    os.makedirs("results/tuning", exist_ok=True)

    # Persist to SQLite so the study can be resumed and shared across parallel
    # array tasks; one study per dataset (load_if_exists -> append, not clobber).
    study_name = f"CNN_{type_dataset}"
    storage = cnn_cfg.get('storage')
    if storage:
        storage = storage.format(dataset=type_dataset)
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=cnn_cfg['n_trials'])
    cnn_params = study.best_params
    cnn_values = study.best_value

    # Machine-readable best params (for a downstream final-train run) ...
    with open(f"results/tuning/{study_name}_best_params.json", "w") as f:
        json.dump({"dataset": type_dataset, "best_params": cnn_params,
                   "best_value_AUCPR": cnn_values}, f, indent=2)
    # ... and a human-readable copy.
    with open(f"results/tuning/{study_name}_params.txt", "w") as f:
        f.write(str(cnn_params))
        f.write("\n")
        f.write("AUC-PRC: "+str(cnn_values))