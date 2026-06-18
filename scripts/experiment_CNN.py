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
from src.utils.setup import load_config, resolve_dataset

from sklearn.metrics import roc_auc_score, average_precision_score

import optuna

def load_data(features_path, labels_path):
    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)
    data = pd.merge(features_df, labels_df, left_on='node', right_on='Account')
    data.drop(columns=['node', 'Account'], inplace=True)
    return data

def transforms_channels(X, n_channels):
    X_tensor = torch.tensor(X, dtype=torch.float32).view(-1, n_channels, 3, 3)
    X_mean = torch.mean(X_tensor, dim = [0,2,3])
    X_std = torch.std(X_tensor, dim = [0,2,3])
    return transforms.Normalize(X_mean, X_std)

def objective(trial):
    ### To Do ###
    # Define hyperparameter search space
    num_channels = 10
    num_epochs = trial.suggest_int('num_epochs', 10, 500, step=10)
    hidden_channels = trial.suggest_int('hidden_channels', 8, 64, step=2)
    num_layers = trial.suggest_int('num_layers', 2, 5)
    kernel_size = trial.suggest_int('kernel_size', 2, 3)
    max_pool = trial.suggest_categorical('max_pool', [True, False])
    half_final_layer = trial.suggest_categorical('half_final_layer', [True, False])
    scale_labels = trial.suggest_int('scale_labels', 0, 100000, step=100)
    #learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    learning_rate = trial.suggest_categorical('learning_rate', [1e-5, 1e-4, 1e-3, 1e-2])

    model = CNN(
        num_channels=num_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        kernel_size=kernel_size,
        max_pool=max_pool,
        half_final_layer=half_final_layer
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

    batch_size = 512

    data_config = load_config("config/data/config.yaml")
    method_config = load_config("config/methods/config.yaml")

    dataset = resolve_dataset(data_config)
    type_dataset = data_config[dataset]['type_dataset']

    num_channels = data_config[dataset]['n_channels']

    data = load_data(f"results/features/{type_dataset}_static_features_t.csv",
                   f"results/features/{type_dataset}_static_labels_t.csv")
    
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

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    cnn_params = study.best_params
    cnn_values = study.best_value
    with open("results/tuning/CNN_params.txt", "w") as f:
        f.write(str(cnn_params))
        f.write("\n")
        f.write("AUC-PRC: "+str(cnn_values))