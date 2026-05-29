import os
import sys
DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import xgboost as xgb
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.nn as nn

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from src.utils.setup import load_config

from src.methods.models import NeuralNetwork

def data_prep_features(data_directory, dataset_type):
    feature_directory = f'{data_directory}/{dataset_type}_static_features_t.csv'
    label_directory = f'{data_directory}/{dataset_type}_static_labels.csv'
    df_features = pd.read_csv(feature_directory)
    df_labels = pd.read_csv(label_directory)
    df_labels.columns = ['node', 'label']

    df = pd.merge(df_features, df_labels, on='node', how='inner')

    X = df.drop(columns=['node','label'])
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1997) 
    return X_train, X_test, y_train, y_test

def experiment_features_LogisticRegression(X_train, X_test, y_train, y_test):
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)

    y_pred_proba = logreg.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    AUC_ROC = roc_auc_score(y_test, y_pred_proba)

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    AUC_PR = average_precision_score(y_test, y_pred_proba)

    with open('results/experiments/features_LogisticRegression.txt', 'w') as f:
        f.write(f'AUC ROC: {AUC_ROC}\n')
        f.write(f'AUC PR: {AUC_PR}\n')
        f.write(f'FPR: {fpr}\n')
        f.write(f'TPR: {tpr}\n')
        f.write(f'Precision: {precision}\n')
        f.write(f'Recall: {recall}\n')

def experiment_features_XGBoost_tune(X_train, y_train):
    dtrain = xgb.DMatrix(X_train, label=y_train)

    params = {'objective': ['binary:logistic'], 
                'max_depth': [3, 4, 5, 6], # depth of each tree
                'eta': [0.01, 0.1, 0.2], # learning rate
                'subsample': [0.6, 0.8, 1.0], # fraction of data to use per tree
                }

    gbm = xgb.XGBClassifier()
    gridsearch = GridSearchCV(estimator=gbm, param_grid=params, scoring='roc_auc', cv=5, verbose=1)
    gridsearch.fit(X_train, y_train)

    with open('results/experiments/HPT_features_XGBoost.txt', 'w') as f:
        f.write(f'Best_parameters:{gridsearch.best_params_}\n')
        f.write(f'Best_AUC_ROC:{gridsearch.best_score_}\n')

    return gridsearch.best_params_


def experiment_features_XGBoost(X_train, X_test, y_train, y_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = experiment_features_XGBoost_tune(X_train, y_train)
    model = xgb.train(params, dtrain, num_boost_round=200)
    y_pred_proba = model.predict(dtest)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    AUC_ROC = roc_auc_score(y_test, y_pred_proba)

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    AUC_PR = average_precision_score(y_test, y_pred_proba)

    with open('results/experiments/features_XGBoost.txt', 'w') as f:
        f.write(f'AUC ROC: {AUC_ROC}\n')
        f.write(f'AUC PR: {AUC_PR}\n')
        f.write(f'FPR: {fpr}\n')
        f.write(f'TPR: {tpr}\n')
        f.write(f'Precision: {precision}\n')
        f.write(f'Recall: {recall}\n')


def experiment_features_NN(X_train, X_test, y_train, y_test):
    input_size = X_train.shape[1]
    hidden_size = 16
    output_size = 1

    n_epochs = 100
    learning_rate = 0.001

    model = NeuralNetwork(
        num_layers=2,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size
    )   

    train_weight = round((y_train == 0).sum() / (y_train == 1).sum())*10

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([train_weight]))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor).squeeze()
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()


        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor).squeeze()
            test_loss = criterion(test_outputs, y_test_tensor)
            train_acc = ((torch.sigmoid(outputs) >= 0.2).float() == y_train_tensor).float().mean()
            test_acc = ((torch.sigmoid(test_outputs) >= 0.2).float() == y_test_tensor).float().mean()
        
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor).squeeze()
        fpr, tpr, thresholds = roc_curve(y_test, torch.sigmoid(test_outputs))
        AUC_ROC = roc_auc_score(y_test, torch.sigmoid(test_outputs))

        precision, recall, thresholds = precision_recall_curve(y_test, torch.sigmoid(test_outputs))
        AUC_PR = average_precision_score(y_test, torch.sigmoid(test_outputs))

    with open('results/experiments/features_NN.txt', 'w') as f:
        f.write(f'AUC ROC: {AUC_ROC}\n')
        f.write(f'AUC PR: {AUC_PR}\n')
        f.write(f'FPR: {fpr}\n')
        f.write(f'TPR: {tpr}\n')
        f.write(f'Precision: {precision}\n')
        f.write(f'Recall: {recall}\n')

if __name__ == "__main__":
    data_config = load_config("config/data/config.yaml")
    method_config = load_config("config/methods/config.yaml")

    network = data_config['parameters']['dataset']
    dataset_type = data_config[network]['type_dataset']

    experiment = method_config['experiment']
    experiment_params = method_config[network][experiment]
    data_directory = experiment_params['data_directory']

    X_train, X_test, y_train, y_test = data_prep_features(data_directory, dataset_type)
    experiment_features_LogisticRegression(X_train, X_test, y_train, y_test)
    experiment_features_XGBoost(X_train, X_test, y_train, y_test)
    experiment_features_NN(X_train, X_test, y_train, y_test)
