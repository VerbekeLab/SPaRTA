import os
import sys
DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score

from src.data.feature_data import NetworkImageDataset
from src.methods.models import CNN_visual_VGG16

batch_size = 4096

dataset = NetworkImageDataset()
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = CNN_visual_VGG16()

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images.type(torch.float32))
        loss = criterion(outputs, labels.float().unsqueeze(1))
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.type(torch.float32))
            predicted = (torch.sigmoid(outputs) > 0.5).int()
            y_pred.extend(torch.sigmoid(outputs).cpu().numpy().tolist())
            y_true.extend(labels.float().cpu().numpy().tolist())
    print(f'Epoch [{epoch+1}/{num_epochs}], AUC-ROC: {roc_auc_score(y_true, y_pred):.4f}')