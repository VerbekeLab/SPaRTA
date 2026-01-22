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

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"); print(f"Training on: {device}")

batch_size = 512

dataset = NetworkImageDataset()
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = CNN_visual_VGG16().to(device)

y_train = dataset.y.numpy()
train_weight = round((y_train == 0).sum() / (y_train == 1).sum())*50; print(f"Positive class weight: {train_weight}")

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([train_weight]).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

print("Starting training...")
print("Number of batches:", len(loader))

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, labels in tqdm(loader):
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)
            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).int()
            y_pred.extend(torch.sigmoid(outputs).cpu().numpy().tolist())
            y_true.extend(labels.float().cpu().numpy().tolist())
    print(f'Epoch [{epoch+1}/{num_epochs}], AUC-ROC: {roc_auc_score(y_true, y_pred):.4f}')

# save the trained model
torch.save(model.state_dict(), 'results/models/cnn_visual_vgg16_experiment_pictures.pth')