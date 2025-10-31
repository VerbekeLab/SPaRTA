import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, X, y, n_channels=1, transform=None):
        self.X = torch.tensor(X, dtype=torch.float32).view(-1, n_channels, 3, 3)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]
        if self.transform:
            image = self.transform(image)
        return image, label