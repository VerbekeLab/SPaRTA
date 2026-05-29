import pandas as pd
import torch
import pickle
from torch.utils.data import Dataset
import numpy as np
import h5py
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self, X, y, n_channels=1, transform=None):
        self.X = torch.tensor(X, dtype=torch.float32).contiguous().view(-1, n_channels, 3, 3)
        self.y = torch.tensor(y.to_numpy(), dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def unpack_batch(packed_batch):
    # packed_batch: shape (batch_size, 9600)
    unpacked = np.unpackbits(packed_batch, axis=1)[:, :3*224*224]
    return torch.tensor(unpacked.reshape(-1, 3, 224, 224), dtype=torch.uint8)

class NetworkImageDataset(Dataset):
    def _load_data_individual(self, path_tensor, i):
        with open(f'{path_tensor}nodes_{i}.pkl', 'rb') as f:
            nodes = list(pickle.load(f))
        f = h5py.File(f'{path_tensor}images_tensor_{i}.h5', 'r')
        packed_batch = f['images_bitmap']
        unpacked_batch = unpack_batch(packed_batch)
        f.close()
        return nodes, unpacked_batch
    
    def _load_tensor_data(self, path_tensor):
        num_files = 422
        all_pictures_list = []
        all_nodes = []
        # Load node order of tensors
        for i in tqdm(range(num_files)):
            nodes, unpacked_batch = self._load_data_individual(path_tensor, i)
            all_nodes.extend(nodes)
            all_pictures_list.append(unpacked_batch)

        all_pictures = torch.cat(all_pictures_list, dim=0)

        return all_nodes, all_pictures

    def __init__(self, path_tensor='results/pickle/', path_labels='results/features/', dataset_type='HI-Small'):
        super().__init__()
        all_nodes, all_pictures = self._load_tensor_data(path_tensor)
        self.X = all_pictures
        # Load label dataframe
        labels_df = pd.read_csv(f'{path_labels}{dataset_type}_static_labels.csv')
        labels_df.columns = ['node', 'label']
        # Reindex labels to match node order in tensors
        labels_df = labels_df.set_index('node').reindex(all_nodes).reset_index()
        self.y = torch.tensor(labels_df['label'], dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        features = self.X[idx]
        label = self.y[idx]
        return features, label
