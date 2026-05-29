import torch
import pickle
import numpy as np
import h5py
from tqdm import tqdm
import os

num_files = 422
#all_pictures = torch.empty((0, 3, 224, 224), dtype=torch.uint8)

path_tensor='results/pickle/'
path_save = 'src/data/pictures/processed/'

all_pictures_list = []

def unpack_batch(packed_batch):
    # packed_batch: shape (batch_size, 9600)
    unpacked = np.unpackbits(packed_batch, axis=1)[:, :3*224*224]
    return torch.tensor(unpacked.reshape(-1, 3, 224, 224), dtype=torch.uint8)

os.makedirs(path_save, exist_ok=True)

# Load node order of tensors
for i in tqdm(range(num_files)):
    f = h5py.File(f'{path_tensor}images_tensor_{i}.h5', 'r')
    packed_batch = f['images_bitmap']
    unpacked_batch = unpack_batch(packed_batch)
    all_pictures_list.append(unpacked_batch)
    f.close()

all_pictures = torch.cat(all_pictures_list, dim=0)
print(f"Shape of all pictures tensor: {all_pictures.shape}")
print(f"Total pictures loaded: {all_pictures.shape[0]}")
print(f"Saving to {path_save}images_tensor.pkl")

with open(f'{path_save}images_tensor.pkl', 'wb') as f:
    pickle.dump(all_pictures, f)