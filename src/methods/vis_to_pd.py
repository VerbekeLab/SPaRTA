import os
import sys

# NOTE: Your script is not in the root directory. We must hence change the system path
DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import torch
from torchvision import transforms
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np

import pickle

def transform_picture_to_tensor(file_path):
    image = Image.open(file_path)
    img = transforms.ToTensor()(image)

    img_name = os.path.basename(file_path).split('_')[-1].split('.')[0]

    return img_name, img

n_cpu = min(4, cpu_count() // 2)

if __name__ == "__main__":
    picture_data = os.listdir('results/images/')
    image_paths = [f"results/images/{file}" for file in picture_data if file.endswith('.pdf')]
    with Pool(
        processes=n_cpu
    ) as pool:
        results = list(tqdm(pool.imap(transform_picture_to_tensor, image_paths), total=len(image_paths)))

    nodes, images = zip(*results)

    images_tensor = torch.stack(images)

    with open('results/images/nodes.pkl', 'wb') as f:
        pickle.dump(list(nodes), f)

    with open('results/images/images_tensor.pkl', 'wb') as f:
        pickle.dump(images_tensor, f)

    print('Done!')