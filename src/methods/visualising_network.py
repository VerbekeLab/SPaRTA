# LOAD MODULES
# Standard library
import os
import sys

# Third party
from tqdm import tqdm

# NOTE: Your script is not in the root directory. We must hence change the system path
DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import networkx as nx
import igraph as ig
import cairo
from PIL import Image
import pickle
import h5py

import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from src.data.network_data_loader import construct_network
from src.utils.graph_processing import graph_degree_rel, graph_degree_abs, graph_community

def define_vertex_color(G, n, G_first):
    vertex_color = []
    for v in G.vs:
        if v.index == n:
            vertex_color.append("#FF00FF")
        elif v.index in G_first:
            vertex_color.append("#FFFFFF")
        else:
            vertex_color.append("#00FFFF")
    return vertex_color

def define_edge_color(G, n):
    edge_color = []
    for e in G.es:
        source = G.vs[e.source].index
        target = G.vs[e.target].index
        if (source == n) or (target == n):
            edge_color.append("#FF00FF")
        else:
            edge_color.append("#00FFFF")
    return edge_color

def define_visual_style(
    vertex_color,
    layout_H,
    edge_color
        ):
    visual_style = {}
    visual_style["vertex_size"] = 5
    visual_style["vertex_label"]=None
    visual_style["vertex_color"] = vertex_color
    visual_style["layout"] = layout_H
    visual_style["edge_width"] = 1.1
    visual_style["background"] = "black"
    visual_style["margin"] = 5
    visual_style["edge_color"] = edge_color
    visual_style["bbox"] = (224, 224)
    visual_style["edge_arrow_width"] = 0.7
    visual_style["edge_arrow_size"] = 0.4
    return visual_style

def visualise_network_node(G, n, layout = 'fr'):
    n_name = G.vs[n]["_nx_name"]
    G_second = G.neighborhood(vertices=n, order=2)

    # convert to igraph
    H = G.induced_subgraph(G_second)
    name_to_idx = {v["_nx_name"]: v.index for v in H.vs}
    n_H = name_to_idx[n_name]

    G_first = H.neighborhood(vertices=n_H, order=1)

    layout_H = H.layout(layout)

    vertex_color = define_vertex_color(H, n_H, G_first)
    edge_color = define_edge_color(H, n_H)
    visual_style = define_visual_style(
        vertex_color,
        layout_H,
        edge_color
    )

    image_node = ig.plot(H, **visual_style)
    image_node.redraw()
    surface = image_node.surface
    ctx=cairo.Context(surface)
    buf = surface.get_data()
    arr = torch.frombuffer(buf, dtype=torch.uint8)/255
    img = arr.view(224, 224, 4)  # height, width, channels
    img = img.permute(2, 0, 1)  # channels, height, width
    # Remove alpha channel
    img_rgb = img[:3, :, :]  # channels, height, width
    img_rgb = img_rgb.round().contiguous()
    return n_name, img_rgb 

def init_worker(G):
    """
    Initializer for worker processes to set the graph in each subprocess.
    """
    global G_worker
    G_worker = G

def process_node(n):
    return visualise_network_node(G_worker, n)

def pack_batch(batch):
    # Convert to NumPy
    arr = batch.numpy().astype(np.uint8)  
    # Flatten each tensor to (C*H*W)
    flat = arr.reshape(batch.shape[0], -1)  
    # Pack bits along last axis
    packed = np.packbits(flat, axis=1)     
    return packed

n_cpu = min(4, cpu_count() // 2)

if __name__ == "__main__":
    os.makedirs('results/pickle', exist_ok=True)
    G, labels = construct_network(dataset='AMLWorld')
    G = graph_degree_abs(G, degree_cutoff=20)
    G = ig.Graph.from_networkx(G)

    nodes = list(range(len(G.vs)))
    num_nodes = len(nodes)
    print(f"Number of nodes: {num_nodes} | Using {n_cpu} processes")

    num_samples = 1000

    with Pool(
        processes=n_cpu,
        initializer=init_worker,
        initargs=(G,)
    ) as pool:

        for batch_i, i in enumerate(tqdm(range(0, num_nodes, num_samples))):

            nodes_to_use = nodes[i : i + num_samples]

            # --- process nodes using the SAME pool ---
            results = list(pool.imap(process_node, nodes_to_use))

            # unpack worker outputs
            nodes_batch, images = zip(*results)

            # convert image list → tensor
            images_tensor = torch.stack(images)

            images_bitmap = pack_batch(images_tensor)  # (N,160,160,3) → (N, 9600)

            # save outputs
            with open(f'results/pickle/nodes_{batch_i}.pkl', 'wb') as f:
                pickle.dump(nodes_batch, f)

            with h5py.File(f'results/pickle/images_tensor_{batch_i}.h5', 'w') as f:
                f.create_dataset('images_bitmap', data=images_bitmap)

            # --- free memory from this batch ---
            del results
            del images
            del images_tensor


    print("Done!")
