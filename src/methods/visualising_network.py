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
from PIL import Image

import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from src.data.network_data_loader import construct_network_ibm
from src.utils.graph_processing import graph_degree_rel, graph_degree_abs, graph_community

def define_vertex_color(G, n, G_first):
    vertex_color = []
    for v in G.vs:
        if v.index == n:
            vertex_color.append("#FF00FF80")
        elif v.index in G_first:
            vertex_color.append("#FFFFFF80")
        else:
            vertex_color.append("#00FFFF80")
    return vertex_color

def define_edge_color(G, n):
    edge_color = []
    for e in G.es:
        source = G.vs[e.source].index
        target = G.vs[e.target].index
        if (source == n) or (target == n):
            edge_color.append("#FF00FF80")
        else:
            edge_color.append("#00FFFF80")
    return edge_color

def define_visual_style(
    vertex_color,
    layout_H,
    edge_color
        ):
    visual_style = {}
    visual_style["vertex_size"] = 12
    visual_style["vertex_label"]=None
    visual_style["vertex_color"] = vertex_color
    visual_style["layout"] = layout_H
    visual_style["edge_width"] = 1.2
    visual_style["background"] = "black"
    visual_style["margin"] = 50
    visual_style["edge_color"] = edge_color
    visual_style["bbox"] = (500, 500)
    visual_style["edge_arrow_width"] = 0.8
    visual_style["edge_arrow_size"] = 0.9
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
    image_node.save(f'results/images/visualisation_network_node_{n_name}.png')
    image = Image.open(f'results/images/visualisation_network_node_{n_name}.png')
    os.remove(f'results/images/visualisation_network_node_{n_name}.png')
    img = torch.ceil(transforms.ToTensor()(image))

    img_flat = img.view(-1)

    return n_name, img_flat


def init_worker(G):
    """
    Initializer for worker processes to set the graph in each subprocess.
    """
    global G_worker
    G_worker = G

def process_node(n):
    result = visualise_network_node(G_worker, n)
    return result

n_cpu = min(4, cpu_count() // 2)

if __name__ == "__main__":
    os.makedirs('results/images', exist_ok=True)
    G, labels = construct_network_ibm()
    G = graph_degree_abs(G, degree_cutoff=20)
    #G = graph_community(G, resolution=10)
    G = ig.Graph.from_networkx(G)

    nodes = list(range(len(G.vs)))
    print(f"Number of nodes: {len(nodes)} | Using {n_cpu} processes")
    with Pool(
        processes=n_cpu,
        initializer=init_worker,
        initargs=(G,)
    ) as pool:
        results = list(tqdm(pool.imap(process_node, nodes), total=len(nodes)))


    nodes, images = zip(*results)


    # Convert images (tuple of 1D torch tensors) to numpy 1D arrays of ints
    imgs_np = [img.cpu().numpy().astype(np.uint8).ravel() for img in images]

    # Build dataframe: one row per node, columns = node + pixel features
    num_pixels = imgs_np[0].shape[0]
    cols = ["node"] + [f"px_{i}" for i in range(num_pixels)]
    rows = [[n] + img.tolist() for n, img in zip(nodes, imgs_np)]
    df = pd.DataFrame(rows, columns=cols)

    # Write to parquet
    df.to_csv("results/images/features.csv", index=False)


    print("Done!")
