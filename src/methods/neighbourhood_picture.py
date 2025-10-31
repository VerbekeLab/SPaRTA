import torch

from src.methods.utils.measure_functions import *
from src.methods.utils.neighbourhood_functions import *

def node_measures(node, G_copy, G_copy_und, include_size = False):
    G_ego_second_und = nx.ego_graph(G_copy_und, node, radius=2)
    G_ego_second = nx.subgraph(G_copy, G_ego_second_und.nodes)
    G_ego_second_rev = G_ego_second.reverse(copy=True)

    nodes_0, nodes_1, nodes_2, nodes_ordered = node_selection(G_ego_second, G_ego_second_und, G_ego_second_rev, node)

    adj_full = nx.adjacency_matrix(G_ego_second, nodelist=nodes_ordered, weight=None).toarray()

    size_0 = len(nodes_0)
    size_1 = len(nodes_1)
    size_2 = len(nodes_2)

    measure_00, size_00 = measure_00_function(adj_full, size_0)
    measure_01, size_01 = measure_01_function(adj_full, size_0, size_1)
    measure_02, size_02 = measure_02_function(adj_full, size_0, size_1, size_2)
    measure_10, size_10 = measure_10_function(adj_full, size_0, size_1)
    measure_11, size_11 = measure_11_function(adj_full, size_0, size_1)
    measure_12, size_12 = measure_12_function(adj_full, size_0, size_1, size_2)
    measure_20, size_20 = measure_20_function(adj_full, size_0, size_1, size_2)
    measure_21, size_21 = measure_21_function(adj_full, size_0, size_1)
    measure_22, size_22 = measure_22_function(adj_full, size_0, size_1, size_2)

    picture_measure = torch.tensor([
        [measure_00, measure_01, measure_02],
        [measure_10, measure_11, measure_12],
        [measure_20, measure_21, measure_22]
    ])

    size_picture = torch.tensor([
        [size_00, size_01, size_02],
        [size_10, size_11, size_12],
        [size_20, size_21, size_22]
    ])

    if include_size:
        # Concat pictures for the different channels
        return torch.cat([picture_measure.unsqueeze(0), size_picture.unsqueeze(0)], dim=0)
    else:
        return picture_measure.unsqueeze(0)
 