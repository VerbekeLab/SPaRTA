import torch

from src.methods.utils.measure_functions import *
from src.methods.utils.neighbourhood_functions import *
from src.methods.utils.summary_functions import *

def gargaml_node_measures(G_ego_second, size_0, size_1, size_2, nodes_ordered, output_tensor):
    adj_full = nx.adjacency_matrix(G_ego_second, nodelist=nodes_ordered, weight='weight').toarray()

    measure_00, size_00 = measure_00_function(adj_full, size_0)
    measure_01, size_01 = measure_01_function(adj_full, size_0, size_1)
    measure_02, size_02 = measure_02_function(adj_full, size_0, size_1, size_2)
    measure_10, size_10 = measure_10_function(adj_full, size_0, size_1)
    measure_11, size_11 = measure_11_function(adj_full, size_0, size_1)
    measure_12, size_12 = measure_12_function(adj_full, size_0, size_1, size_2)
    measure_20, size_20 = measure_20_function(adj_full, size_0, size_1, size_2)
    measure_21, size_21 = measure_21_function(adj_full, size_0, size_1)
    measure_22, size_22 = measure_22_function(adj_full, size_0, size_1, size_2)

    if output_tensor:
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
        
        result = {
            'measure': picture_measure.unsqueeze(0),
            'size': size_picture.unsqueeze(0)
            }

    else:
        picture_measure = [
            measure_00, measure_01, measure_02,
            measure_10, measure_11, measure_12,
            measure_20, measure_21, measure_22
        ]

        size_picture = [
            size_00, size_01, size_02,
            size_10, size_11, size_12,
            size_20, size_21, size_22
        ]

        result = {
            'measure': picture_measure,
            'size': size_picture
        }
    return result

def transaction_measures(G_ego_second, size_0, size_1, nodes_ordered, aggregations, weight='amount_trans'):
    # aggreation of the transaction (e.g., sum, mean, max)
    # weight = 'amount_trans' or 'num_trans'
    results_transaction = {}
    adj_full = nx.adjacency_matrix(G_ego_second, nodelist=nodes_ordered, weight=weight).toarray()
    for aggregation in aggregations:
        summary_00 = summary_00_function(adj_full, size_0, aggregation)
        summary_01 = summary_01_function(adj_full, size_0, size_1, aggregation)
        summary_02 = summary_02_function(adj_full, size_0, size_1, aggregation)
        summary_10 = summary_10_function(adj_full, size_0, size_1, aggregation)
        summary_11 = summary_11_function(adj_full, size_0, size_1, aggregation)
        summary_12 = summary_12_function(adj_full, size_0, size_1, aggregation)
        summary_20 = summary_20_function(adj_full, size_0, size_1, aggregation)
        summary_21 = summary_21_function(adj_full, size_0, size_1, aggregation)
        summary_22 = summary_22_function(adj_full, size_0, size_1, aggregation)

      
        results_transaction[f'transaction_{weight}_summary_{aggregation}'] = [
            summary_00, summary_01, summary_02,
            summary_10, summary_11, summary_12,
            summary_20, summary_21, summary_22
        ]

    return results_transaction

def node_measures(node, G_copy, G_copy_und, G_copy_rev, output_tensor = True):
    G_ego_second_und = nx.ego_graph(G_copy_und, node, radius=2)
    G_ego_second = nx.subgraph(G_copy, G_ego_second_und.nodes)
    G_ego_second_rev = nx.ego_graph(G_copy_rev, node, 2)

    nodes_0, nodes_1, nodes_2, nodes_ordered = node_selection(G_ego_second, G_ego_second_und, G_ego_second_rev, node)

    size_0 = len(nodes_0)
    size_1 = len(nodes_1)
    size_2 = len(nodes_2)

    result_gargaml = gargaml_node_measures(G_ego_second, size_0, size_1, size_2, nodes_ordered, output_tensor)
    result_transaction_amount = transaction_measures(G_ego_second, size_0, size_1, nodes_ordered, aggregations=['sum', 'mean', 'max', 'std'], weight='amount_trans')
    result_transaction_count = transaction_measures(G_ego_second, size_0, size_1, nodes_ordered, aggregations=['sum', 'mean', 'max', 'std'], weight='num_trans')
    
    result = { # combine all results into a single dictionary
        **result_gargaml,
        **result_transaction_amount,
        **result_transaction_count
    }

    return result
