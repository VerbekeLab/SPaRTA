import os
import sys
DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import warnings; warnings.simplefilter('ignore')

import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from src.utils.setup import load_config
from src.utils.graph_processing import graph_community

from src.data.network_data_loader import *
from src.data.utils.dates import define_dates
from src.methods.neighbourhood_picture import *

# Global variable for worker processes
graph_for_worker = None
graph_for_worker_rev = None
graph_for_worker_undirected = None

def init_worker(graph, graph_rev, graph_undirected):
    """
    Initializer for worker processes to set the graph in each subprocess.
    """
    global graph_for_worker
    global graph_for_worker_rev
    global graph_for_worker_undirected
    graph_for_worker = graph
    graph_for_worker_rev = graph_rev
    graph_for_worker_undirected = graph_undirected

def process_node(node):
    results = node_measures(
        node,
        graph_for_worker,
        graph_for_worker_undirected,
        graph_for_worker_rev,
        output_tensor=False
    )
    results_list = [node] + results['measure'] + results['size']

    for weight in ['amount_trans', 'num_trans']:
        for agg in ['sum', 'mean', 'max', 'std']:
            key_measure = f'transaction_{weight}_summary_{agg}'
            results_list += results[key_measure]

    return tuple(results_list)

n_cpu = min(4, cpu_count() // 2)
if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    data_config = load_config("config/data/config.yaml")
    method_config = load_config("config/methods/config.yaml")

    network = data_config['parameters']['dataset']
    type_dataset = data_config[network]['type_dataset']
    dynamic = data_config['parameters']['time_dynamic']

    if dynamic:
        time_step=data_config[network]['network_construction']['time_step']
        time_width=data_config[network]['network_construction']['time_width']
        time_type=data_config[network]['network_construction']['time_type']

        echo = data_config[network]['network_construction']['echo']
        days_echo = data_config[network]['network_construction']['days_echo']

    keys_to_include = ['measure_00', 'measure_01', 'measure_02', 
                        'measure_10', 'measure_11', 'measure_12',
                        'measure_20', 'measure_21', 'measure_22',
                        'size_00', 'size_01', 'size_02', 
                        'size_10', 'size_11', 'size_12',
                        'size_20', 'size_21', 'size_22',
                        ]
            
    weights = ['amount_trans', 'num_trans']
    aggregations = ['sum', 'mean', 'max', 'std']

    for weight in weights:
        for agg in aggregations:
            for suff in ['00', '01', '02', '10', '11', '12', '20', '21', '22']:
                key_measure = f'transaction_{weight}_summary_{agg}_{suff}'
                keys_to_include.append(key_measure)

    if network in ('IBM', 'AMLSim'):
        if dynamic:
            start_dates, end_dates = define_dates(
                load_transactions(network, type_dataset=type_dataset)['timestamp'],
                time_step=time_step,
                time_width=time_width,
                time_type=time_type
            )

            networks = construct_network_time(start_dates, end_dates, dataset=network, type_dataset=type_dataset, echo=echo, days_echo=days_echo)
            for i in range(len(networks)):
                print(f"Processing dynamic network snapshot {i+1}/{len(networks)}...")
                G, labels = networks[i]
                G_reduced = graph_community(G)
                G_und = G_reduced.to_undirected()
                G_rev = G_reduced.reverse(copy=True)
                nodes = list(G_reduced.nodes)
                chunksize = max(1, len(nodes) // (n_cpu * 10))
                print(f"Number of nodes: {len(nodes)} | Using {n_cpu} processes")
                with Pool(
                    processes=n_cpu,
                    initializer=init_worker,
                    initargs=(G_reduced, G_rev, G_und,)
                ) as pool:
                    results = list(tqdm(pool.imap(process_node, nodes, chunksize=chunksize), total=len(nodes)))

                df = pd.DataFrame(results, columns=['node'] + keys_to_include)

                if echo:
                    out_path_features = f"results/features/{type_dataset}_dynamic_{i}_features_echo_t.csv"
                    out_path_labels = f"results/features/{type_dataset}_dynamic_{i}_labels_echo_t.csv"
                else:
                    out_path_features = f"results/features/{type_dataset}_dynamic_{i}_features_t.csv"
                    out_path_labels = f"results/features/{type_dataset}_dynamic_{i}_labels_t.csv"

                df.to_csv(out_path_features, index=False)
                print(f"Results saved to {out_path_features}")
                labels.to_csv(out_path_labels)
                print(f"Labels saved to {out_path_labels}")


        else:
            print("Processing static network...")
            G, labels = construct_network(dataset=network, type_dataset=type_dataset)

            G_reduced = graph_community(G)
            G_und = G_reduced.to_undirected()
            G_rev = G_reduced.reverse(copy=True)

            nodes = list(G_reduced.nodes)
            chunksize = max(1, len(nodes) // (n_cpu * 10))
            print(f"Number of nodes: {len(nodes)} | Using {n_cpu} processes")
            with Pool(
                processes=n_cpu,
                initializer=init_worker,
                initargs=(G_reduced, G_rev, G_und,)
            ) as pool:
                results = list(tqdm(pool.imap(process_node, nodes, chunksize=chunksize), total=len(nodes)))

            df = pd.DataFrame(results, columns=['node'] + keys_to_include)

            out_path_features = f"results/features/{type_dataset}_static_features_t.csv"
            df.to_csv(out_path_features, index=False)
            print(f"Results saved to {out_path_features}")
            out_path_labels = f"results/features/{type_dataset}_static_labels_t.csv"
            labels.to_csv(out_path_labels)
            print(f"Labels saved to {out_path_labels}")

    else:
        raise ValueError(f"Feature extraction not supported for dataset: {network}")

