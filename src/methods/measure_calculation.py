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

def init_worker(graph, graph_rev, graph_undirected, weight):
    """
    Initializer for worker processes to set the graph in each subprocess.
    """
    global graph_for_worker
    global graph_for_worker_rev
    global graph_for_worker_undirected
    global weight_for_worker
    graph_for_worker = graph
    graph_for_worker_rev = graph_rev
    graph_for_worker_undirected = graph_undirected
    weight_for_worker = weight


def process_node(node):
    results = node_measures(
        node,
        graph_for_worker,
        graph_for_worker_undirected,
        graph_for_worker_rev,
        output_tensor=False, 
        weight=weight_for_worker
    )

    return tuple([node] + results['measure'] + results['size'])

n_cpu = min(4, cpu_count() // 2)
if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    data_config = load_config("config/data/config.yaml")
    method_config = load_config("config/methods/config.yaml")

    network = data_config['parameters']['dataset']
    dynamic = data_config['parameters']['time_dynamic']

    if dynamic:
        type_dataset = data_config[network]['type_dataset']
        time_step=data_config[network]['network_construction']['time_step']
        time_width=data_config[network]['network_construction']['time_width']
        time_type=data_config[network]['network_construction']['time_type']

        echo = data_config[network]['network_construction']['echo']
        days_echo = data_config[network]['network_construction']['days_echo']

    if network == 'IBM':
        if dynamic:
            start_dates, end_dates = define_dates(
                load_transactions_ibm(type_dataset=type_dataset)['Timestamp'],
                time_step=time_step,
                time_width=time_width,
                time_type=time_type
            )

            networks = construct_network_ibm_time(start_dates, end_dates, type_dataset=type_dataset, echo=echo, days_echo=days_echo)
            for i in range(len(networks)):
                print(f"Processing dynamic network snapshot {i+1}/{len(networks)}...")
                G, labels = networks[i]
                G_reduced = graph_community(G)
                G_und = G_reduced.to_undirected()
                G_rev = G_reduced.reverse(copy=True)
                nodes = list(G_reduced.nodes)
                print(f"Number of nodes: {len(nodes)} | Using {n_cpu} processes")
                with Pool(
                    processes=n_cpu,
                    initializer=init_worker,
                    initargs=(G_reduced, G_rev, G_und, 'weight',)
                ) as pool:
                    results = list(tqdm(pool.imap(process_node, nodes), total=len(nodes)))

                (
                    nodes, 
                    measure_00_list, measure_01_list, measure_02_list, measure_10_list, measure_11_list, measure_12_list, measure_20_list, measure_21_list, measure_22_list, 
                    size_00_list, size_01_list, size_02_list, size_10_list, size_11_list, size_12_list, size_20_list, size_21_list, size_22_list
                ) = zip(*results)

                df = pd.DataFrame({
                    "node": nodes,
                    "measure_00": measure_00_list,
                    "measure_01": measure_01_list,
                    "measure_02": measure_02_list,
                    "measure_10": measure_10_list,
                    "measure_11": measure_11_list,
                    "measure_12": measure_12_list,
                    "measure_20": measure_20_list,
                    "measure_21": measure_21_list,
                    "measure_22": measure_22_list, 
                    "size_00": size_00_list,
                    "size_01": size_01_list,
                    "size_02": size_02_list,
                    "size_10": size_10_list,
                    "size_11": size_11_list,
                    "size_12": size_12_list,
                    "size_20": size_20_list,
                    "size_21": size_21_list,
                    "size_22": size_22_list
                })

                if echo:
                    out_path_features = f"results/features/{type_dataset}_dynamic_{i}_features_echo.csv"
                    out_path_labels = f"results/features/{type_dataset}_dynamic_{i}_labels_echo.csv"
                else:
                    out_path_features = f"results/features/{type_dataset}_dynamic_{i}_features.csv"
                    out_path_labels = f"results/features/{type_dataset}_dynamic_{i}_labels.csv"
                
                df.to_csv(out_path_features, index=False)
                print(f"Results saved to {out_path_features}")
                labels.to_csv(out_path_labels)
                print(f"Labels saved to {out_path_labels}")


        else:
            print("Processing static network...")
            G, labels = construct_network_ibm()

            G_reduced = graph_community(G)
            G_und = G_reduced.to_undirected()
            G_rev = G_reduced.reverse(copy=True)

            nodes = list(G_reduced.nodes)
            print(f"Number of nodes: {len(nodes)} | Using {n_cpu} processes")
            with Pool(
                processes=n_cpu,
                initializer=init_worker,
                initargs=(G_reduced, G_rev, G_und, None,)
            ) as pool:
                results = list(tqdm(pool.imap(process_node, nodes), total=len(nodes)))

            (
                nodes, 
                measure_00_list, measure_01_list, measure_02_list, measure_10_list, measure_11_list, measure_12_list, measure_20_list, measure_21_list, measure_22_list, 
                size_00_list, size_01_list, size_02_list, size_10_list, size_11_list, size_12_list, size_20_list, size_21_list, size_22_list
            ) = zip(*results)

            df = pd.DataFrame({
                "node": nodes,
                "measure_00": measure_00_list,
                "measure_01": measure_01_list,
                "measure_02": measure_02_list,
                "measure_10": measure_10_list,
                "measure_11": measure_11_list,
                "measure_12": measure_12_list,
                "measure_20": measure_20_list,
                "measure_21": measure_21_list,
                "measure_22": measure_22_list, 
                "size_00": size_00_list,
                "size_01": size_01_list,
                "size_02": size_02_list,
                "size_10": size_10_list,
                "size_11": size_11_list,
                "size_12": size_12_list,
                "size_20": size_20_list,
                "size_21": size_21_list,
                "size_22": size_22_list
            })
            out_path_features = f"results/features/{type_dataset}_static_features.csv"
            df.to_csv(out_path_features, index=False)
            print(f"Results saved to {out_path_features}")
            out_path_labels = f"results/features/{type_dataset}_static_labels.csv"
            labels.to_csv(out_path_labels)
            print(f"Labels saved to {out_path_labels}")

    else:
        pass

