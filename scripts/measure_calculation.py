import os
import sys
DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from src.utils.setup import load_config
from src.utils.graph_processing import graph_community

from src.data.network_data_loader import *
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

    return tuple([node] + results['measure'] + results['size'])

n_cpu = min(4, cpu_count() // 2)
if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    data_config = load_config("config/data/config.yaml")

    network = data_config['parameters']['dataset']
    dynamic = data_config['parameters']['time_dynamic']

    type_dataset = data_config[network]['type_dataset']

    start_dates = data_config[network]['start_dates']
    end_dates = data_config[network]['end_dates']

    if network == 'IBM':
        if dynamic:
            networks = construct_network_ibm_time(start_dates, end_dates)
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
                    initargs=(G_reduced, G_rev, G_und,)
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
                out_path = f"results/{type_dataset}_dynamic_{i}.csv"
                df.to_csv(out_path, index=False)
                print(f"Results saved to {out_path}")


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
                initargs=(G_reduced, G_rev, G_und,)
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
            out_path = f"results/{type_dataset}_static.csv"
            df.to_csv(out_path, index=False)
            print(f"Results saved to {out_path}")

    else:
        pass

