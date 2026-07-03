import os
import sys
DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import warnings; warnings.simplefilter('ignore')

import gc
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from src.utils.setup import load_config, resolve_dataset, resolve_dynamic, resolve_timing, run_tag
from src.utils.graph_processing import graph_community
from src.utils.feature_io import save_table

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

n_cpu = min(4, cpu_count() // 2)

def process_graph(G):
    G_reduced = graph_community(G)
    G_und = G_reduced.to_undirected()
    G_rev = G_reduced.reverse(copy=True)

    nodes = list(G_reduced.nodes)
    # Keep chunks small so imap hands results back frequently: a large chunksize
    # makes a worker finish its whole chunk before returning anything, so tqdm
    # only advances in big bursts (and the first update stalls until the first
    # full chunk completes). ~200 chunks/worker gives a smooth bar; the per-result
    # IPC cost is negligible for these lightweight tuples.
    chunksize = max(1, len(nodes) // (n_cpu * 200))
    print(f"Number of nodes: {len(nodes)} | Using {n_cpu} processes")
    with Pool(
        processes=n_cpu,
        initializer=init_worker,
        initargs=(G_reduced, G_rev, G_und,)
    ) as pool:
        # imap_unordered surfaces each chunk as soon as ANY worker finishes it
        # (ordered imap would block on chunk 0 even if later chunks are ready).
        # Order is irrelevant here: every row embeds its own node id (see process_node).
        results = list(tqdm(pool.imap_unordered(process_node, nodes, chunksize=chunksize), total=len(nodes)))

    return pd.DataFrame(results, columns=['node'] + keys_to_include)

if __name__ == "__main__":
    # Own our output directory instead of relying on the Slurm wrapper's mkdir: the
    # static branch below writes straight into 'results/features' with no further
    # makedirs, so it must exist here regardless of where/how this script is invoked.
    os.makedirs('results/features', exist_ok=True)
    data_config = load_config("config/data/config.yaml")
    method_config = load_config("config/methods/config.yaml")

    network = resolve_dataset(data_config)
    type_dataset = data_config[network]['type_dataset']
    dynamic = resolve_dynamic(data_config)

    if dynamic:
        # SPARTA_* env vars let one Slurm array task pick a timing combo without editing
        # the YAML (see resolve_timing); run_tag namespaces this combo's output dir so a
        # sweep's combos coexist in results/features/<tag>/ instead of overwriting.
        timing = resolve_timing(data_config, network)
        time_step = timing['time_step']
        time_width = timing['time_width']
        time_type = timing['time_type']

        echo = timing['echo']
        days_echo = timing['days_echo']

        out_dir = os.path.join("results/features", run_tag(timing))

    if network in ('AMLWorld', 'AMLSim', 'Tide'):
        if dynamic:
            # Load the transactions ONCE and share the frame between the date grid and
            # the snapshot generator. The old eager construct_network_time re-read the
            # raw files for every window AND kept all ~370 snapshot graphs alive at
            # once, which got long runs OOM-killed by Slurm.
            transactions = load_transactions(network, type_dataset=type_dataset)
            start_dates, end_dates = define_dates(
                transactions['timestamp'],
                time_step=time_step,
                time_width=time_width,
                time_type=time_type
            )
            n_snapshots = len(start_dates)

            networks_iter = construct_network_time_iter(
                start_dates, end_dates, dataset=network, type_dataset=type_dataset,
                echo=echo, days_echo=days_echo, transactions=transactions
            )
            os.makedirs(out_dir, exist_ok=True)
            # Manual counter, NOT enumerate(): CPython's enumerate (and zip) reuse
            # their cached result tuple, which keeps a strong ref to the PREVIOUS
            # (G, labels) until AFTER the generator has built the next snapshot —
            # holding two graphs resident and defeating the del/gc below.
            i = -1
            for G, labels in networks_iter:
                i += 1
                print(f"Processing dynamic network snapshot {i+1}/{n_snapshots}...")
                df = process_graph(G)

                suffix = "_echo" if echo else ""
                stem_features = os.path.join(out_dir, f"{type_dataset}_dynamic_{i}_features{suffix}")
                stem_labels = os.path.join(out_dir, f"{type_dataset}_dynamic_{i}_labels{suffix}")

                out_path_features = save_table(df, stem_features)
                print(f"Results saved to {out_path_features}")
                # reset_index() keeps the Account index as the first column, matching
                # the (index-writing) to_csv layout the legacy label files carry.
                out_path_labels = save_table(labels.reset_index(), stem_labels)
                print(f"Labels saved to {out_path_labels}")

                # Free this snapshot before the generator builds the next one — the
                # loop variables would otherwise keep the old graph alive alongside it.
                del G, labels, df
                gc.collect()


        else:
            print("Processing static network...")
            G, labels = construct_network(dataset=network, type_dataset=type_dataset)
            df = process_graph(G)

            out_path_features = save_table(df, f"results/features/{type_dataset}_static_features")
            print(f"Results saved to {out_path_features}")
            out_path_labels = save_table(labels.reset_index(), f"results/features/{type_dataset}_static_labels")
            print(f"Labels saved to {out_path_labels}")

    else:
        raise ValueError(f"Feature extraction not supported for dataset: {network}")

