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

from src.utils.setup import (load_config, resolve_dataset, resolve_dynamic, resolve_timing, run_tag,
                             resolve_order, static_stem)
from src.utils.graph_processing import graph_community
from src.utils.feature_io import find_table, save_table

from src.data.network_data_loader import *
from src.data.utils.dates import define_dates
from src.data.feature_schema import feature_columns, cell_indices
from src.methods.neighbourhood_picture import *

# Global variable for worker processes
graph_for_worker = None
graph_for_worker_rev = None
graph_for_worker_undirected = None
order_for_worker = 2
cells_for_worker = cell_indices(2)

def init_worker(graph, graph_rev, graph_undirected, order=2):
    """
    Initializer for worker processes to set the graph (and the neighbourhood order) in each
    subprocess.
    """
    global graph_for_worker
    global graph_for_worker_rev
    global graph_for_worker_undirected
    global order_for_worker
    global cells_for_worker
    graph_for_worker = graph
    graph_for_worker_rev = graph_rev
    graph_for_worker_undirected = graph_undirected
    order_for_worker = order
    cells_for_worker = cell_indices(order)

def process_node(node):
    # order 2: radius-2 ego network, all 9 cells (90 columns). order 1: radius-1 ego network,
    # keep only the cells holding the node's own edges (01, 10) -> 20 columns. The row layout
    # matches feature_columns(order), which names the DataFrame columns in process_graph.
    results = node_measures(
        node,
        graph_for_worker,
        graph_for_worker_undirected,
        graph_for_worker_rev,
        output_tensor=False,
        radius=order_for_worker
    )
    cells = cells_for_worker
    pick = lambda values: [values[i] for i in cells]
    results_list = [node] + pick(results['measure']) + pick(results['size'])

    for weight in ['amount_trans', 'num_trans']:
        for agg in ['sum', 'mean', 'max', 'std']:
            key_measure = f'transaction_{weight}_summary_{agg}'
            results_list += pick(results[key_measure])

    return tuple(results_list)

# The legacy 90-column order-2 schema. Kept under its historical name for notebooks and ad-hoc
# imports; the pipeline itself reads src.data.feature_schema.feature_columns(order).
keys_to_include = feature_columns(2)

weights = ['amount_trans', 'num_trans']
aggregations = ['sum', 'mean', 'max', 'std']

n_cpu = min(4, cpu_count() // 2)

def process_graph(G, order=None):
    order = resolve_order() if order is None else order
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
    print(f"Number of nodes: {len(nodes)} | Using {n_cpu} processes | neighbourhood order {order}")
    with Pool(
        processes=n_cpu,
        initializer=init_worker,
        initargs=(G_reduced, G_rev, G_und, order)
    ) as pool:
        # imap_unordered surfaces each chunk as soon as ANY worker finishes it
        # (ordered imap would block on chunk 0 even if later chunks are ready).
        # Order is irrelevant here: every row embeds its own node id (see process_node).
        results = list(tqdm(pool.imap_unordered(process_node, nodes, chunksize=chunksize), total=len(nodes)))

    return pd.DataFrame(results, columns=['node'] + feature_columns(order))

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

        # run_tag carries the neighbourhood order (infix _o1 for the first-order baseline).
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

            os.makedirs(out_dir, exist_ok=True)

            # RESUME: skip snapshots already on disk. Filtering the DATE GRID (not the loop
            # body) is what makes it free — the generator builds each graph BEFORE the body
            # runs, so a body-level `continue` would still pay construction for every window.
            # BOTH tables must be present: a kill between the two writes leaves a snapshot
            # with no labels, which _load_snapshot rejects. find_table (not os.path.exists)
            # so a legacy .csv run counts as done, and save_table writes atomically so a
            # present file is a complete one. This is what lets a wall-time-killed extraction
            # finish by resubmission instead of restarting from snapshot 0.
            suffix = "_echo" if echo else ""
            stem = lambda i, kind: os.path.join(out_dir, f"{type_dataset}_dynamic_{i}_{kind}{suffix}")
            pending = [i for i in range(n_snapshots)
                       if find_table(stem(i, "features")) is None
                       or find_table(stem(i, "labels")) is None]
            if not pending:
                print(f"All {n_snapshots} snapshots already extracted in {out_dir} — nothing to do.")
            elif len(pending) < n_snapshots:
                print(f"Resuming: {n_snapshots - len(pending)}/{n_snapshots} snapshots already "
                      f"in {out_dir}; building {len(pending)}.")

            networks_iter = construct_network_time_iter(
                [start_dates[i] for i in pending], [end_dates[i] for i in pending],
                dataset=network, type_dataset=type_dataset,
                echo=echo, days_echo=days_echo, transactions=transactions
            )
            # Manual counter, NOT enumerate(): CPython's enumerate (and zip) reuse
            # their cached result tuple, which keeps a strong ref to the PREVIOUS
            # (G, labels) until AFTER the generator has built the next snapshot —
            # holding two graphs resident and defeating the del/gc below.
            pos = -1
            for G, labels in networks_iter:
                pos += 1
                # The SNAPSHOT index, not the loop position: filenames encode it and
                # _load_snapshot reads range(T) contiguously, so it must not shift.
                i = pending[pos]
                print(f"Processing dynamic network snapshot {i+1}/{n_snapshots}...")
                df = process_graph(G)

                stem_features = stem(i, "features")
                stem_labels = stem(i, "labels")

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

            # static_stem: HI-Small -> HI-Small_o1 for the first-order baseline, so it never
            # overwrites the order-2 tables (dynamic runs are namespaced by run_tag instead).
            stem_static = static_stem(type_dataset)
            out_path_features = save_table(df, f"results/features/{stem_static}_static_features")
            print(f"Results saved to {out_path_features}")
            out_path_labels = save_table(labels.reset_index(), f"results/features/{stem_static}_static_labels")
            print(f"Labels saved to {out_path_labels}")

    else:
        raise ValueError(f"Feature extraction not supported for dataset: {network}")

