# GraphSAGE GNN baseline, as in GARG-AML (Deprez et al.): each account's ONLY input feature is
# its log-degree, so the model learns from topology alone, and a 1- and a 2-layer model are
# trained per cell to show what the second hop adds (two layers = the second-order
# neighbourhood SPaRTA's frames are built from; a third would look beyond it).
#
# Scored on the SAME (node, anchor) rows and temporal split as experiment_baseline.py /
# experiment_LSTM.py (rows come from the same window store), so it slots into the per-cell
# comparison. The GNN sees one snapshot graph per row — the anchor snapshot for nowcast, the
# snapshot before it for forecast — i.e. the GNN counterpart of the single-snapshot (local)
# view. Each band (train / val / test) is the disjoint union of its anchors' snapshot graphs:
# undirected, unweighted, parallel transactions merged, self-loops dropped, no Louvain pruning
# (GARG-AML Table 9). In echo mode the graph covers the echo window [end - days_echo, end), the
# same transactions the features saw, without the decay weights.
#
# Fixed hyperparameters (config block `gnn`, GARG-AML's values), no Optuna: mini-batches from
# NeighborLoader with fan-out (25, 10), early stopping on val-band AUC-PR, and exact full-graph
# inference on the (small) val and test bands. Minimal by design, after the PyG
# introduction + NeighborLoader tutorials.
import os
import sys
DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import degree, to_undirected

from src.utils.setup import (load_config, resolve_dataset, resolve_timing, resolve_sequence,
                             run_tag, resolve_ts_cache_dir, resolve_window_store_dir,
                             require_prebuilt_windows)
from src.data.network_data_loader import load_transactions
from src.data.sequence_data import build_sequence_windows, temporal_split
from src.data.utils.dates import define_dates
from src.methods.models import GraphSAGE
from src.methods import evaluation

SEED = 1997


def band_graph(src, dst, ts, start, end, graph_snap, row_codes, y_rows):
    """Disjoint union of the snapshot graphs one band's rows are scored on.

    src/dst/ts: account codes + timestamps of all transactions, sorted by ts. A row is
    (graph snapshot, account); its node is that account in that snapshot's graph, so the same
    account in two snapshots is two nodes and NeighborLoader never samples across snapshots.
    A row whose account has no transaction in the window (forecast only) is an isolated node,
    i.e. degree 0 — the GNN analogue of the baselines' all-zero frame.
    Returns (Data with x = log(1 + degree), y; seed node per row, in row order)."""
    snaps = np.unique(graph_snap)
    n_acc = int(max(src.max(), dst.max(), row_codes.max())) + 1
    e_src, e_dst = [], []
    for slot, s in enumerate(snaps):        # key = slot * n_acc + account code
        lo, hi = np.searchsorted(ts, [start[s], end[s]])    # transactions in [start, end)
        e_src.append(slot * n_acc + src[lo:hi])
        e_dst.append(slot * n_acc + dst[lo:hi])
    e_src, e_dst = np.concatenate(e_src), np.concatenate(e_dst)
    seed_keys = np.searchsorted(snaps, graph_snap) * n_acc + row_codes
    keys, node = np.unique(np.concatenate([e_src, e_dst, seed_keys]), return_inverse=True)
    n, E = len(keys), len(e_src)

    # Symmetrise + coalesce: undirected, and parallel transactions merge into one edge.
    edge_index = to_undirected(torch.from_numpy(np.stack([node[:E], node[E:2 * E]])), num_nodes=n)
    seeds = torch.from_numpy(node[2 * E:])
    x = torch.log1p(degree(edge_index[0], num_nodes=n)).unsqueeze(-1)
    y = torch.zeros(n)
    y[seeds] = torch.as_tensor(y_rows, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, y=y), seeds


@torch.no_grad()
def predict(model, data, seeds, device):
    """P(laundering) for the seed nodes: one exact forward pass over the whole band graph."""
    model.eval()
    out = model(data.x.to(device), data.edge_index.to(device))
    return torch.sigmoid(out[seeds.to(device)]).cpu().numpy()


def train_graphsage(num_layers, train, train_seeds, val, val_seeds, cfg, device):
    """Mini-batch training with NeighborLoader (fan-out cfg['num_neighbors'][:num_layers]),
    BCE with pos_weight = neg/pos of the train rows, early stopping on val-band AUC-PR with the
    best epoch restored. Returns (model, best val AUC-PR, best epoch)."""
    torch.manual_seed(SEED)
    model = GraphSAGE(train.num_node_features, cfg["hidden"], num_layers, cfg["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    y_tr = train.y[train_seeds]
    pos_weight = ((y_tr == 0).sum() / (y_tr == 1).sum().clamp(min=1)).reshape(1).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader = NeighborLoader(train, input_nodes=train_seeds, shuffle=True,
                            num_neighbors=cfg["num_neighbors"][:num_layers],
                            batch_size=cfg["batch_size"], num_workers=cfg.get("num_workers", 0))
    y_val = val.y[val_seeds].numpy()

    best_ap, best_epoch, best_state, wait = -1.0, -1, None, 0
    for epoch in range(cfg["epochs"]):
        model.train()
        total = 0.0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch.x, batch.edge_index)[:batch.batch_size]   # seeds come first
            loss = criterion(out, batch.y[:batch.batch_size])
            loss.backward()
            opt.step()
            total += float(loss) * batch.batch_size
        ap = average_precision_score(y_val, predict(model, val, val_seeds, device))
        print(f"  epoch {epoch:3d} | loss {total / len(train_seeds):.4f} | val AUC-PR {ap:.4f}",
              flush=True)
        if ap > best_ap:
            best_ap, best_epoch, wait = ap, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model, best_ap, best_epoch


if __name__ == "__main__":
    np.random.seed(SEED)

    data_config = load_config("config/data/config.yaml")
    method_config = load_config("config/methods/config.yaml")
    dataset = resolve_dataset(data_config)
    type_dataset = data_config[dataset]["type_dataset"]

    # Same SPARTA_* overrides and stem as experiment_baseline.py, so one array task = one cell.
    net_cfg = resolve_timing(data_config, dataset)
    tag = run_tag(net_cfg)
    seq_cfg = resolve_sequence(data_config, dataset)
    K, task = seq_cfg["K"], seq_cfg["task"]
    cfg = method_config[dataset]["gnn"]
    stem = f"{type_dataset}_{tag}_K{K}_{task}"
    # CUDA or CPU, never MPS: every NeighborLoader batch has a different shape, and MPS caches
    # one compiled graph per shape — on AMLSim it grew to ~47 GB within three epochs, pushed
    # macOS into swap until the disk filled, and died with "MPS backend out of memory".
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- The baselines' rows: same windows, same split, same no-history filter. Only the row
    # identities are used, so the store's 90-feature block stays memory-mapped (never read).
    values, idx, y, anchors, nodes = build_sequence_windows(
        dataset, type_dataset, net_cfg["echo"], K, task,
        features_dir=os.path.join(cfg.get("data_directory", "results/features"), tag),
        cache_dir=resolve_ts_cache_dir(tag), store_dir=resolve_window_store_dir(tag),
        require_prebuilt=require_prebuilt_windows(), mmap_values=True,
        time_step=net_cfg["time_step"], time_width=net_cfg["time_width"],
        time_type=net_cfg["time_type"])
    bands = temporal_split(anchors, y, n_test_anchors=seq_cfg["n_test_anchors"],
                           n_val_anchors=seq_cfg["n_val_anchors"], dataset_label=type_dataset)
    bands = [b[(idx[b] >= 0).any(1)] for b in bands]
    del values, idx

    # --- Transactions on the feature extraction's snapshot grid (measure_calculation.py).
    tx = load_transactions(dataset, type_dataset)
    start, end = define_dates(tx["timestamp"], net_cfg["time_step"], net_cfg["time_width"],
                              net_cfg["time_type"])
    start, end = np.asarray(start, dtype="datetime64[ns]"), np.asarray(end, dtype="datetime64[ns]")
    if net_cfg["echo"]:     # echo ignores the width: window = [end - days_echo, end)
        start = end - np.timedelta64(int(net_cfg["days_echo"]) * 24, "h")
    if anchors.max() >= len(end):
        raise SystemExit(f"Window rows reach snapshot {anchors.max()} but the transactions give "
                         f"only {len(end)} snapshots — the store is from a different grid.")
    codes, accounts = pd.factorize(pd.concat([tx["from_account"], tx["to_account"]],
                                             ignore_index=True))
    order = np.argsort(tx["timestamp"].to_numpy(), kind="stable")
    ts = tx["timestamp"].to_numpy()[order]
    src, dst = codes[:len(tx)][order], codes[len(tx):][order]
    acc_index = pd.Index(np.asarray(accounts).astype(str))
    del tx, codes, order

    graphs = []
    for name, rows in zip(("train", "val", "test"), bands):
        row_codes = acc_index.get_indexer(np.asarray(nodes[rows]).astype(str))
        if (row_codes < 0).any():
            raise SystemExit(f"{(row_codes < 0).sum()} {name} rows are accounts that never "
                             f"transact — node ids do not match between features and transactions.")
        graph_snap = anchors[rows] - (1 if task == "forecast" else 0)
        data, seeds = band_graph(src, dst, ts, start, end, graph_snap, row_codes, y[rows])
        graphs.append((data, seeds))
        print(f"{name}: {len(rows):,} rows ({int(y[rows].sum()):,} positive) on "
              f"{len(np.unique(graph_snap))} snapshot graphs | {data.num_nodes:,} nodes, "
              f"{data.num_edges:,} directed edges", flush=True)
    (train, train_seeds), (val, val_seeds), (test, test_seeds) = graphs

    # Standardise log-degree with train-band statistics.
    mu, sd = train.x.mean(), train.x.std().clamp(min=1e-8)
    for data, _ in graphs:
        data.x = (data.x - mu) / sd

    os.makedirs("results/experiments", exist_ok=True)
    os.makedirs("results/tuning", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    scores, params = {}, {}
    for num_layers in cfg["layers"]:
        name = f"GraphSAGE-{num_layers}L"
        print(f"{name} | task={task} K={K} tag={tag} | device={device}", flush=True)
        model, best_ap, best_epoch = train_graphsage(num_layers, train, train_seeds, val,
                                                     val_seeds, cfg, device)
        scores[name] = predict(model, test, test_seeds, device)
        params[name] = {"num_layers": num_layers, "best_epoch": best_epoch,
                        "best_value_AUCPR": best_ap}
        torch.save(model.state_dict(), f"results/models/{name}_{stem}.pt")

    test_rows = bands[2]
    y_test = y[test_rows]
    metrics_path = f"results/experiments/{stem}_gnn.txt"
    evaluation.write_metrics(metrics_path, scores, y_test)
    evaluation.save_predictions(f"{evaluation.PRED_DIR}/{stem}_gnn.parquet", scores, y_test,
                                anchors[test_rows], np.asarray(nodes)[test_rows])
    with open(f"results/tuning/gnn_{stem}_best_params.json", "w") as f:
        json.dump({"dataset": type_dataset, "run_tag": tag, "task": task, "K": K,
                   "echo": net_cfg["echo"], "config": cfg, "models": params}, f, indent=2)
    print(f"Wrote metrics -> {metrics_path}")
