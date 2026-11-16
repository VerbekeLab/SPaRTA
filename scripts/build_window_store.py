# Stage 1.5: build the prebuilt per-TAG window store for the active timing combo.
#
# Reads results/features/<tag>/ once and writes the stacked snapshot block to the store
# directory (src/data/window_store.py). Run this ONCE PER TAG on a CPU node; every training
# job for that tag's 6 (K x task) combos then derives its windows from it in seconds instead of
# re-reading and re-assembling the feature tables (~40 min each, previously paid up to six
# times per tag — twice on GPU nodes).
#
# Config comes from the same SPARTA_* env the training scripts read (slurm/sweep_common.sh
# sets it), so the tag this writes is exactly the tag they look for.
#   python -u scripts/build_window_store.py
import os
import sys
DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import time

from src.data.network_data_loader import load_transactions
from src.data.utils.dates import define_dates
from src.data import window_store
from src.utils.setup import (load_config, resolve_dataset, resolve_timing, run_tag,
                             resolve_window_store_dir)

if __name__ == "__main__":
    data_config = load_config("config/data/config.yaml")
    method_config = load_config("config/methods/config.yaml")

    dataset = resolve_dataset(data_config)
    type_dataset = data_config[dataset]["type_dataset"]
    net_cfg = resolve_timing(data_config, dataset)
    echo = net_cfg["echo"]
    suffix = "_echo" if echo else ""
    tag = run_tag(net_cfg)

    # The features dir is namespaced by tag exactly as the training scripts namespace it.
    features_dir = os.path.join(
        method_config[dataset]["baselines"].get("data_directory", "results/features"), tag)
    store_dir = resolve_window_store_dir(tag)
    if store_dir is None:
        raise SystemExit("SPARTA_WINDOW_STORE=off — nothing to build. Unset it to build a store.")

    # T is derived from the snapshot grid, the one thing that needs the transaction file. Doing
    # it HERE (once per tag, on a CPU node) is why the training jobs never have to.
    transactions = load_transactions(dataset, type_dataset)
    _, end_dates = define_dates(transactions["timestamp"], net_cfg["time_step"],
                               net_cfg["time_width"], net_cfg["time_type"])
    T = len(end_dates)
    del transactions

    if window_store.store_exists(store_dir, type_dataset, suffix):
        print(f"Store already complete at {store_dir} ({type_dataset}{suffix}) — nothing to do. "
              f"Delete its files to rebuild.")
        raise SystemExit(0)

    print(f"Building window store | dataset={type_dataset} tag={tag} echo={echo} T={T}\n"
          f"  features_dir = {features_dir}\n  store_dir    = {store_dir}", flush=True)
    t0 = time.perf_counter()
    meta = window_store.build_store(store_dir, type_dataset, suffix, features_dir, T)
    dt = time.perf_counter() - t0

    print(f"\nWrote store in {dt / 60:.1f} min: {meta['n_rows']:,} (snapshot, node) rows x "
          f"{meta['n_features']} features, {meta['n_nodes']:,} unique nodes over T={meta['T']} "
          f"snapshots ({meta['n_rows'] * meta['n_features'] * 4 / 1e9:.2f} GB float32)")
    print(f"Every (K, task) combo of tag {tag} now derives its windows from this in seconds.")
