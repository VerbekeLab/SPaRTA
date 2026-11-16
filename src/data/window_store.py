# Prebuilt per-TAG window store: the Stage-1.5 artefact that sits between the per-snapshot
# feature tables (results/features/<tag>/) and the K-step window tensors the experiments train
# on.
#
# WHY THIS EXISTS. The deduplicated `values` block build_sequence_dataset produces — each
# present (snapshot, node) feature row, once — depends only on the TAG. K and the task select
# which rows each window points at; they do not change the rows. The sweep has 6 (K x task)
# combos per tag, so the old flow re-read every feature file and re-assembled the same block
# six times, ~40 min each, and stored six near-identical copies. This module builds it ONCE per
# tag on a CPU node; a training job then derives its own (K, task) windows from it by index
# arithmetic in seconds, touching no feature files and no transaction file.
#
# LAYOUT (one set of files per tag directory, prefixed by {type_dataset}{suffix}):
#   _values.npy         [M, 90] float32  every snapshot's 90 feature columns, stacked
#                                        snapshot-major; within a snapshot, rows keep the
#                                        feature FILE's order (see ORDERING below)
#   _offsets.npy        [T+1]   int64    snapshot i occupies values[offsets[i]:offsets[i+1]]
#   _node_codes.npy     [M]     int32    each row's node, as an index into _node_universe
#   _node_universe.npy  [U]     <U       sorted unique node ids
#   _labels.npy         [M]     float32  is_laundering per (snapshot, node) cell
#   _meta.json                           T, M, U, the 90 column names, format version
#
# ORDERING. Rows stay in the feature file's order rather than sorted, so the derived
# (idx, y, anchors, nodes) come out in exactly the order the old row-by-row builder emitted —
# same samples, same positions, so temporal_split bands and every downstream array are
# unchanged. Sorting would have been marginally faster to look up and would have silently
# permuted every band.
import json
import os

import numpy as np
import pandas as pd

from src.data.sequence_data import _load_snapshot, _snapshot_index, N_FEATURES
from src.methods.measure_calculation import keys_to_include

FORMAT_VERSION = 1

_PARTS = ("values", "offsets", "node_codes", "node_universe", "labels", "meta")


def store_paths(store_dir, type_dataset, suffix):
    """Absolute path of every file in the store, keyed by part name."""
    stem = os.path.join(store_dir, f"{type_dataset}{suffix}")
    return {p: f"{stem}_{p}.json" if p == "meta" else f"{stem}_{p}.npy" for p in _PARTS}


def store_exists(store_dir, type_dataset, suffix):
    """True only when EVERY part is present — a half-written store must not look usable."""
    if not store_dir:
        return False
    return all(os.path.exists(p) for p in store_paths(store_dir, type_dataset, suffix).values())


class WindowStore:
    """In-memory handle on a store. `values` is a memmap when mmap=True (the derive step never
    touches it — only offsets/node_codes/labels — so a job that keeps the compact form never
    pages the 2.6 GB block in at all)."""

    def __init__(self, values, offsets, node_codes, node_universe, labels, meta):
        self.values = values
        self.offsets = offsets
        self.node_codes = node_codes
        self.node_universe = node_universe
        self.labels = labels
        self.meta = meta
        self.T = int(meta["T"])

    def __repr__(self):
        return (f"WindowStore(T={self.T}, rows={len(self.node_codes):,}, "
                f"nodes={len(self.node_universe):,}, features={self.meta['n_features']})")


def build_store(store_dir, type_dataset, suffix, features_dir, T, verbose=True):
    """Read the T per-snapshot feature/label tables once and write the store. Peak memory is
    one snapshot DataFrame plus two copies of the stacked block (the per-snapshot list and the
    concatenation), which is ~2 x 2.6 GB at AMLWorld scale — an order of magnitude under the
    dense [N, K, 90] tensor the old in-job build materialised.

    Written to a per-process temp dir and os.replace'd into place part by part, so a reader
    never sees a partial store (mirrors the atomic .npz write it replaces)."""
    os.makedirs(store_dir, exist_ok=True)
    blocks, label_parts, node_parts, counts = [], [], [], []
    for i in range(T):
        snap = _load_snapshot(i, type_dataset, suffix, features_dir)
        blocks.append(snap[keys_to_include].to_numpy(dtype=np.float32))
        label_parts.append(snap["is_laundering"].to_numpy(dtype=np.float32))
        node_parts.append(snap.index.to_numpy().astype(str))
        counts.append(len(snap))
        if verbose:
            print(f"  snapshot {i + 1}/{T}: {len(snap):,} nodes", flush=True)
        del snap

    # Same scrub the old builder applied to its dedup rows (sequence_data.py), so the derived
    # X is identical: once per unique row, not once per window copy.
    values = np.nan_to_num(np.concatenate(blocks), copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    del blocks
    labels = np.concatenate(label_parts)
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    # One global universe so a node's code means the same thing in every snapshot; sorted, so
    # the codes are reproducible across rebuilds.
    node_universe, inverse = np.unique(np.concatenate(node_parts), return_inverse=True)
    del node_parts
    node_codes = inverse.astype(np.int32)

    meta = {"format_version": FORMAT_VERSION, "type_dataset": type_dataset, "suffix": suffix,
            "T": int(T), "n_rows": int(values.shape[0]), "n_nodes": int(len(node_universe)),
            "n_features": int(values.shape[1]), "feature_columns": list(keys_to_include),
            "features_dir": features_dir, "snapshot_counts": [int(c) for c in counts]}

    paths = store_paths(store_dir, type_dataset, suffix)
    arrays = {"values": values, "offsets": offsets, "node_codes": node_codes,
              "node_universe": node_universe, "labels": labels}
    for part, arr in arrays.items():
        tmp = f"{paths[part]}.{os.getpid()}.tmp"
        # Passing a file object stops np.save from appending its own ".npy" to the temp name.
        with open(tmp, "wb") as fh:
            np.save(fh, arr, allow_pickle=False)
        os.replace(tmp, paths[part])
    tmp = f"{paths['meta']}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(meta, fh, indent=2)
    os.replace(tmp, paths["meta"])       # meta LAST: store_exists() only passes once it lands
    return meta


def load_window_store(store_dir, type_dataset, suffix, mmap=True):
    """Open a store and validate it against the current feature schema. mmap=True leaves
    `values` on disk until something indexes it."""
    paths = store_paths(store_dir, type_dataset, suffix)
    with open(paths["meta"]) as fh:
        meta = json.load(fh)
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"{paths['meta']} is format version {meta.get('format_version')}, this code reads "
            f"{FORMAT_VERSION}. Rebuild with scripts/build_window_store.py.")
    if meta.get("feature_columns") != list(keys_to_include):
        raise ValueError(
            f"{paths['meta']} was built from a different feature schema "
            f"({meta.get('n_features')} columns) than measure_calculation.keys_to_include "
            f"({N_FEATURES}). Rebuild the store with scripts/build_window_store.py.")
    return WindowStore(
        values=np.load(paths["values"], mmap_mode="r" if mmap else None),
        offsets=np.load(paths["offsets"]),
        node_codes=np.load(paths["node_codes"]),
        node_universe=np.load(paths["node_universe"]),
        labels=np.load(paths["labels"]),
        meta=meta)


def windows_from_store(store, K, task):
    """Derive (values, idx, y, anchors, nodes) for one (K, task) from a store.

    Returns the same DEDUPLICATED form build_sequence_dataset's cache returns: `values` holds
    each (snapshot, node) row once and idx[sample, step] points into it (-1 = node absent at
    that step, i.e. mask False). Cost is O(T*K) vectorized Index.get_indexer calls — a few
    hundred — instead of the old O(N*K) Python iterations with a pandas .loc per cell.

    Membership and row lookup fall out of the SAME call: get_indexer returns -1 for labels it
    cannot find, which is exactly the mask semantics."""
    T = store.T
    if T < K:
        raise ValueError(f"store has T={T} snapshots but K={K}; need T >= K. Lower K.")
    if task not in ("nowcast", "forecast"):
        raise ValueError(f"task must be 'nowcast' or 'forecast', got {task!r}")

    off = store.offsets
    # One hashtable per snapshot over its node CODES (ints, not strings — cheaper to hash and
    # the universe is shared, so a code means the same node in every snapshot).
    indexes = [pd.Index(store.node_codes[off[i]:off[i + 1]]) for i in range(T)]

    first_anchor = K - 1 if task == "nowcast" else K
    idx_parts, y_parts, anchor_parts, code_parts = [], [], [], []
    for a in range(first_anchor, T):
        lo, hi = int(off[a]), int(off[a + 1])
        anchor_codes = store.node_codes[lo:hi]
        win = np.full((hi - lo, K), -1, dtype=np.int32)
        for step in range(K):
            si = _snapshot_index(a, step, K, task)
            pos = indexes[si].get_indexer(anchor_codes)
            hit = pos >= 0
            win[hit, step] = off[si] + pos[hit]
        idx_parts.append(win)
        y_parts.append(store.labels[lo:hi])
        anchor_parts.append(np.full(hi - lo, a, dtype=int))
        code_parts.append(anchor_codes)

    idx = np.concatenate(idx_parts)
    y = np.concatenate(y_parts).astype(np.float32)
    anchors = np.concatenate(anchor_parts)
    # object dtype to match what the .npz cache path returns (callers index it with node ids).
    nodes = store.node_universe[np.concatenate(code_parts)].astype(object)
    return store.values, idx, y, anchors, nodes
