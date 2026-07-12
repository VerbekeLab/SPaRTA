# Sequence dataset for the time-series experiments (LSTM / Transformer / XGBoost).
# Builds K-step windows over the per-snapshot SPaRTA feature CSVs written by
# measure_calculation.py (time_dynamic=True). A sample is one (node, anchor) pair;
# multiple anchors per node slide across the T snapshots. Masking handles internal
# padding (a node present at steps t and t+2 but absent at t+1). Minimalist style —
# mirrors notebooks/test_timeseries.ipynb; reuses repo building blocks unchanged.
import os

import numpy as np
import pandas as pd
import torch

from src.data.utils.dates import define_dates
from src.data.network_data_loader import load_transactions
from src.methods.measure_calculation import keys_to_include  # 90 feature columns, canonical order
from src.utils.feature_io import find_table, load_table

SEED = 1997
N_FEATURES = len(keys_to_include)  # 90


def _snapshot_index(anchor, step, K, task):
    """Snapshot index that window position `step` (0..K-1) maps to for an anchor.
    nowcast window = [a-K+1 .. a]; forecast window = [a-K .. a-1]."""
    base = anchor - (K - 1) if task == "nowcast" else anchor - K
    return base + step


def _load_snapshot(i, type_dataset, suffix, features_dir):
    """Read snapshot i's features + labels (Parquet preferred, legacy CSV fallback),
    validate 90 cols, return a node-indexed DataFrame of the 90 features plus an
    `is_laundering` column."""
    feat_stem = os.path.join(features_dir, f"{type_dataset}_dynamic_{i}_features{suffix}")
    label_stem = os.path.join(features_dir, f"{type_dataset}_dynamic_{i}_labels{suffix}")
    feat_path = find_table(feat_stem)
    if feat_path is None:
        raise ValueError(
            f"Missing features file {feat_stem}.parquet (or legacy .csv). Regenerate "
            f"the 90-column dynamic features with src/methods/measure_calculation.py "
            f"(time_dynamic=True)."
        )
    if find_table(label_stem) is None:
        raise ValueError(
            f"Missing labels file {label_stem}.parquet (or legacy .csv). Regenerate "
            f"with src/methods/measure_calculation.py (time_dynamic=True)."
        )

    features = load_table(feat_stem, dtype={"node": str})
    # Parquet keeps the node column's written dtype (possibly int); normalise to
    # str to match the CSV parse hint above and the labels' Account cast below.
    features["node"] = features["node"].astype(str)
    missing = [c for c in keys_to_include if c not in features.columns]
    if missing:
        raise ValueError(
            f"{feat_path} carries {len(features.columns)} columns and is missing "
            f"{len(missing)} of the {N_FEATURES} required feature columns "
            f"(first missing: {missing[0]}). This is the legacy 19-column variant; "
            f"regenerate with src/methods/measure_calculation.py (time_dynamic=True)."
        )

    labels = load_table(label_stem)
    # Labels file has exactly 2 columns: Account + the label (old files say
    # "Is Laundering", regenerated ones "is_laundering"). Positional rename handles both.
    labels = labels.rename(columns={labels.columns[0]: "Account", labels.columns[1]: "is_laundering"})
    labels["Account"] = labels["Account"].astype(str)

    merged = features.merge(
        labels[["Account", "is_laundering"]], left_on="node", right_on="Account", how="left"
    )
    merged["is_laundering"] = merged["is_laundering"].fillna(0).astype(float)
    return merged.set_index("node")


def build_sequence_dataset(dataset, type_dataset, echo, K, task,
                           features_dir="results/features", cache_dir="results/timeseries",
                           use_cache=True, transactions=None,
                           time_step=1, time_width=1, time_type="days"):
    """Build the (X, mask, y, anchors, nodes) windowed sequence dataset.

    X      float32 [N, K, F]  (F=90; padded/absent steps = 0)
    mask   bool    [N, K]     (True where the node is present at that step)
    y      float32 [N]        (is_laundering of the node at its anchor snapshot)
    anchors int    [N]        (anchor snapshot index per sample)
    nodes  object  [N]        (node id per sample)

    A sample = (node, anchor): for each valid anchor a (nowcast a in [K-1 .. T-1],
    forecast a in [K .. T-1]) and each node present at snapshot a, emit one K-step
    window mapped to snapshots [a-K+1 .. a] (nowcast) / [a-K .. a-1] (forecast).
    """
    if task not in ("nowcast", "forecast"):
        raise ValueError(f"task must be 'nowcast' or 'forecast', got {task!r}")

    suffix = "_echo" if echo else ""

    # Derive T from the dataset's daily snapshot grid.
    if transactions is None:
        transactions = load_transactions(dataset, type_dataset)
    _, end_dates = define_dates(transactions["timestamp"], time_step, time_width, time_type)
    T = len(end_dates)

    if T < K:
        raise ValueError(
            f"Dataset {dataset}/{type_dataset} has T={T} snapshots but K={K}; "
            f"need T >= K. Skip this dataset or lower K."
        )

    # Optional cache keyed by (type_dataset, echo, K, task, T) — T is included so a
    # feature regeneration that changes the snapshot count invalidates the stale cache.
    cache_path = os.path.join(cache_dir, f"{type_dataset}{suffix}_K{K}_T{T}_{task}.npz")
    if use_cache and os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        return d["X"], d["mask"], d["y"], d["anchors"], d["nodes"]

    # Load + validate every snapshot once.
    snapshots = [_load_snapshot(i, type_dataset, suffix, features_dir) for i in range(T)]

    first_anchor = K - 1 if task == "nowcast" else K
    valid_anchors = list(range(first_anchor, T))

    X_rows, mask_rows, y_rows, anchor_rows, node_rows = [], [], [], [], []
    for a in valid_anchors:
        anchor_df = snapshots[a]
        for node in anchor_df.index:
            win = np.zeros((K, N_FEATURES), dtype=np.float32)
            m = np.zeros(K, dtype=bool)
            for step in range(K):
                si = _snapshot_index(a, step, K, task)
                snap = snapshots[si]
                if node in snap.index:
                    win[step] = snap.loc[node, keys_to_include].to_numpy(dtype=np.float32)
                    m[step] = True
            X_rows.append(win)
            mask_rows.append(m)
            y_rows.append(float(anchor_df.loc[node, "is_laundering"]))
            anchor_rows.append(a)
            node_rows.append(node)

    # Free each source as soon as it is converted: X_rows holds a second full copy of X's
    # data (millions of small arrays at AMLWorld scale) and snapshots several GB of frames,
    # so dropping them here — plus the in-place nan scrub — keeps the build's peak at ~2x
    # the final X (list + array, briefly) instead of ~3x.
    del snapshots
    X = np.asarray(X_rows, dtype=np.float32).reshape(-1, K, N_FEATURES)
    del X_rows
    mask = np.asarray(mask_rows, dtype=bool).reshape(-1, K)
    del mask_rows
    y = np.asarray(y_rows, dtype=np.float32)
    anchors = np.asarray(anchor_rows, dtype=int)
    nodes = np.asarray(node_rows, dtype=object)

    X = np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        # Compressed: X is mostly zeros (padding + absent nodes), so the plain savez cache was
        # several times larger on disk for no benefit. Write to a per-process temp file then
        # os.replace() into place: the fan-out baseline tasks (one per model, e.g. XGBoost and
        # the MLP) may rebuild the same combo concurrently, and an atomic replace guarantees a
        # reader sees either no file or a complete one — never a half-written .npz. Passing a
        # file object stops np.savez_compressed from appending its own ".npz" to the temp name.
        tmp_path = f"{cache_path}.{os.getpid()}.tmp"
        with open(tmp_path, "wb") as fh:
            np.savez_compressed(fh, X=X, mask=mask, y=y, anchors=anchors, nodes=nodes)
        os.replace(tmp_path, cache_path)

    return X, mask, y, anchors, nodes


class SequenceDataset(torch.utils.data.Dataset):
    """Wraps (X, mask, y); __getitem__ -> (x[K,F] f32, m[K] bool, y f32)."""

    def __init__(self, X, mask, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.mask = torch.as_tensor(mask, dtype=torch.bool)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.mask[i], self.y[i]


def fit_scaler(X, mask, nodes, anchors, K, task, train_idx):
    """Mask-aware (mu, sd) over the unique (node, snapshot-index) cells present in the
    train windows only — so a snapshot reused across many windows isn't over-weighted.
    The snapshot index is recomputed per step from anchors/K/task (the builder's formula).
    Returns (mu[F], sd[F]) with sd floored at 1e-8."""
    seen = set()
    rows = []
    for i in train_idx:
        a = int(anchors[i])
        node = nodes[i]
        for step in range(K):
            if not mask[i, step]:
                continue
            si = _snapshot_index(a, step, K, task)
            cell = (node, si)
            if cell in seen:
                continue
            seen.add(cell)
            rows.append(X[i, step])
    flat = np.asarray(rows, dtype=np.float64).reshape(-1, X.shape[-1])
    mu = flat.mean(0)
    sd = flat.std(0)
    sd = np.where(sd < 1e-8, 1e-8, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_scaler(X, mask, mu, sd):
    """(X - mu) / sd, then re-zero padded (~mask) steps so mask semantics hold.
    Allocates exactly ONE full-size array (in-place divide, no-copy cast): X is tens of
    GB at AMLWorld scale, so every avoided temporary is a real slice of the job's memory
    cap. Same float32 elementwise ops as the old chained expression — identical output."""
    Xs = X - mu
    Xs /= sd
    Xs[~mask] = 0
    return Xs.astype(np.float32, copy=False)


def temporal_split(anchors, y, n_test_anchors=1, n_val_anchors=1, dataset_label=None):
    """Temporal holdout by anchor index: sort the unique valid anchors ascending,
    last n_test_anchors -> test, the n_val_anchors before -> val, the rest -> train.
    Each sample is mapped to its band by its anchor. Enforces the §3 guards:
    enough unique anchors, and >=1 positive in each of the val and test bands."""
    unique = np.array(sorted(set(int(a) for a in anchors)))
    if len(unique) < n_test_anchors + n_val_anchors + 1:
        raise ValueError(
            f"Dataset {dataset_label}: only {len(unique)} unique valid anchors "
            f"(T-derived), need >= {n_test_anchors + n_val_anchors + 1} "
            f"(n_test_anchors={n_test_anchors} + n_val_anchors={n_val_anchors} + 1 train)."
        )

    test_anchors = set(unique[-n_test_anchors:].tolist())
    val_anchors = set(unique[-(n_test_anchors + n_val_anchors):-n_test_anchors].tolist())

    test_idx, val_idx, train_idx = [], [], []
    for i, a in enumerate(anchors):
        a = int(a)
        if a in test_anchors:
            test_idx.append(i)
        elif a in val_anchors:
            val_idx.append(i)
        else:
            train_idx.append(i)
    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    if y[val_idx].sum() < 1:
        raise ValueError(
            f"Dataset {dataset_label}: validation band (anchors {sorted(val_anchors)}) "
            f"has no positives; AUC-PR is undefined. T-derived anchors = {unique.tolist()}."
        )
    if y[test_idx].sum() < 1:
        raise ValueError(
            f"Dataset {dataset_label}: test band (anchors {sorted(test_anchors)}) "
            f"has no positives; AUC-PR is undefined. T-derived anchors = {unique.tolist()}."
        )
    return train_idx, val_idx, test_idx
