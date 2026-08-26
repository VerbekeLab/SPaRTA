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


def _expand_windows(values, idx):
    """Rebuild the dense (X, mask) pair from the deduplicated cache layout:
    `values` [M, F] holds each present (snapshot, node) feature row once and
    `idx` [N, K] points into it (-1 = node absent at that step)."""
    mask = idx >= 0
    if len(values) == 0:
        return np.zeros((*idx.shape, N_FEATURES), dtype=np.float32), mask
    X = values[np.maximum(idx, 0)]
    X[~mask] = 0.0
    return X, mask


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


def build_sequence_windows(dataset, type_dataset, echo, K, task,
                           features_dir="results/features", cache_dir="results/timeseries",
                           use_cache=True, transactions=None,
                           time_step=1, time_width=1, time_type="days",
                           store_dir=None, require_prebuilt=False):
    """Windowed sequence dataset in its COMPACT form — the one callers should prefer.

    values  float32 [M, F]  each present (snapshot, node) feature row, ONCE (F=90)
    idx     int32   [N, K]  idx[sample, step] -> row of `values`; -1 = node absent that step
    y       float32 [N]     is_laundering of the node at its anchor snapshot
    anchors int     [N]     anchor snapshot index per sample
    nodes   object  [N]     node id per sample

    `mask` is just ``idx >= 0``, and the dense window tensor is ``values[idx]`` with the -1
    positions zeroed. Adjacent anchors share K-1 snapshots, so that dense form is ~K times
    LARGER than `values` (2.6 GB -> 18 GB at AMLWorld K=7, ~31 GB at K=17) — which is why the
    experiments keep this form and gather per batch instead. Use build_sequence_dataset() only
    when a consumer genuinely needs the whole dense tensor at once.

    Sources, in order: the prebuilt per-tag window store (`store_dir`, seconds — see
    src/data/window_store.py), this combo's .npz cache, then a full re-assembly from the
    per-snapshot feature files (~40 min at AMLWorld scale). `require_prebuilt` turns the last
    two into a hard failure, which is what a GPU job wants: paying that re-assembly with an
    accelerator allocated and idle is the failure mode Stage 1.5 exists to remove.

    A sample = (node, anchor): for each valid anchor a (nowcast a in [K-1 .. T-1],
    forecast a in [K .. T-1]) and each node present at snapshot a, emit one K-step
    window mapped to snapshots [a-K+1 .. a] (nowcast) / [a-K .. a-1] (forecast).
    """
    if task not in ("nowcast", "forecast"):
        raise ValueError(f"task must be 'nowcast' or 'forecast', got {task!r}")

    suffix = "_echo" if echo else ""

    # --- Fast path: the prebuilt per-tag store. Checked BEFORE the transaction load below,
    # because T comes from its meta.json — so a job with a store never parses the raw
    # transaction file (hundreds of MB) merely to count snapshots. Imported here rather than at
    # module scope: window_store reads _load_snapshot/_snapshot_index from THIS module.
    from src.data import window_store

    if store_dir is not None and window_store.store_exists(store_dir, type_dataset, suffix):
        # mmap=False: callers gather from `values` many times (once per batch), so it belongs
        # in RAM rather than being paged out of a memmap on every touch.
        store = window_store.load_window_store(store_dir, type_dataset, suffix, mmap=False)
        return window_store.windows_from_store(store, K, task)

    if require_prebuilt:
        raise SystemExit(
            f"No prebuilt window store for {type_dataset}{suffix} under {store_dir!r}, and "
            f"require_prebuilt is set (SPARTA_REQUIRE_WINDOWS).\n"
            f"Build it first:  sbatch slurm/build_windows.slurm <Dataset>\n"
            f"Or unset SPARTA_REQUIRE_WINDOWS to re-assemble from the feature files "
            f"(~40 min of single-threaded pandas, with this job's GPU idle).")

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
    # cache_dir=None disables caching entirely (see setup.resolve_ts_cache_dir).
    use_cache = use_cache and cache_dir is not None
    cache_path = None if cache_dir is None else os.path.join(
        cache_dir, f"{type_dataset}{suffix}_K{K}_T{T}_{task}.npz")
    if use_cache and os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        if "X" in d.files:
            # Legacy full-window dump (pre-dedup format): wrap it in the compact contract
            # rather than reject it. reshape is a VIEW of the loaded X, so this costs only the
            # index array — it just carries no dedup saving, unlike a real compact source.
            X_legacy, mask_legacy = d["X"], d["mask"]
            n, k = mask_legacy.shape
            idx_legacy = np.arange(n * k, dtype=np.int32).reshape(n, k)
            idx_legacy[~mask_legacy] = -1
            return (X_legacy.reshape(-1, X_legacy.shape[-1]), idx_legacy,
                    d["y"], d["anchors"], d["nodes"])
        nodes = d["node_values"][d["node_codes"]].astype(object)
        return d["values"], d["idx"], d["y"], d["anchors"], nodes

    # Load + validate every snapshot once.
    snapshots = [_load_snapshot(i, type_dataset, suffix, features_dir) for i in range(T)]

    first_anchor = K - 1 if task == "nowcast" else K
    valid_anchors = list(range(first_anchor, T))

    # Windows are built (and cached) DEDUPLICATED: `values` holds each present
    # (snapshot, node) feature row once, `idx[sample, step]` points into it (-1 = node
    # absent at that step). Adjacent anchors share K-1 snapshots, so the old per-window
    # X dump wrote each row up to K times — ~K x the disk (zlib can't fold duplicates
    # that far apart in the stream) and K x the .loc lookups, for no extra information.
    row_cache = {}  # (snapshot index, node) -> row index into values
    values_rows = []
    idx_rows, y_rows, anchor_rows, node_rows = [], [], [], []
    for a in valid_anchors:
        anchor_df = snapshots[a]
        for node in anchor_df.index:
            win_idx = np.full(K, -1, dtype=np.int32)
            for step in range(K):
                si = _snapshot_index(a, step, K, task)
                snap = snapshots[si]
                if node in snap.index:
                    r = row_cache.get((si, node))
                    if r is None:
                        r = len(values_rows)
                        values_rows.append(
                            snap.loc[node, keys_to_include].to_numpy(dtype=np.float32))
                        row_cache[(si, node)] = r
                    win_idx[step] = r
            idx_rows.append(win_idx)
            y_rows.append(float(anchor_df.loc[node, "is_laundering"]))
            anchor_rows.append(a)
            node_rows.append(node)

    # Free each source as soon as it is converted: snapshots hold several GB of frames at
    # AMLWorld scale, and values_rows/idx_rows are second copies of their arrays. The dedup
    # layout keeps the build's peak well below the old ~2x-the-final-X: values is ~1/K of X.
    del snapshots, row_cache
    values = np.asarray(values_rows, dtype=np.float32).reshape(-1, N_FEATURES)
    del values_rows
    idx = np.asarray(idx_rows, dtype=np.int32).reshape(-1, K)
    del idx_rows
    y = np.asarray(y_rows, dtype=np.float32)
    anchors = np.asarray(anchor_rows, dtype=int)
    nodes = np.asarray(node_rows, dtype=object)

    # Scrub the deduplicated rows before expansion — same result as scrubbing X, once per
    # unique row instead of once per window copy.
    values = np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        # Compressed dedup cache: values + int32 idx (mask is derived as idx >= 0, node ids
        # factorised to codes + uniques). Write to a per-process temp file then os.replace()
        # into place: the fan-out baseline tasks (one per model, e.g. XGBoost and the MLP)
        # may rebuild the same combo concurrently, and an atomic replace guarantees a reader
        # sees either no file or a complete one — never a half-written .npz. Passing a file
        # object stops np.savez_compressed from appending its own ".npz" to the temp name.
        node_values, node_codes = np.unique(nodes.astype(str), return_inverse=True)
        tmp_path = f"{cache_path}.{os.getpid()}.tmp"
        with open(tmp_path, "wb") as fh:
            np.savez_compressed(fh, values=values, idx=idx, y=y, anchors=anchors,
                                node_codes=node_codes.astype(np.int32),
                                node_values=node_values)
        os.replace(tmp_path, cache_path)

    return values, idx, y, anchors, nodes


def build_sequence_dataset(*args, **kwargs):
    """DENSE wrapper over build_sequence_windows: (X, mask, y, anchors, nodes) with
    X float32 [N, K, F] (absent steps zeroed) and mask bool [N, K].

    Kept for callers that need the whole dense tensor in one array (notebooks, ad-hoc
    analysis). The experiments do NOT use it: X is ~K times `values`, so materialising it
    dominates their memory request for no benefit — they call build_sequence_windows and
    gather per batch instead."""
    values, idx, y, anchors, nodes = build_sequence_windows(*args, **kwargs)
    X, mask = _expand_windows(values, idx)
    return X, mask, y, anchors, nodes


def fit_scaler_compact(values, idx, train_idx):
    """(mu, sd) over the unique (snapshot, node) cells the TRAIN windows reference — the
    compact-form equivalent of fit_scaler, without materialising any windows.

    A row of `values` IS one (snapshot, node) cell, so "don't over-weight a snapshot reused
    across many windows" is exactly np.unique over the referenced row indices; fit_scaler
    spells the same thing out with a Python set of (node, snapshot) tuples. Accumulated in
    float64 as before. The dedup ROW ORDER differs (ascending vs first-encounter), so mu/sd
    can differ from fit_scaler in the last ULP — ~1e-16 relative, not a semantic change."""
    ref = np.unique(idx[train_idx])
    ref = ref[ref >= 0]
    flat = values[ref]
    # dtype=float64 accumulates in double WITHOUT materialising a float64 copy of the block
    # (that copy was ~5 GB at AMLWorld scale, and is the largest host allocation left on this
    # path). float32 -> float64 is exact, so the sums are bit-identical to converting first.
    mu = flat.mean(0, dtype=np.float64)
    sd = flat.std(0, dtype=np.float64)
    sd = np.where(sd < 1e-8, 1e-8, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_scaler_compact(values, mu, sd):
    """(values - mu) / sd on the compact block — one allocation, K times smaller than
    apply_scaler's. Absent steps need no re-zeroing here: they are not rows of `values` at
    all (idx == -1), and the gather that builds a window zeroes them."""
    Vs = values - mu
    Vs /= sd
    return Vs.astype(np.float32, copy=False)


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
