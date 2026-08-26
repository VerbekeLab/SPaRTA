# SPaRTA — Snapshots for Pattern Recognition in Time-variant Adjacency matrices

Research code for the SPaRTA paper (Deprez, Verdonck, Verbeke). Builds 3×3 ego-network
"pictures" from financial transaction graphs and trains classifiers (LogReg, XGBoost,
shallow NN, CNN, VGG16) to flag money-laundering accounts. Supports a static and a
time-windowed (snapshot) construction with optional exponential time-decay ("echo").

## Where things live

- **Dataset loaders** — [src/data/utils/AMLWorld.py](src/data/utils/AMLWorld.py),
  [AMLSim.py](src/data/utils/AMLSim.py), [Tide.py](src/data/utils/Tide.py). Loader
  contract: read raw CSVs from `data/<DatasetName>/`, rename to the canonical schema
  `{from_account, to_account, timestamp, amount, is_laundering}`, drop self-loops,
  convert `amount` to USD, parse `timestamp` to `datetime64[ns]`, and return a
  DataFrame. Add a new dataset by mirroring an existing loader and wiring it into
  `load_transactions` in [src/data/network_data_loader.py](src/data/network_data_loader.py:11).
- **Network construction** — [src/data/network_data_loader.py](src/data/network_data_loader.py):
  `construct_network` (static) and `construct_network_time` (snapshots).
  Time-windowing helpers live in [src/data/utils/dates.py](src/data/utils/dates.py).
- **GARG-AML / SPaRTA measures** — 3×3 block-density measures in
  [src/methods/utils/measure_functions.py](src/methods/utils/measure_functions.py)
  and per-node assembly in [src/methods/neighbourhood_picture.py](src/methods/neighbourhood_picture.py)
  (`gargaml_node_measures`, `transaction_measures`, `node_measures`).
  Neighbourhood selection in [src/methods/utils/neighbourhood_functions.py](src/methods/utils/neighbourhood_functions.py).
- **Feature extraction entry point** — [src/methods/measure_calculation.py](src/methods/measure_calculation.py)
  (`__main__`). Run from the repo root; writes `results/features/<type>_..._features.parquet`
  and matching `_labels.parquet` (zstd, via [src/utils/feature_io.py](src/utils/feature_io.py),
  which writes to a temp file and `os.replace`s it into place so a killed job never leaves a
  truncated table). The **dynamic** branch is resumable: it filters the date grid to snapshots
  whose feature+label tables are not yet on disk, so a wall-time kill continues on resubmit
  instead of restarting from snapshot 0. The static branch has no such granularity.
  Readers take extension-less stems and prefer Parquet with a legacy `.csv` fallback;
  `scripts/convert_features_to_parquet.py` migrates pre-existing CSVs in place
  (`--delete` to reclaim the disk space).
- **Compact windows, not dense tensors.** `build_sequence_windows`
  ([src/data/sequence_data.py](src/data/sequence_data.py)) returns `(values, idx, y, anchors,
  nodes)`: `values` [M, 90] holds each (snapshot, node) row once and `idx` [N, K] points into
  it (-1 = absent, so `mask` is `idx >= 0`). Adjacent anchors share K-1 snapshots, so the dense
  [N, K, 90] tensor is ~K times larger (3 GB -> ~30 GB at AMLWorld K=17) — it is what used to
  set the training jobs' `--mem` and GPU residency. **The experiments never build it**: the
  baselines gather each band's 2D array straight from `values` (nowcast takes ONE column
  instead of building all K and discarding 16/17 of it), and the sequence models keep one
  shared `values_t` on the device and gather `values_t[idx[batch]]` per batch via
  `gather_windows`. `fit_scaler_compact` / `apply_scaler_compact` are the compact-form scalers
  (`np.unique` over referenced rows replaces the per-cell Python set). `build_sequence_dataset`
  remains as a dense wrapper for notebooks and ad-hoc analysis — don't use it in the pipeline.
- **Window store (Stage 1.5)** — [src/data/window_store.py](src/data/window_store.py) +
  [scripts/build_window_store.py](scripts/build_window_store.py), submitted as
  [slurm/build_windows.slurm](slurm/build_windows.slurm) (one task per timing combo). Stacks
  every snapshot's 90 feature columns ONCE PER TAG into `_values.npy` + `_offsets.npy` +
  `_node_codes.npy` + `_node_universe.npy` + `_labels.npy` + `_meta.json`. The block depends
  on the tag alone — K and task only select which rows a window points at — so all 6
  (K × task) cells of a tag derive their windows from it via vectorised
  `Index.get_indexer` in seconds. `build_sequence_windows` tries it FIRST (before the
  transaction load, since T comes from `_meta.json`), then the legacy `.npz` cache, then a
  full re-assembly. Rows keep the feature file's order, so derived
  `(X, mask, y, anchors, nodes)` are bit-identical to the old builder's. GPU jobs set
  `SPARTA_REQUIRE_WINDOWS=1` so a missing store fails fast rather than costing ~40 min of
  single-threaded pandas with an idle A100; `SPARTA_WINDOW_STORE=off` restores the old path.
- **Network → image renderer** — [src/methods/visualising_network.py](src/methods/visualising_network.py)
  renders 224×224 RGB PNGs of each node's 2-hop subgraph via igraph + cairo, packs
  bits, and writes batched `.pkl` + `.h5` to `results/pickle/`.
- **Model definitions** — [src/methods/models.py](src/methods/models.py)
  (`NeuralNetwork`, `CNN`, `CNN_time`, `CNN_visual_VGG16`).
- **Training entry points** — [scripts/experiment_features.py](scripts/experiment_features.py)
  (LogReg + XGBoost grid search + NN on the 3×3 features),
  [scripts/experiment_CNN.py](scripts/experiment_CNN.py) (CNN over the 3×3 pictures
  with Optuna HPO), [scripts/experiment_pictures.py](scripts/experiment_pictures.py)
  (VGG16 over the 224×224 rendered network images),
  [scripts/experiment_LSTM.py](scripts/experiment_LSTM.py) (LSTM + Transformer over
  K-snapshot windows with Optuna HPO; sequence models only),
  [scripts/experiment_baseline.py](scripts/experiment_baseline.py) (non-sequential
  baselines — tuned XGBoost, a tuned feed-forward MLP (standardised inputs, mini-batch +
  early stopping), and unsupervised IsolationForest — on the **same** K-snapshot windows
  the sequence experiment uses, so the two are directly comparable).
  `experiment_CNN_LSTM.py` and `src/data/utils/synthetic.py` are empty placeholders to
  be implemented — fill them in following the patterns of the populated siblings; don't delete.
- **Configs** — [config/data/config.yaml](config/data/config.yaml) selects the active
  `dataset` and toggles `time_dynamic` / `echo`;
  [config/methods/config.yaml](config/methods/config.yaml) selects the active
  `experiment` and per-dataset HPs. Load with `src.utils.setup.load_config`. Both
  configs are indexed by the active `dataset` key — add a matching block when adding
  a new dataset, otherwise downstream `data_config[network]` accesses will `KeyError`.
- **Utilities** — [src/utils/graph_processing.py](src/utils/graph_processing.py)
  (`graph_community` Louvain pruning, hub removal), [src/utils/setup.py](src/utils/setup.py)
  (YAML loader).
- **Notebooks** — exploratory in `notebooks/`; not part of the experimental pipeline.

## Library preferences

- **Graphs:** networkx for all algorithmic work (ego graphs, communities, adjacency).
  igraph + cairo only for rendering 224×224 PNGs in `visualising_network.py`.
- **DL:** PyTorch (no Lightning trainer is actually wired up despite the dependency).
- **Tabular / preprocessing:** pandas; matplotlib for figures.
- **Classical ML:** xgboost (with `GridSearchCV`), sklearn `LogisticRegression`.
- **HPO:** optuna (CNN); sklearn `GridSearchCV` (XGBoost).
- **Metrics:** sklearn `average_precision_score` and `roc_auc_score`. **AUC-PR is the
  primary metric**, AUC-ROC is secondary — report both, optimise against AUC-PR.
- **Config:** YAML via PyYAML.

## Conventions

- **Canonical transaction schema:** `from_account, to_account, timestamp, amount,
  is_laundering`. Self-loops dropped. Amounts in USD. New loaders must conform.
- **Node labels:** derived by `define_ML_labels` — a node is positive if **any** of its
  in/out edges are laundering (the per-account laundering fraction is `> 0`; see
  [network_data_loader.py:36](src/data/network_data_loader.py:36)). Don't change this
  threshold without a paper-level reason.
- **Random seed:** `1997` everywhere (`train_test_split`, Louvain). Keep it.
- **Multiprocessing:** `n_cpu = min(4, cpu_count() // 2)`; use the `init_worker` +
  global-graph pattern already in `measure_calculation.py` / `visualising_network.py`.
- **Script preamble:** scripts and the `__main__` modules under `src/methods/` start with
  `os.chdir("./"); sys.path.append("./")` and must be **run from the repo root**.
- **Output layout:**
  - `results/features/` — extracted measure tables, zstd Parquet (one features + one
    labels per run; pre-migration runs may still be `.csv` — readers accept both).
  - `results/windows/<tag>/` — prebuilt per-tag window store (`$VSC_SCRATCH/SPaRTA/windows/<tag>`
    on VSC; see `resolve_window_store_dir`). Transient — rebuild with `build_windows.slurm`.
  - `results/pickle/` — batched rendered images (`nodes_<i>.pkl` + `images_tensor_<i>.h5`).
  - `results/experiments/` — per-model metric dumps (text).
  - `results/tuning/` — Optuna / grid-search best params, plus one **persistent** SQLite per
    Optuna study (`<study_name>.db`). These are never deleted: `optimize_study`
    ([src/utils/setup.py](src/utils/setup.py)) tops a study up to `n_trials` and runs nothing
    when it is already there, so a resubmitted job skips tuning instead of restarting it, and
    a `.db` with `n_trials` finished trials is the on-disk record that that cell is done
    (what [slurm/missing_runs.sh](slurm/missing_runs.sh) audits). Delete one by hand only to
    force a fresh search — required after changing a `search_space` categorical.
  - `results/models/` — saved `state_dict`s.
- **Class imbalance:** `BCEWithLogitsLoss(pos_weight=...)` with the empirical
  negative/positive ratio (sometimes scaled). Don't substitute focal loss etc. silently.
- **No `_t` suffix on feature files.** Feature/label files are `<type>_static_…` / `<type>_dynamic_{i}_…`
  (`_echo` for time-decay). A legacy `_t` suffix once marked transaction-weight features; it was
  dropped — all current outputs include the transaction summaries.
- **Echo mode:** when `echo=True`, the time window is overridden to `[end - days_echo,
  end]` and edges/labels are weighted by `exponential_time_decay`. See
  [network_data_loader.py:63](src/data/network_data_loader.py:63).

## Hard rules

- **`data/` is read-only.** All subdirectories (`data/AMLWorld/`, `data/Tide/`, `data/amlsim/`,
  `data/Elliptic/`, MNIST/CIFAR) hold raw or vendor-supplied files. Never write,
  rename, or regenerate inside `data/`; generated artefacts go under `results/`.
  (There is no `data/raw/` — the convention applies to all of `data/`.)
- **Persistent-node, timestamped datasets only.** Use AMLWorld, AMLSim, Tide. **Do not
  add transaction-as-node datasets** (e.g. the classic Elliptic dataset). The
  `data/Elliptic/` directory exists as a placeholder but must stay empty.
- **Legacy GARG-AML code is frozen for reproducibility.** The nine
  `measure_XY_function`s in [src/methods/utils/measure_functions.py](src/methods/utils/measure_functions.py),
  `node_selection` in [neighbourhood_functions.py](src/methods/utils/neighbourhood_functions.py),
  and `gargaml_node_measures` in [neighbourhood_picture.py](src/methods/neighbourhood_picture.py)
  must remain bit-exact — don't refactor or change their numeric outputs. Adding new
  measures, summaries, or columns alongside them is fine (e.g. `summary_functions.py`
  and `transaction_measures` are SPaRTA-side and freely extensible).
- **Don't commit data, results, notebooks, or generated images.** `.gitignore`
  already excludes `/data/`, `/res/`, `/results/`, `/notebooks/`,
  `src/data/pictures/`, `*.png`, `*.pdf` — keep it that way.
- **Don't break the canonical column schema.** Downstream code (network builder,
  label derivation, snapshot windowing) assumes those exact names.

## Environment

Conda recipe in [environment.txt](environment.txt) (Python 3.12, includes the
igraph + pycairo + h5py + optuna extras). The minimal `requirements.txt` is for
pip-only installs and omits igraph/pycairo/h5py/pyarrow — prefer `environment.txt`
for full reproducibility.
