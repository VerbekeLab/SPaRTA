# Time-series notebook — implementation plan

Blueprint for `notebooks/test_timeseries.ipynb`: a small, readable end-to-end test of
sequence classification on SPaRTA features (LSTM + Transformer). This notebook is the
prototype for the eventual `.py` implementation in the repo. **All code lives in the
notebook**; existing files are *reused* (imported), never edited.

## Locked decisions

| # | Decision | Choice |
|---|----------|--------|
| 1 | Dataset | **AMLWorld HI-Small** (notebook is explicit; chosen for many partially-present nodes) |
| 2 | Time-series shape | **Single 3-snapshot series** — sequence length = 3, one window spanning 3 consecutive daily snapshots |
| 3 | Per-snapshot construction | **Echo on, `days_echo=3`, 1-day steps** — reuse `load_network_time(echo=True, days_echo=3)` (matches `config/data/config.yaml`) |
| 4 | Cache | **Per-node 90-dim feature vectors per snapshot**, written under `results/timeseries/` (never `data/`) |
| 5 | Label | **Status at the last (3rd) snapshot** via `define_ML_labels`; modeled set restricted to nodes **present at the last snapshot** |
| 6 | Snapshot selection | **3 consecutive daily windows with the most laundering nodes** (scanned, overridable variable) |
| 7 | LSTM padding | **Zero-fill missing steps + take last-valid step's hidden state** |
| 8 | Transformer padding | **`src_key_padding_mask`** (native), mean-pool over valid tokens |
| 9 | Sampling (class balance) | **All positives present (cap 50) + random negatives → ~150**, seed 1997 (laundering ≈ 0.4%, so pure-random gives ~0 positives) |

Conventions kept: seed **1997**, **AUC-PR primary** / AUC-ROC secondary,
`BCEWithLogitsLoss(pos_weight=neg/pos)`, `_t` + `echo` file-name suffixes.

## Feature vector (90-dim per node per snapshot)

Reuses `node_measures` (frozen GARG-AML + SPaRTA transaction measures). Flattened in the
**exact order** of `measure_calculation.keys_to_include` so the notebook matches the main
pipeline's feature semantics:

- `measure_00..22` (9) + `size_00..22` (9) = 18
- `transaction_amount_trans_summary_{sum,mean,max,std}_00..22` = 36
- `transaction_num_trans_summary_{sum,mean,max,std}_00..22` = 36
- **Total = 90**

## Reused, unchanged code

| What | Where | Use |
|------|-------|-----|
| `load_network_time` | `src/data/network_data_loader.py` | build each snapshot graph (echo) |
| `define_ML_labels` | same | label at the last snapshot |
| `define_dates` | `src/data/utils/dates.py` | enumerate daily windows |
| `graph_community` | `src/utils/graph_processing.py` | Louvain pruning per snapshot |
| `node_measures` | `src/methods/neighbourhood_picture.py` | 2-hop ego features per node |
| `keys_to_include` | `src/methods/measure_calculation.py` | 90-feature column order |

**Deviation from convention (justified for a test):** the heavy pipeline extracts
features with a multiprocessing `Pool` + `init_worker`. With ≤300 nodes the notebook runs
`node_measures` **serially** for simplicity. The graph setup mirrors `process_graph`:
`G_red = graph_community(G); G_und = G_red.to_undirected(); G_rev = G_red.reverse()`.

## Sampling & modeled set

Measured reality: HI-Small = **5.08M txns over 10 days (2022-09-01 → 2022-09-10)**;
laundering rate **~0.4%**, so *random* sampling yields ~0 positives. A consequence of the
"restrict to nodes present at the last snapshot" labeling choice is that **only the last
snapshot's sample defines the modeled set** — earlier snapshots only supply each modeled
node's history. So the modeled set is sampled at the anchor, **stratified to include
positives** (decision #9):

1. **Pick dates** (decision #6): daily grid via `define_dates`; anchor = the day whose
   3-day echo window holds the most distinct laundering accounts; snapshots = the 3 days
   ending at the anchor.
2. **Build** each snapshot: `load_network_time(echo=True, days_echo=3)` → `graph_community`.
3. **Modeled set `M`** (from the last/anchor snapshot's pruned graph): **all positive
   nodes present** (cap `MAX_POS=50`) + **random negatives** up to `N_TOTAL≈150`, seed 1997.
   Label `y` = `is_laundering` at the anchor.
4. For every node in `M`, compute the 90-dim features at **each** of the 3 snapshots
   where it is present (presence = in that snapshot's pruned graph). A node present at
   steps 1 & 3 but not 2 → step 2 **internally padded**; every `M` node is present at
   step 3 by construction.

Tractability (measured): one snapshot ≈ `load_network_time` 6s + Louvain 22s +
`node_measures` 16 ms/node ⇒ **~2 min first run** for all 3; instant after feature caching.

## Combined single pass (feature calc + masking together)

One walk over the 3 snapshots builds features **and** the padded/masked tensor at once —
no second pass:

```
X    = zeros(N, 3, 90)      # N = |M|, padded steps stay 0
mask = zeros(N, 3, bool)    # True where the node is present
for t, snapshot in enumerate(snapshots):
    feats_t = load_cache(t) or compute_features(snapshot, pool=P)  # cache under results/timeseries/
    for node in M present at t:
        X[idx(node), t] = feats_t[node]
        mask[idx(node), t] = True
y = label_at_last_snapshot(M)
```

Cache files (per snapshot, mirroring repo naming):
`results/timeseries/HI-Small_snap{t}_features_echo.csv` (+ `_labels_echo.csv`) and a
small `meta.json` recording chosen dates + sampled node sets for reproducibility.

## Preprocessing

- **Standardize** each of the 90 features (z-score) using stats over **valid (unmasked)
  training** entries only; re-zero padded positions after scaling so mask semantics hold.
- **Split**: stratified 80/20 on `M`, seed 1997.
- `pos_weight = round(neg/pos)` for `BCEWithLogitsLoss`.

## Models (PyTorch, deliberately small)

**LSTM**
- `nn.LSTM(input_size=90, hidden_size≈32, batch_first=True)` over the zero-filled sequence.
- Gather the output at each node's **last valid step** (from `mask`) → **`Dropout(0.3)`** → `Linear(hidden,1)` → logit.

**Transformer**
- `Linear(90 → d_model≈32)` input projection.
- **Sinusoidal positional encoding** (fixed, added to projected features).
- `nn.TransformerEncoder` (1–2 layers, few heads, `batch_first=True`, **`dropout=0.2`** in attention + FFN), `src_key_padding_mask = ~mask`.
- **Mean-pool over valid tokens** → **`Dropout(0.2)`** → `Linear(d_model,1)` → logit.

**XGBoost (baseline — final step only)**
- Gradient-boosted trees on the **anchor snapshot's 90 features only** (step 3); steps 1–2
  ignored. Raw (un-standardised) features; imbalance via `scale_pos_weight`. Untuned, fixed
  params — a non-sequential reference for what the temporal history actually adds.
- ⚠️ **Import order:** `xgboost` must be imported **before** `torch` (on macOS both bundle an
  OpenMP runtime; torch-first segfaults the kernel). The setup cell handles this, mirroring
  `experiment_features.py`. After editing, *restart kernel & run all* so the order takes effect.

Shared training loop (`train_eval`): Adam, **shuffled mini-batches** (`batch_size=16`,
`TensorDataset`/`DataLoader`) for up to **`max_epochs=300`** with **early stopping** —
a stratified **20% validation split** (seed 1997) is carved from the training tensors, val
AUC-PR is scored every epoch, the **best-scoring checkpoint** is kept, and training stops after
**`patience=30`** epochs without improvement (best weights restored before testing). This
replaced the original full-batch, fixed-100-epoch loop, which under-trained the (then
dropout-free) sequence models. The loss comes from a small reusable
`make_loss(kind, pos_weight, alpha, gamma)` factory (one place, reused across every training
round): `kind="bce"` (default — `BCEWithLogitsLoss(pos_weight)`) or `kind="focal"`
(torchvision `sigmoid_focal_loss`, `alpha=0.75` up-weighting the rare positive class,
`gamma=2`; **alpha replaces `pos_weight`**, no double-counting). Focal is **not** custom code —
`torchvision.ops.sigmoid_focal_loss` is already a project dependency. Switch any model with
`train_eval(..., loss="focal")`. Each model records its test predictions into a shared
`MODEL_SCORES` dict and reports **AUC-PR** + AUC-ROC. The validation split, mini-batching and
early stopping carry over to the Forecasting section unchanged (same `train_eval`). Tiny
sample → noisy metrics (a ~24-node val set makes early-stop epoch choice jittery); goal is
*correctness of the pipeline*, not SOTA.

## Evaluation

Beyond single AUC numbers, the final section draws two comparison plots over `MODEL_SCORES`
(same test split for all models): a **ROC curve** (with chance diagonal) and a
**Precision–Recall curve** (with the positive-prevalence no-skill line). PR is the more
informative view under heavy class imbalance. Colours use the repo's Okabe-Ito palette
(`src/utils/visualisation.OKABE_ITO`). The plotting is a reusable `plot_curves(scores, y_true,
title)`; `train_eval`/`record_scores` are generalized to accept alternate tensors + a separate
scores dict, so the forecasting section reuses them unchanged.

## Forecasting (nowcasting vs. forecasting)

The default task is **nowcasting**: the label is the node's status at the anchor (step 3) and
every model — including XGBoost — sees step-3 features, which are *concurrent* with the label.
A separate **forecasting** section makes it predictive instead: predict the anchor label from
**prior steps (1–2) only**, step 3 withheld. Sequence models take the 2-step history; the
XGBoost baseline takes the *same* history flattened (180 feats), so the only difference is
sequential-vs-not. Nodes with no prior observation are dropped (also avoids an all-masked
Transformer row). Same split / standardisation / losses / seed → a controlled A/B on the value
of the time dimension. (Empirically: withholding step 3 closes the gap — the Transformer
overtakes XGBoost on AUC-PR — confirming the nowcasting win was a concurrent-feature artifact.)

## Notebook cell layout

1. **Setup** — `os.chdir`/`sys.path` preamble, imports, seed, `results/timeseries/` dir.
2. **Subsample** — pick 3 snapshots (most-laundering scan), build pruned graphs, sample
   100/step, form pool `P`. Cache graphs' metadata.
3. **Combined feature + pad/mask pass** — single walk producing `X`, `mask`, `y`, node index
   (with per-snapshot feature caching). Print a few examples incl. an internally-padded node.
4. **Preprocess** — standardize, stratified split.
5. **LSTM** — model class; train/eval with **both losses** → `LSTM (bce)`, `LSTM (focal)`.
6. **Transformer** — `PositionalEncoding` + model class; train/eval with **both losses**.
7. **XGBoost** — baseline on step-3 features only; records into `MODEL_SCORES`.
8. **Evaluation** — ROC + PR curves comparing all **five** configurations (Okabe-Ito palette):
   LSTM (bce/focal), Transformer (bce/focal), XGBoost (step 3). Defines reusable `plot_curves`.
9. **Forecasting** — predict the anchor from prior steps (1–2) only, step 3 withheld; sequence
   models on the 2-step history vs. XGBoost on the same history flattened (180 feats); drops
   no-history nodes; separate `FORECAST_SCORES` → its own ROC + PR plot via `plot_curves`.

## Open items / would require editing existing files (none planned)

No edits to existing files are needed. If during implementation a reused function turns
out unsuitable, I will stop and discuss before changing it (per your instruction).

---

# Migration to the repository (.py)

Port the notebook prototype into the package so it runs for **all datasets** (AMLWorld,
AMLSim, Tide) and integrates with the existing pipeline. Keep the notebook's minimalist
style; make it efficient (mini-batching, GPU, caching).

## Locked decisions (from Q&A)

| Topic | Decision |
|-------|----------|
| Sequence | **Sliding window of `K` snapshots** ending at each anchor. `K` = config (default **3**). `T` (snapshot count) is data-derived from `define_dates` (`len(end_dates)`). |
| Samples | **Multiple per node** — one per valid anchor `a`. Anchor slides across snapshots. |
| Split | **Temporal holdout** — earliest anchors → train, a middle band → val (for Optuna/early-stop), latest `n_test_anchors` → test. **No embargo (by decision):** echo windows overlap (3-day window, 1-day step), so a snapshot's features can appear in both a train window and the test window. This is **kept on purpose** because it mirrors production — the model is retrained on all history to date and then scores the next snapshot, so consecutive scoring windows naturally overlap. Only *label-time ordering* is enforced (no future anchor leaks into train), and the scaler/model are fit on train data only, so there is no target leakage. The same account recurring across time is likewise intended/realistic. |
| Task | **Config toggle `nowcast` | `forecast`** (default `nowcast`). `nowcast`: window `[a−K+1 … a]`, includes the anchor step. `forecast`: window `[a−K … a−1]`, anchor step withheld. |
| HPO | **Per-dataset Optuna** (mirrors `experiment_CNN.py`), optimise **AUC-PR** over architecture (hidden/`d_model`, layers, heads), **dropout**, lr, batch size, epochs, and **loss** (`bce`|`focal` + `alpha`/`gamma`). The search space respects `d_model % nhead == 0` by parameterising `d_model = head_dim × nhead`. **Class imbalance is handled by the loss search itself, not a separate weight knob:** `bce` uses the plain empirical `pos_weight = neg/pos`, and where the ~250× full-population imbalance makes that BCE unstable Optuna can simply select `focal` (tuned `alpha`/`gamma`), which up-weights the rare positive without an exploding gradient. So there is **no** `pos_weight_scale` parameter — one less knob, and `loss` already routes around bad imbalance. `K` and `task` are fixed per run (they change data shape), not tuned. |
| Models | LSTM + Transformer + XGBoost baseline, all in one script; evaluation factored out. The XGBoost baseline gets its **own minimal tuning** (small Optuna/grid over `max_depth`/`eta`/`subsample`/`n_estimators`, scored AUC-PR on the val band) so the sequence-vs-not comparison isn't confounded by one side being tuned and the other fixed. |
| Population | **Full node set, real imbalance** (no subsampling). Reuse `measure_calculation.py`'s per-snapshot feature CSVs. |

## Data spec (concrete)

1. **Source features** (reused, not recomputed): read
   `results/features/{type}_dynamic_{i}_features{_echo}.csv` + `…_labels{_echo}.csv`
   for `i = 0…T−1`. Merge `features.node ↔ labels.Account`. Build a `node × snapshot`
   feature table + a **presence mask** (present ⇔ node has a row in snapshot `i`).
   ⚠️ The on-disk dynamic files are an **older 19-column variant** (no transaction
   features); the 90-feature outputs must be **regenerated** (run `measure_calculation.py`
   with `time_dynamic=True`).
2. **Windowing**: for each anchor `a` in `[K−1 … T−1]` (nowcast) / `[K … T−1]` (forecast),
   for each node present at `a`: sample = the `K`-step window (masked where absent),
   `label` = node's `is_laundering` at snapshot `a`. Guard: skip a dataset/anchor if `T < K`.
3. **Temporal split** by anchor index (config cutoffs/fractions; default: last `n_test_anchors`
   (≥1) → test, the band of `n_val_anchors` before → val, the rest → train). Label-time ordering
   preserved (no future anchor in train); overlapping feature windows across the boundary are
   intentional (see Split). **Guards:** require `#valid anchors ≥ n_test_anchors + n_val_anchors + 1`
   (≈ `T ≥ K + 2` for the default 1/1 split), and assert each of val/test holds ≥1 positive —
   otherwise fail with a clear message naming the dataset and its derived `T` (AUC-PR is undefined
   on a positive-free split, and single-anchor test sets are tiny/noisy).
4. **Standardise** mask-aware on **train windows only**, computing `(mu, sd)` over the **unique
   `(node, snapshot)` cells** present in train windows (so a snapshot reused across many windows
   isn't over-weighted); re-zero padded steps after scaling; persist `(mu, sd)` with the model.

## File map

| Action | Path | Contents |
|--------|------|----------|
| **CREATE** | `src/data/sequence_data.py` | `build_sequence_dataset(dataset, type_dataset, echo, K, task, …)` → `(X[N,K,F], mask[N,K], y[N], anchors[N], nodes[N])` from the per-snapshot CSVs; `SequenceDataset`; mask-aware scaler; temporal-split helper. Reads `{type}_dynamic_{i}_features{_echo}.csv` + `…_labels{_echo}.csv`, with the `_echo` suffix chosen from `network_construction.echo` (not hardcoded). **Validates** that every `i = 0…T−1` file exists and carries the full 90 columns (`keys_to_include`), erroring with a pointer to `measure_calculation.py` if the older 19-col variant is found. Temporal-split helper enforces the anchor-count / ≥1-positive guards (Data spec §3). Optional `.npz` cache under `results/timeseries/`. |
| **CREATE** | `src/methods/losses.py` | `make_loss(kind, pos_weight, alpha, gamma)` — BCE + torchvision `sigmoid_focal_loss`. |
| **UPDATE** | `src/methods/models.py` | Add `PositionalEncoding`, `LSTMClassifier`, `TransformerClassifier` (length-agnostic `forward(x, mask)`). **`LSTMClassifier`**: pass `dropout=` into `nn.LSTM` (active for `num_layers>1`) **and** keep the last-valid-step head dropout, so tuned multi-layer LSTMs are actually regularised. **`TransformerClassifier`**: take `nhead` + `head_dim` and set `d_model = head_dim × nhead`, so every Optuna trial satisfies `d_model % nhead == 0`. |
| **FILL** | `src/methods/evaluation.py` | `plot_curves(scores, y_true, title, save_path)` (Okabe-Ito) + metric/summary writers → `results/experiments/`. Reusable. |
| **FILL** | `scripts/experiment_LSTM.py` | Entry point: config → `build_sequence_dataset` → temporal split → Optuna(LSTM, Transformer) → train best → **XGBoost baseline with its own minimal tuning** (small grid/Optuna scored AUC-PR on the val band) → evaluate → save `results/{tuning,models,experiments}`. **`import xgboost` before `torch`.** |
| **UPDATE** | `config/methods/config.yaml` | Add a `timeseries` block per dataset via anchors: Optuna `search_space` (`hidden`/`num_layers`/`head_dim`/`nhead`/`dropout`/`lr`/`batch_size`/`epochs`/`loss`+`alpha`/`gamma`) plus a small `xgb_search_space`; `n_trials`, `storage`. No `pos_weight_scale` — `bce` uses the empirical `pos_weight = neg/pos` and the `loss` search selects `focal` when that's unstable (see HPO). Extend the `experiment` selector. |
| **UPDATE** | `config/data/config.yaml` | Add a small `sequence` block per dataset: `K` (default 3), `task` (`nowcast`), `n_test_anchors` / `n_val_anchors` (default 1 each). |
| **LEAVE empty** | `scripts/experiment_CNN_LSTM.py`, `src/data/utils/synthetic.py` | Out of scope. |

**Reused unchanged:** `measure_calculation.py`, `network_data_loader.py`, `dates.py`,
`graph_processing.py`, `setup.py` (`load_config`/`resolve_*`/`suggest_param`),
`visualisation.py` (`OKABE_ITO`), `feature_data.py`.

**Prerequisite / all-datasets:** features must exist for the active dataset
(`measure_calculation.py` already supports all three). `T` adapts per dataset; per-dataset
config via YAML anchors.

## Improvements carried over from the notebook

Full population (real imbalance) · Optuna HPO on AUC-PR (sequence models **and** a minimally
tuned XGBoost baseline) · dropout wired through multi-layer LSTM + valid `d_model`/`nhead`
trials · imbalance handled by the tuned `bce`/`focal` loss search (empirical `pos_weight` for
BCE, `focal` selected when BCE is unstable — no extra weight knob) · mini-batch DataLoader +
longer training + early stopping on val · 3-way temporal split with anchor-count / ≥1-positive
guards (overlapping windows kept on purpose — production-faithful) · GPU/MPS · persisted
mask-aware scaler · saved params/state_dict/metrics+curves · xgboost-before-torch import order.

## Implementation workflow (after sign-off)

1. **Build** (parallel agents): `sequence_data.py`, `models.py` additions, `losses.py`,
   `evaluation.py`, config blocks, `experiment_LSTM.py`.
2. **Integrate + smoke-verify**: run end-to-end on the available features with a tiny Optuna
   budget; assert shapes, no NaN, temporal split has both classes, runs per dataset.
3. **Adversarial review**: correctness (label-time ordering & train-only fitting — note the
   overlapping feature windows are intentional, not a bug; masking, label/anchor alignment,
   anchor-count / ≥1-positive guards), integration (config keys, reuse, naming), faithfulness to
   the notebook's minimalism.
