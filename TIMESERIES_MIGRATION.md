# SPaRTA — Time-series migration (notebook → repository .py)

> **Purpose:** self-contained, resumable checklist for porting `notebooks/test_timeseries.ipynb`
> into the package per the "Migration to the repository (.py)" section of
> `notebooks/TIMESERIES_PLAN.md`. Works for **all datasets** (AMLWorld, AMLSim, Tide),
> integrates with the existing pipeline, keeps the notebook's minimalist style.
> Tick boxes as you go; fill the `Done note:` line on completion so a later session resumes from here.

## For Claude: how to use and maintain this file

- On entry: read this section, then **Ground rules**, then **Interface contract**, then the first
  task whose box is not `[x]`. Each task is self-contained.
- `[ ]` not started · `[~]` in progress · `[x]` done & verified. Set `[~]` before editing; on a
  `[~]` task assume a prior session was interrupted — check `git diff` before continuing.
- Update task text to match reality if it drifts. Update this file in the same turn as the code change.
- Run from the **repo root** with the `SPaRTA_env` interpreter:
  `/Users/brunodeprez/anaconda3/envs/SPaRTA_env/bin/python` (NOT the default python3 → py3.7, breaks).

## Ground rules

- **Reused UNCHANGED (do not edit):** `measure_calculation.py`, `network_data_loader.py`,
  `dates.py`, `graph_processing.py`, `setup.py`, `visualisation.py` (`OKABE_ITO`), `feature_data.py`,
  the frozen GARG-AML numerics. If a reused function turns out unsuitable, STOP and ask before editing.
- **Leave empty (out of scope):** `scripts/experiment_CNN_LSTM.py`, `src/data/utils/synthetic.py`.
- **Conventions kept:** seed **1997**; **AUC-PR primary**, AUC-ROC secondary; `_echo` filename suffix
  driven by `network_construction.echo` (not hardcoded); GPU/MPS auto-detect (mirror `experiment_CNN.py`);
  `import xgboost` BEFORE `torch` (OpenMP segfault otherwise).
- **No subsampling** — full node set, real imbalance. Imbalance handled by the tuned `bce`/`focal`
  loss search (empirical `pos_weight = neg/pos` for bce; `focal` selected when bce unstable). **No
  `pos_weight_scale` knob.**

## Locked decisions (from TIMESERIES_PLAN.md "Migration" Q&A)

| Topic | Decision |
|-------|----------|
| Sequence | Sliding window of `K` snapshots ending at each anchor. `K` = config (default **3**). `T` = `len(define_dates(...))`, data-derived per dataset. |
| Samples | Multiple per node — one per valid anchor `a` (anchor slides across snapshots). |
| Task | Config toggle `nowcast`\|`forecast`. nowcast window `[a−K+1 … a]` (incl. anchor); forecast `[a−K … a−1]` (anchor withheld). |
| Split | **Temporal holdout** by anchor index: latest `n_test_anchors` → test, preceding `n_val_anchors` → val, rest → train. **No embargo (intentional):** echo windows overlap; production-faithful. Only label-time ordering enforced; scaler/model fit on train only. |
| HPO | Per-dataset Optuna, optimise **AUC-PR** on the val band. Search arch (`hidden`/`head_dim`/`nhead`/`num_layers`), `dropout`, `lr`, `batch_size`, `epochs`, `loss` (`bce`\|`focal` + `alpha`/`gamma`). `d_model = head_dim × nhead` (so `d_model % nhead == 0`). `K`/`task` fixed per run. XGBoost gets its own minimal tuning on the val band. |
| Population | Full node set, real imbalance; reuse `measure_calculation.py` per-snapshot CSVs. |

## Data spec (concrete)

1. **Source features (reused, not recomputed):** `results/features/{type}_dynamic_{i}_features{_echo}.csv`
   + `…_labels{_echo}.csv` for `i = 0…T−1`. Merge `features.node ↔ labels.Account`; label column
   renamed positionally to `is_laundering` (old files say `Is Laundering`). Build node×snapshot feature
   table + presence mask (present ⇔ node has a row in snapshot `i`).
   ⚠️ On-disk dynamic files are the **older 19-col variant** (no transaction features) → must be
   **regenerated** (`measure_calculation.py`, `time_dynamic=True`). The loader validates 90 cols and
   errors with a pointer to `measure_calculation.py` if it sees the 19-col variant.
2. **Windowing:** for each anchor `a` in `[K−1 … T−1]` (nowcast) / `[K … T−1]` (forecast), for each
   node present at `a`: sample = the `K`-step window (masked where absent), `label` = `is_laundering`
   at snapshot `a`. Guard: skip dataset/anchor if `T < K`.
3. **Temporal split** by anchor index. Guards: `#valid anchors ≥ n_test_anchors + n_val_anchors + 1`
   (≈ `T ≥ K + 2` for default 1/1); assert val and test each hold ≥1 positive; else fail with a clear
   message naming the dataset and its derived `T`.
4. **Standardise** mask-aware on **train windows only**: `(mu, sd)` over the **unique `(node, snapshot)`
   cells** present in train windows (a snapshot reused across windows isn't over-weighted); re-zero
   padded steps after scaling; persist `(mu, sd)` with the model.

## Interface contract (freeze — every module codes against this)

**`src/data/sequence_data.py`** (CREATE)
- `build_sequence_dataset(dataset, type_dataset, echo, K, task, features_dir="results/features", cache_dir="results/timeseries", use_cache=True, transactions=None, time_step=1, time_width=1, time_type="days") -> (X, mask, y, anchors, nodes)`
  - `X` float32 `[N,K,F]` (F=90, padded steps = 0); `mask` bool `[N,K]`; `y` float32 `[N]`;
    `anchors` int `[N]` (anchor snapshot index per sample); `nodes` object `[N]` (node id per sample).
  - Derive `T = len(define_dates(transactions['timestamp'], time_step, time_width, time_type))`,
    loading `load_transactions(dataset, type_dataset)` if `transactions is None`.
  - Validate every `i=0…T−1` features+labels file exists and features carry exactly the 90
    `keys_to_include` columns; else `raise ValueError` pointing at `measure_calculation.py`.
  - `_echo` suffix from `echo`. `np.nan_to_num` the feature matrix. Optional `.npz` cache keyed by
    `(type_dataset, echo, K, task)` under `cache_dir`.
- `SequenceDataset(torch.utils.data.Dataset)` — wraps `(X, mask, y)`; `__getitem__ -> (x[K,F] f32, m[K] bool, y f32)`.
- `fit_scaler(X, mask, anchors, K, task, train_idx) -> (mu[F], sd[F])` — dedup on `(sample's node, snapshot index)`
  over masked cells of `train_idx`; snapshot index recomputed from `anchors`,`K`,`task` (same formula as builder).
  *(node identity for dedup: pass `nodes` too if needed — keep signature `fit_scaler(X, mask, nodes, anchors, K, task, train_idx)`.)*
- `apply_scaler(X, mask, mu, sd) -> X'` — `(X-mu)/sd`, then re-zero `~mask`.
- `temporal_split(anchors, y, n_test_anchors=1, n_val_anchors=1, dataset_label=None) -> (train_idx, val_idx, test_idx)`
  — sample-index arrays; enforces the §3 guards (anchor count, ≥1 positive in val and test).

**`src/methods/losses.py`** (CREATE)
- `make_loss(kind="bce", pos_weight=None, alpha=0.75, gamma=2.0) -> callable(logits, targets)->scalar`
  — `bce`: `BCEWithLogitsLoss(pos_weight)`; `focal`: `torchvision.ops.sigmoid_focal_loss(alpha,gamma,reduction="mean")`
  (alpha replaces pos_weight; pos_weight ignored for focal). logits/targets shape `(B,)`.

**`src/methods/models.py`** (UPDATE — keep existing classes byte-for-byte)
- `PositionalEncoding(d_model, max_len=512)`; `forward(x) = x + pe[:, :x.size(1)]`.
- `LSTMClassifier(n_features=90, hidden=32, num_layers=1, dropout=0.3)`:
  `nn.LSTM(n_features, hidden, num_layers, batch_first=True, dropout=(dropout if num_layers>1 else 0.0))`
  + `nn.Dropout(dropout)` on the last-valid-step hidden state + `nn.Linear(hidden,1)`.
  `forward(x, mask)`: last valid step via `mask.float().cumsum(1).argmax(1)`, gather, drop, linear, squeeze(-1).
- `TransformerClassifier(n_features=90, head_dim=8, nhead=4, num_layers=1, dim_feedforward=None, dropout=0.2)`:
  `d_model = head_dim*nhead`; `Linear(n_features,d_model)`; `PositionalEncoding(d_model)`;
  `TransformerEncoderLayer(d_model, nhead, dim_feedforward or 2*d_model, dropout, batch_first=True)` × num_layers;
  `forward(x, mask)`: `src_key_padding_mask=~mask`, mean-pool over valid tokens (`clamp(min=1)`), drop, linear, squeeze(-1).
- Both length-agnostic. **Caller must drop all-masked rows** (forecast no-history) — nowcast anchor step is always valid.

**`src/methods/evaluation.py`** (FILL)
- `evaluate_scores(y_true, scores) -> (aucpr, aucroc)`.
- `plot_curves(scores_dict, y_true, title, save_path=None)` — ROC + PR on shared axes, Okabe-Ito,
  prevalence/chance reference lines; saves to `save_path` if given.
- `write_metrics(path, scores_dict, y_true)` — per-model AUC-PR/AUC-ROC (+ curve arrays) → `results/experiments/`.

**`scripts/experiment_LSTM.py`** — sequence models only; preamble; GPU/MPS.
- config → `build_sequence_dataset` → `temporal_split` → `fit_scaler`/`apply_scaler` (fit on train) → tensors/device.
- Script-local `train_model(model, train_tensors, val_tensors, loss, ...)`: mini-batch Adam, early-stop on the
  **temporal val band** (not a random carve), restore best weights, return best val AUC-PR + state. (Adapts the
  notebook's `train_eval` to the explicit val band.)
- Optuna study per architecture (LSTM, Transformer), maximise val-band AUC-PR via `suggest_param` over
  `method_config[dataset]['timeseries']['search_space']`. Train best on train (early-stop on val), score test.
- Save: `results/tuning/timeseries_..._best_params.json`, `results/models/{LSTM|Transformer}_{type}_{task}{suffix}.pt`
  (state_dict + scaler `(mu,sd)`), `results/experiments/{type}_timeseries_{task}{suffix}.{txt,png}`.

**`scripts/experiment_baseline.py`** — non-sequential baselines on the **same** K-snapshot windows
(`import xgboost` before `torch`; no torch needed elsewhere here).
- config → `build_sequence_dataset` → `temporal_split` → per-task 2D arrays (`_xgb_arrays`).
- **XGBoost**: nowcast → anchor-step (step `K−1`) 90 feats; forecast → **all K window steps flattened
  (`K*90`)** — the forecast window `[a−K … a−1]` already excludes the anchor, so every step is history;
  baselines must see the **same** K-step history the sequence models do (slicing shorter silently
  handicaps them → confounds the sequential-vs-not A/B). Own small Optuna (`xgb_search_space`, budget
  `baselines.n_trials`) on the val band. Raw features, `scale_pos_weight`.
- **IsolationForest** (`run_iforest_baseline`): **unsupervised** anomaly detector on the **same per-task
  arrays as XGBoost**. Ignores labels — fits on train, ranks test by `-score_samples` (higher = more
  anomalous). **Fixed** params from the `iforest` config block (not tuned: nothing to optimise against
  AUC-PR without labels).
- Save: `results/tuning/baselines_..._best_params.json`, `results/experiments/{type}_baselines_{task}{suffix}.{txt,png}`.
- Kept separate from `experiment_LSTM.py` so baselines and sequence experiments evolve independently
  and so other (non-baseline, non-sequential) methods can be added as their own `experiment_*.py`.

**`config/data/config.yaml`** (UPDATE) — add per-dataset `sequence` block (via anchors):
`K: 3`, `task: nowcast`, `n_test_anchors: 1`, `n_val_anchors: 1`.

**`config/methods/config.yaml`** (UPDATE) — two decoupled per-dataset blocks (via anchors):
- `timeseries:` sequence-model-only — `search_space` (`hidden`/`num_layers`/`head_dim`/`nhead`/`dropout`/`lr`/
  `batch_size`/`epochs`/`loss`+`alpha`/`gamma`), `n_trials`, `storage` (null default), `data_directory`.
- `baselines:` non-sequential — `xgb_search_space` (`max_depth`/`eta`/`subsample`/`n_estimators`),
  `iforest` (fixed `n_estimators`/`max_samples`), `n_trials` (XGBoost Optuna budget), `data_directory`.
Extend the `experiment` selector comment to mention `'timeseries'` and `'baselines'`. **No `pos_weight_scale`.**

## Task checklist

> **COMPLETE 2026-06-29.** All 7 files written + smoke-verified green on synthetic 90-col fixtures
> (workflows `wf_e0f79186-91d` build, cut off by session limit → resumed as `wf_63aee978-e28` for
> script+verify+review). Review findings triaged and fixed; re-verified green incl. on the actual `mps`
> device. **The migration code is done; the user's remaining step is the real-data run** (regenerate
> 90-col features via `measure_calculation.py` time_dynamic=True, then run `scripts/experiment_LSTM.py`).

- [x] **T1 — `src/data/sequence_data.py`** (CREATE). build_sequence_dataset/SequenceDataset/fit_scaler/
  apply_scaler/temporal_split. **Done note:** smoke-verified (shapes, internal-pad mask [T,F,T], scaler dedup
  77→45 cells, all 4 guards raise, real-T derivation AMLWorld=9/AMLSim=199/Tide=370). Cache key now includes T.
- [x] **T2 — `src/methods/losses.py`** (CREATE). make_loss bce/focal. **Done note:** smoke-verified (finite on toy batch; focal alpha replaces pos_weight).
- [x] **T3 — `src/methods/models.py`** (UPDATE: +PositionalEncoding/LSTMClassifier/TransformerClassifier).
  **Done note:** 4 existing classes byte-for-byte unchanged (difflib-confirmed); forward verified CPU **and mps**.
  Fixed: `enable_nested_tensor=False` on TransformerEncoder (MPS eval crash otherwise).
- [x] **T4 — `src/methods/evaluation.py`** (FILL). evaluate_scores/plot_curves/write_metrics. **Done note:** smoke-verified (png + metrics written, Okabe-Ito).
- [x] **T5 — `config/data/config.yaml`** (UPDATE). **Done note:** `sequence` block ×3 via anchors; parse-verified AMLWorld/AMLSim/Tide.
- [x] **T6 — `config/methods/config.yaml`** (UPDATE). **Done note:** `timeseries` block ×3 via anchors + `xgb_n_trials`; selector comment extended; all search-space specs suggest_param-valid.
- [x] **T7 — `scripts/experiment_LSTM.py`** (FILL). make_model/train_model/predict/run_xgb_baseline + `__main__`.
  **Done note:** full `__main__` run end-to-end via runpy (stubbed loader + fixtures) for nowcast AND forecast.
  Fixed: forecast XGBoost uses all K steps (K*90); `predict()` chunked (OOM-safe); `xgb_n_trials` config-driven;
  `task` in study/model/tuning names (no nowcast↔forecast collision).
- [x] **T8 — Smoke-verify** end-to-end. **Done note:** harness `scratchpad/smoke_timeseries.py` + `reverify_fixes.py`
  + `mps_check.py` all green; fixtures in scratch dir (real `results/features/` untouched).
- [x] **T9 — Adversarial review** (correctness / integration / faithfulness). **Done note:** see "Review findings" below.

## Review findings (T9) — all resolved

| # | Sev | Finding | Resolution |
|---|-----|---------|-----------|
| 1 | MAJOR | forecast XGBoost fed K−1 snapshots vs sequence models' K → unequal-history confound | `_xgb_arrays` forecast → `X.reshape(N,-1)` (K*90); contract wording corrected |
| 2 | MAJOR | Transformer eval-forward crashes on **MPS** (nested-tensor fast path) — the Mac's auto-selected device | `enable_nested_tensor=False`; verified on real mps |
| 3 | MAJOR | nowcast/forecast share model/tuning/study names → silent overwrite + SQLite study corruption | `task` threaded into study_name, model `.pt`, tuning JSON |
| 4 | MINOR | `.npz` cache key omitted `T` → stale cache after feature regen | `T` added to cache filename |
| 5 | NIT | XGBoost `n_trials` hardcoded 20 | now `xgb_n_trials` config key (default 20) |
| 6 | NIT | `predict()` single forward pass → OOM risk on full population | chunked over `batch_size` |
| — | NIT | `epochs` largely inert under early-stop; windowing double-loop slow at scale | comments added; windowing left as-is per reviewer (`.npz` cache mitigates; don't pre-optimize) |
| — | — | one bogus placeholder finding (`x.py`) from a malformed review agent | ignored; integration re-reviewed on final code |

## Known follow-ups (non-blocking)

- **Real-data run is the user's next step** (not done here per the synthetic-fixtures decision): regenerate
  90-col dynamic features (`measure_calculation.py`, `time_dynamic=True`) for the active dataset, then
  `python scripts/experiment_LSTM.py` from repo root with `SPaRTA_env`.
- If real-data build time bites, vectorize `build_sequence_dataset`'s per-snapshot `.loc` loop (reindex
  each snapshot frame onto the node set once). Only if profiling shows it matters.

## Verification (T8) — method

Real 90-col features do not exist on disk (only 19-col). Smoke-verify the new code on **synthetic
90-col fixtures** written to a scratch `features_dir`: for each dataset, `T = len(define_dates(real
transactions))`, write `T` fake `{type}_dynamic_{i}_features{_echo}.csv` (90 `keys_to_include` cols,
random rows, overlapping node sets so masks have structure + a planted internal-padding node) +
matching `_labels{_echo}.csv` (`Account,is_laundering` with a controlled positive rate). Then run
`build_sequence_dataset` → `temporal_split` → scaler → models forward/backward → a 1-trial Optuna →
XGBoost, for nowcast **and** forecast, asserting: correct `X/mask/y` shapes, no NaN/Inf in `X` or
logits, val & test each have ≥1 positive, guards raise on too-small `T`. This exercises every new
path (incl. the real per-dataset `T` derivation) without the hours-long feature extraction.

**Real-data run (prerequisite, separate):** regenerate 90-col features per dataset
(`measure_calculation.py` with `time_dynamic=True`), then run `scripts/experiment_LSTM.py`.

## Open decisions

- **[RESOLVED 2026-06-29] Verification scope = SYNTHETIC FIXTURES ONLY.** Validate all new code on
  generated 90-col fixtures for all 3 datasets (real `T` derivation via `define_dates`, nowcast +
  forecast, masking, split guards, 1-trial Optuna + XGBoost). Do **not** regenerate real features and
  do **not** touch `measure_calculation.py`. The user will run real feature regeneration + the real
  `experiment_LSTM.py` themselves later. The smoke harness lives in the scratchpad; fixtures live in a
  scratch dir (never `results/features/`, so the real 19-col files are untouched).
