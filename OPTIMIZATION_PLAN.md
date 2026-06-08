# SPaRTA — Hot-Path Optimization Plan

> **Purpose:** step-by-step checklist for optimizing the SPaRTA hot path. Each task is
> self-contained (file, problem, exact change, verification) so it can be actioned in
> isolation without re-reading the whole codebase. Tick boxes as you go.

## For Claude: how to use and maintain this file

**On entry (start of any session that touches this work):**
1. Read **this section**, then **Ground rules**, then the **first task whose box is not
   `[x]`**. You do NOT need to read the whole file or the whole codebase — each task is
   self-contained. Do tasks top-to-bottom (Tier 1 → 2 → 3) unless the user names one.
2. If a task is `[~]` (in progress), assume a prior session was interrupted — re-read that
   task and check the working tree (`git diff`) before continuing, to avoid double-applying.

**While doing a task:**
3. Set its box to `[~]` before you start editing.
4. Make ONLY the change described. Respect the FROZEN list in Ground rules. If reality
   differs from the task description (line moved, code already changed), update the task
   text to match what you actually did — keep the file truthful.
5. Run the **verify** step for that task. Do not mark done on an unverified change.

**On finishing a task:**
6. Set the box to `[x]`, fill the `Done note:` line with a one-line summary + any timing
   (e.g. "vectorized; load time 240s→9s; max diff 3e-13"). If verification revealed a new
   constraint or follow-up, add it as a sub-bullet under that task.

**Updating the plan itself (allowed and expected):**
- You MAY add, split, reorder, or remove tasks if the work demands it — but say so in your
  reply to the user and note it in the affected task. Prefer editing in place over rewriting.
- Never silently delete a task; if dropping one, change its box to `[x]` or `[ ]` and add a
  `Done note:` explaining why (e.g. "skipped — superseded by Task 3").
- Keep the **Out of scope** and **Separately noted** sections intact unless the user
  re-scopes; moving an item OUT of those requires explicit user approval.
- This file is the source of truth for progress. Update it in the same turn you make the
  code change, so a later session can resume from it alone.

## Ground rules (agreed with user)

- **Scope:** hot path only — data load → network construction → per-node feature
  extraction. **Do NOT touch** the image/VGG16 renderer
  ([visualising_network.py](src/methods/visualising_network.py),
  [combine_pictures.py](src/data/utils/combine_pictures.py),
  `NetworkImageDataset`), the empty placeholders, or notebooks.
- **Reproducibility:** equivalent-within-tolerance is OK. Vectorization may shift floats
  at ~1e-12 (noise). Outputs must stay equivalent — no feature/label/metric should
  meaningfully change.
- **Goal:** max speed, **minimal diff**. Each change is a mechanical equivalence transform.
- **FROZEN — never edit numerics of:** the nine `measure_XY_function`s in
  [measure_functions.py](src/methods/utils/measure_functions.py), `node_selection` in
  [neighbourhood_functions.py](src/methods/utils/neighbourhood_functions.py), and
  `gargaml_node_measures` in [neighbourhood_picture.py](src/methods/neighbourhood_picture.py).
  `transaction_measures` IS SPaRTA-side and freely editable.

## Status legend

- `[ ]` not started · `[~]` in progress · `[x]` done & verified
- When you finish a task: tick the box, fill in the **Done note** line, and (if helpful)
  record before/after timing.

## How to verify (general)

Run the active pipeline from the **repo root** and confirm it completes and that the
output CSV is unchanged vs a pre-change baseline:

```bash
# from repo root, BEFORE changes — capture a baseline once:
python -m src.methods.measure_calculation
cp results/features/HI-Small_static_features_t.csv /tmp/baseline_features_t.csv

# AFTER each change — re-run and diff (expect identical, or differences only at ~1e-12):
python -m src.methods.measure_calculation
python - <<'PY'
import pandas as pd, numpy as np
a = pd.read_csv('/tmp/baseline_features_t.csv').sort_values('node').reset_index(drop=True)
b = pd.read_csv('results/features/HI-Small_static_features_t.csv').sort_values('node').reset_index(drop=True)
num = a.select_dtypes('number').columns
print('max abs diff:', np.nanmax(np.abs(a[num].values - b[num].values)))
print('shapes:', a.shape, b.shape)
PY
```

Config note: active run is `dataset: IBM`, `time_dynamic: False` (static branch) —
see [config/data/config.yaml](config/data/config.yaml).

---

## Tier 1 — biggest win, near-zero risk

### [ ] 1. Vectorize currency conversion (hoist rate dicts + `.map`)

**Files:** [src/data/utils/IBM.py](src/data/utils/IBM.py:11),
[src/data/utils/Tide.py](src/data/utils/Tide.py:11)

**Problem:** `convert_to_usd` rebuilds both rate dictionaries on *every call*, and is
called once per row inside a Python list comprehension over the full raw transaction
table (millions of rows). Largest single data-load cost.

**Change (IBM.py):**
- Move `EXCHANGE_NAME_TO_CODE` and `USD_EXCHANGE_RATES` out of `convert_to_usd` to
  module-level constants (built once at import). Keep the exact same rate values.
- Replace the list comprehension at [IBM.py:74](src/data/utils/IBM.py:74):
  ```python
  transactions['amount'] = (
      transactions['amount']
      / transactions['Payment Currency'].map(EXCHANGE_NAME_TO_CODE).map(USD_EXCHANGE_RATES)
  )
  ```
- `convert_to_usd` can stay (unused) or be removed — prefer removing it to keep things light,
  but only if nothing else imports it (grep first).

**Change (Tide.py):** same pattern at [Tide.py:46](src/data/utils/Tide.py:46):
```python
transactions['amount'] = transactions['amount'] / transactions['currency'].map(USD_EXCHANGE_RATES)
```
(Tide is a single-step name→rate map; no name-to-code table.)

**Watch:** `.map` yields `NaN` for unknown currencies where the old code raised `KeyError`.
For valid data this is identical. If you want to preserve the strict-fail behavior, add an
assert that the mapped rate has no NaN.

**Verify:** general verify above; `amount` column must match baseline to float noise.

**Done note:** _____

---

### [ ] 2. Replace `iterrows()` graph build with `from_pandas_edgelist`

**File:** [src/data/network_data_loader.py:36](src/data/network_data_loader.py:36) (`load_network`)

**Problem:** building the `DiGraph` edge-by-edge via `transactions.iterrows()` is the
slowest possible path (a Series allocated per edge).

**Change:** rewrite `load_network` to assign a `weight` column then build in one call:
```python
def load_network(transactions, echo=False):
    edges = transactions.copy()
    edges['weight'] = edges['decay'] if echo else 1
    return nx.from_pandas_edgelist(
        edges, 'from_account', 'to_account',
        edge_attr=['weight', 'amount_trans', 'num_trans'],
        create_using=nx.DiGraph,
    )
```

**Why safe:** edges are unique post-`groupby` and self-loops are already dropped in the
loaders, so there is no dedup/ordering subtlety. Edge attributes are identical.

**Verify:** general verify above. Optionally assert `G.number_of_edges()` and
`G.number_of_nodes()` match a pre-change run.

**Done note:** _____

---

## Tier 2 — the per-node Pool loop (runs once per node)

### [ ] 3. Collapse the two transaction adjacency builds (SPaRTA-side)

**File:** [src/methods/neighbourhood_picture.py:57](src/methods/neighbourhood_picture.py:57)
(`transaction_measures`) and its caller `node_measures`
([neighbourhood_picture.py:82](src/methods/neighbourhood_picture.py:82)).

**Problem:** per node, the same ego-subgraph is densified **3×**: once in the FROZEN
`gargaml_node_measures` (leave it), and twice in `transaction_measures` — once for
`amount_trans`, once for `num_trans`. Each uses `nx.adjacency_matrix(...).toarray()`
(sparse CSR intermediate, then densify).

**Change (two parts, both SPaRTA-side — frozen code untouched):**
1. Swap `nx.adjacency_matrix(G, nodelist=..., weight=w).toarray()` →
   `nx.to_numpy_array(G, nodelist=..., weight=w)` (builds dense directly, skips the CSR).
2. Build the `amount_trans` and `num_trans` matrices in a **single pass** instead of two
   separate `transaction_measures` calls — i.e. walk the subgraph's edges once and fill
   both dense matrices, then run the nine `summary_XY_function`s on each.

**Hard invariants — DO NOT change:**
- The returned dict keys: `transaction_{weight}_summary_{agg}` for
  `weight ∈ {amount_trans, num_trans}`, `agg ∈ {sum, mean, max, std}`.
- The 9-element order `[00,01,02,10,11,12,20,21,22]` inside each list.
- Therefore the CSV column order in
  [measure_calculation.py:79-86](src/methods/measure_calculation.py:79) stays identical.

**Verify:** general verify above — every `transaction_*` column must match baseline to
float noise. This is the most error-prone task; diff carefully.

**Done note:** _____

---

## Tier 3 — optional / dynamic-path only

### [ ] 4. Vectorize the echo time-decay

**File:** [src/data/network_data_loader.py:73](src/data/network_data_loader.py:73)
(echo branch of `load_network_time`); helper
[src/data/utils/dates.py:38](src/data/utils/dates.py:38) (`exponential_time_decay`).

**Problem:** `.apply(lambda x: exponential_time_decay(x, end_date, ...))` runs per row.
The math is pure NumPy datetime arithmetic and vectorizes over the whole Series.

**Change:** compute decay on the column directly, e.g.:
```python
gamma = -np.log(0.01) / days_echo
delta_days = (end_date - transactions_time_filtered['timestamp']).dt.total_seconds() / 86400
transactions_time_filtered['decay'] = np.exp(-gamma * delta_days)
```
Keep `exponential_time_decay` if used elsewhere (grep); otherwise it can stay as-is.

**Note:** only runs in `time_dynamic: True` + `echo: True` mode (NOT the active static
config). Include only if you want the dynamic path covered.

**Verify:** run with `time_dynamic: True`, `echo: True` and diff a `*_dynamic_*_echo_t.csv`.

**Done note:** _____

### [ ] 5. (NOTE ONLY — likely skip) `graph_community` intra-community edge loop

**File:** [src/utils/graph_processing.py:78](src/utils/graph_processing.py:78)

The `for u, v in G.edges()` rebuild is O(E) Python but runs **once per snapshot**, not per
node. Inherently a networkx-construction loop. **Recommendation: leave it** unless
profiling shows it matters. No action by default.

---

## Out of scope (do NOT do here)

- Image/VGG16 renderer, `combine_pictures`, `NetworkImageDataset`, empty placeholders, notebooks.
- Frozen GARG-AML numerics (see Ground rules).

## Separately noted (NOT optimizations — confirm with user before touching)

- **Potential bug:** [measure_calculation.py:150](src/methods/measure_calculation.py:150)
  writes `*_static_labels_t.csv`, but readers
  [experiment_features.py:24](scripts/experiment_features.py:24) and
  [experiment_CNN.py:114](scripts/experiment_CNN.py:114) read `*_static_labels.csv`
  (no `_t`). Filename mismatch → readers may not find the labels.
- Dead `dtrain` at [experiment_features.py:56](scripts/experiment_features.py:56)
  (built, never used) and a duplicate `unpack_batch` in
  [feature_data.py](src/data/feature_data.py) / [combine_pictures.py](src/data/utils/combine_pictures.py),
  plus hardcoded `num_files = 422`. Cleanliness only; out of the agreed scope.
