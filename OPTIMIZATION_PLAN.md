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
7. Commit the change on the `optimisation` branch, **one commit per verified task**
   (message e.g. `optimise: vectorize currency conversion (Task 1)`), so each tick is a
   clean resume point and the `[~]` git-diff check works.

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

## How to verify (targeted / fast)

Each change is an equivalence transform, so verify by comparing the **new** output against
the **old** logic *in-process on the real data* — NOT by re-running the full pipeline (the
per-node loop is the slow part and is unaffected by Tier 1). Write a short throwaway check
that computes both ways and asserts the gap is float-noise.

- **Pass threshold:** max abs diff `< 1e-9` on all numeric outputs (datetimes/strings must
  match exactly). Run from the **repo root** in the conda env from `environment.txt`.
- **Datasets present locally:** IBM `HI-Small`, AMLSim, Tide `HI`/`LI` — verify the
  dataset(s) a task actually touches (e.g. the Tide currency change must be checked on Tide,
  not just IBM).
- **Existing baseline** `results/features/HI-Small_static_features_t.csv` (a prior full run)
  is a convenient *secondary* sanity check, but only trust it if no source change has
  happened since it was produced; the in-process old-vs-new comparison is the primary check.
- Per-task verify snippets are inline below. General shape:

```bash
# from repo root, in the conda env
python - <<'PY'
import numpy as np
# build OLD result (pre-change logic, inlined) and NEW result (current code), then:
# assert np.nanmax(np.abs(old - new)) < 1e-9   ;  print("OK")
PY
```

Config note: active run is `dataset: IBM`, `time_dynamic: False` (static branch) —
see [config/data/config.yaml](config/data/config.yaml).

---

## Tier 1 — biggest win, near-zero risk

### [x] 1. Vectorize currency conversion (hoist rate dicts + `.map`)

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

**Watch / required:** `.map` yields `NaN` for unknown currencies where the old code raised
`KeyError`. To preserve today's strict-fail behavior (user default), assert no NaN before
dividing, e.g. `assert not rate.isna().any(), "unknown currency in <dataset>"`.

**Verify (targeted):** in one script, load with the NEW code and recompute `amount` the
OLD way (per-row `convert_to_usd` with the same rate tables); assert
`np.nanmax(np.abs(old_amount - new_amount)) < 1e-9`. Do this for **both IBM and Tide**.

**Done note:** DONE 2026-06-08. Hoisted both dicts to module level, removed `convert_to_usd`
(only self-used), vectorized with `.map` + strict-fail `assert`. Verified old-vs-new on
SPaRTA_env (py3.12, pd2.3.3): IBM 4.49M rows **max abs diff 0.0**, Tide 7.59M rows **0.0**;
IBM loader runs end-to-end.
- **Blocking bug found & fixed (in scope):** `'US Dollar'` (1.9M rows, the most common
  currency) was missing from `EXCHANGE_NAME_TO_CODE` → both old and new code crashed. Added
  `"US Dollar": "USD"` (USD already = base rate 1.0). IBM loader was non-functional before this.
- **Separate pre-existing bug found (NOT fixed — see Separately noted):** the Tide loader's
  `pd.to_datetime(format='%Y-%m-%d %H:%M:%S')` crashes on fractional-second timestamps.
  Unrelated to currency; off the active path.

---

### [x] 2. Replace `iterrows()` graph build with `from_pandas_edgelist`

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

**Verify (targeted):** build the graph both ways (old `iterrows` loop vs new
`from_pandas_edgelist`) from the same aggregated frame; assert identical node set, edge
set, and per-edge `weight`/`amount_trans`/`num_trans`. No node loop needed.

**Done note:** DONE 2026-06-08. Rewrote `load_network` to `from_pandas_edgelist` (weight
column = decay if echo else 1). Verified on IBM HI-Small agg frame: 100k-edge sample →
node set match, edge set match, **max attr abs diff 0.0**; full build → 647,316 edges
(== unique agg rows) / 422,648 nodes (== distinct accounts).

---

## Tier 2 — the per-node Pool loop (runs once per node)

### [x] 3. Swap transaction adjacency to `to_numpy_array` (SPaRTA-side, safe)

**File:** [src/methods/neighbourhood_picture.py:57](src/methods/neighbourhood_picture.py:57)
(`transaction_measures`).

**Problem:** per node the ego-subgraph is densified **3×**: once in the FROZEN
`gargaml_node_measures` (leave it) and twice in `transaction_measures` (`amount_trans`,
`num_trans`). Each transaction build uses `nx.adjacency_matrix(...).toarray()` — a sparse
CSR intermediate that is then densified.

**Change (this task — minimal, low risk):** in `transaction_measures` only, swap
`nx.adjacency_matrix(G_ego_second, nodelist=nodes_ordered, weight=weight).toarray()` →
`nx.to_numpy_array(G_ego_second, nodelist=nodes_ordered, weight=weight)` (builds dense
directly, skips the CSR intermediate). Nothing else changes.

**Hard invariants — DO NOT change:**
- Returned dict keys `transaction_{weight}_summary_{agg}` for
  `weight ∈ {amount_trans, num_trans}`, `agg ∈ {sum, mean, max, std}`.
- The 9-element order `[00,01,02,10,11,12,20,21,22]` inside each list.
- CSV column order in
  [measure_calculation.py:79-86](src/methods/measure_calculation.py:79) stays identical.
- Leave `gargaml_node_measures` (FROZEN) and `node_selection` (FROZEN) untouched.

**Verify (targeted, subsample):** pick ~200 nodes from the reduced graph; for each, run
`node_measures` with OLD vs NEW `transaction_measures` and assert every `transaction_*`
value matches to `< 1e-9`. No full run needed.

**Done note:** DONE 2026-06-08. One-line swap in `transaction_measures`
(`adjacency_matrix().toarray()` → `to_numpy_array()`); frozen `gargaml_node_measures` and
`node_selection` untouched. Verified on 200 real community-pruned ego-subgraphs (both
weights): **max adjacency diff 0.0, max summary diff 0.0**.

### [ ] 3b. (OPTIONAL — deferred) Single-pass build of both transaction matrices

Build the `amount_trans` and `num_trans` matrices in **one** edge walk instead of two
`transaction_measures` calls. Larger speedup, but more invasive / higher risk. Same
invariants and verify as Task 3. **Do only after Task 3 is `[x]`, and only if the user
asks** — user chose "safe swap only" on 2026-06-08.

**Done note:** deferred per user (2026-06-08).

---

## Tier 3 — dynamic-path & low-priority

### [x] 4. Vectorize the echo time-decay

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
config), but **included per user decision (2026-06-08)** — a cheap win in core construction
code. `end_date` is a `pd.Timestamp` and the column is `datetime64[ns]`, so
`(end_date - col).dt.total_seconds()` is well-defined; check the sign matches the old
`np.datetime64(end_date) - np.datetime64(time_stamp)` (end minus timestamp, ≥ 0 in-window).

**Verify (targeted):** on the echo-filtered frame, compute `decay` both ways (old per-row
`exponential_time_decay` vs new vectorized) and assert `max abs diff < 1e-9`. No full run
needed; a dynamic+echo end-to-end run is an optional secondary check.

**Done note:** DONE 2026-06-08. Replaced the per-row `.apply` with a vectorized form,
replicating the old truncate-to-whole-seconds via `// pd.Timedelta(seconds=1)`. Helper
`exponential_time_decay` left in dates.py for reference (still imported). Verified on IBM
echo window (1.32M rows): **decay max abs diff 0.0**; echo loader runs end-to-end
(398,001 edges, weights 0.01–1.0).

### [ ] 5. (NOTE ONLY — likely skip) `graph_community` intra-community edge loop

**File:** [src/utils/graph_processing.py:78](src/utils/graph_processing.py:78)

The `for u, v in G.edges()` rebuild is O(E) Python but runs **once per snapshot**, not per
node. Inherently a networkx-construction loop. **Recommendation: leave it** unless
profiling shows it matters. No action by default.

---

## Bundled correctness fix (approved by user 2026-06-08)

### [x] 6. Align label-CSV filename between writer and readers

**Problem (confirmed live):** the feature extractor writes
`*_static_labels_t.csv` ([measure_calculation.py:150](src/methods/measure_calculation.py:150)),
but the experiment readers open `*_static_labels.csv` (no `_t`):
[experiment_features.py:24](scripts/experiment_features.py:24) and
[experiment_CNN.py:114](scripts/experiment_CNN.py:114). A *stale* `HI-Small_static_labels.csv`
(older run) is present on disk, so readers silently load OUTDATED labels instead of failing.

**Change:** point the two readers at the `_t` file the writer actually produces — change
`{...}_static_labels.csv` → `{...}_static_labels_t.csv` in both scripts. This keeps the
writer's `_t` convention (per CLAUDE.md) intact; do NOT make the writer drop `_t`.
`grep -rn "_static_labels" scripts/ src/` first to catch any other reader.

**Out-of-scope sibling (do NOT change here):** `NetworkImageDataset` in
[feature_data.py:59](src/data/feature_data.py:59) reads the same no-`_t` name but belongs to
the image/VGG pipeline — leave it; note it for a future image-pipeline pass.

**Verify:** `grep -rn "_static_labels.csv" scripts/` returns nothing (no-`_t` reads gone);
run `python scripts/experiment_features.py` from repo root and confirm it loads labels and
writes metrics to `results/experiments/`.

**Done note:** DONE 2026-06-08. Pointed both readers at `_static_labels_t.csv`. Verified:
no no-`_t` reads remain in `scripts/`; `data_prep_features` loads (train 338118×90, test
84530×90, pos rate 0.0056) and `experiment_CNN.load_data` merges (422648 rows). Skipped the
full training run (heavy) — the data-loading path is what the fix touches.
- **Caveat flagged to user:** the on-disk `HI-Small_static_labels_t.csv` is STALE (column
  `Is Laundering`, pre-canonicalization) — it predates the current loaders (which crashed on
  `US Dollar` before Task 1). Current `define_ML_labels` writes `is_laundering`, which
  `experiment_CNN.__main__` expects, so **regenerate features/labels with the fixed pipeline
  before training**. `experiment_features` is robust either way (positional rename).

---

## Out of scope (do NOT do here)

- Image/VGG16 renderer, `combine_pictures`, `NetworkImageDataset`, empty placeholders, notebooks.
- Frozen GARG-AML numerics (see Ground rules).

## Separately noted (cleanliness only — out of agreed scope, do NOT touch)

- **Pre-existing bug — FIXED 2026-06-08 (user-approved):** the Tide loader
  [Tide.py](src/data/utils/Tide.py) parsed timestamps with
  `pd.to_datetime(format='%Y-%m-%d %H:%M:%S')`, which crashed on the 14,828 fractional-second
  values (e.g. `...:15.446144`) in `generated_transactions_HI.csv` (7.59M rows total).
  Changed to `format='ISO8601'`; loader now reads all rows. Note: NOT truncated to seconds —
  `datetime64` is 8 bytes/value at any resolution, so that saves no memory, and the echo
  decay already truncates the lag to whole seconds anyway.
- Dead `dtrain` at [experiment_features.py:56](scripts/experiment_features.py:56)
  (built, never used) and a duplicate `unpack_batch` in
  [feature_data.py](src/data/feature_data.py) / [combine_pictures.py](src/data/utils/combine_pictures.py),
  plus hardcoded `num_files = 422`. Cleanliness only; not part of this work.
