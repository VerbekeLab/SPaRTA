# Configuration

Two YAML files drive the pipeline. Load them with `src.utils.setup.load_config`.
Both are **indexed by the active `dataset`** — every dataset you use needs its own block
or downstream `config[dataset]` lookups will `KeyError`.

| File | Controls | Read by |
|------|----------|---------|
| [`data/config.yaml`](data/config.yaml) | which dataset, how the network/snapshots are built | `measure_calculation.py`, all experiment scripts |
| [`methods/config.yaml`](methods/config.yaml) | which experiment runs, where features live | the `scripts/experiment_*.py` |

---

## `data/config.yaml`

### `parameters` — global switches
| Key | Meaning | Values |
|-----|---------|--------|
| `dataset` | Active dataset. Selects the block below and the loader. | `IBM`, `AMLSim`, `Tide` |
| `time_dynamic` | `False` → one static network. `True` → a sequence of time-windowed snapshots. | `True` / `False` |

### `<dataset>` — per-dataset block (keyed by the active `dataset`)
| Key | Meaning | Read by |
|-----|---------|---------|
| `type_dataset` | Variant passed to the loader; also the output-file prefix. e.g. `HI-Small` (IBM), `AMLSim`, `HI` (Tide → `data/Tide/generated_transactions_HI.csv`). | loaders, output filenames |
| `n_channels` | Feature channels per node, reshaped to `(n_channels, 3, 3)` for the CNN. | `experiment_CNN.py` only |

### `<dataset>.network_construction` — **only used when `time_dynamic: True`**
| Key | Meaning |
|-----|---------|
| `time_type` | Time unit for the windows: `hours`, `days`, `weeks`, `months`. |
| `time_step` | Stride between consecutive snapshots (in `time_type` units). |
| `time_width` | Width of each snapshot window (in `time_type` units). `step < width` ⇒ overlapping windows. |
| `echo` | If `True`, weight edges by exponential time-decay toward the window end (recent = heavier). |
| `days_echo` | Echo half-life control. When `echo: True` the window is forced to `[end − days_echo, end]` and decay reaches 0.01 at `days_echo` (`γ = −ln(0.01)/days_echo`). |

---

## `methods/config.yaml`

| Key | Meaning |
|-----|---------|
| `experiment` | Selects which `<dataset>.<experiment>` block to read for `data_directory`. Comment lists `features_NN` / `features_XGBoost`. |
| `<dataset>.<experiment>.data_directory` | Where `measure_calculation.py` wrote the feature/label CSVs. **This is the only HP block value the code currently reads.** |

### ⚠️ Currently-ignored keys
The remaining hyperparameters under each experiment block are **not wired up**:

- NN (`num_layers`, `hidden_size`, `output_size`, `n_epochs`, `learning_rate`) and
  XGBoost (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`)
  are **hardcoded** in [`scripts/experiment_features.py`](../scripts/experiment_features.py),
  not loaded from this file.
- `experiment_features.py` always runs **all three** models (LogReg + XGBoost + NN);
  the `experiment` value does not switch between them.
- `experiment_CNN.py` reads only `n_channels` from the data config; its HPs come from Optuna.

So today, editing those numbers here has **no effect**. See the note below.

---

## Where to change things

| I want to… | Edit |
|------------|------|
| Switch dataset | `data/config.yaml → parameters.dataset` (add a matching block in **both** files) |
| Static vs. snapshot features | `data/config.yaml → parameters.time_dynamic` |
| Resize/re-stride snapshot windows | `data/config.yaml → <dataset>.network_construction.time_*` |
| Turn time-decay on/off | `data/config.yaml → <dataset>.network_construction.echo` / `days_echo` |
| Change the CNN input channel count | `data/config.yaml → <dataset>.n_channels` |
| Point experiments at a feature dir | `methods/config.yaml → <dataset>.<experiment>.data_directory` |
| Change NN/XGBoost HPs | **not via config yet** — edit `scripts/experiment_features.py` directly |

---

## My take

The data config is clean and fully wired. The methods config, by contrast, is mostly
**aspirational**: it advertises model hyperparameters that the scripts ignore, which is
misleading for anyone tuning from here. Two ways to fix the gap (recommended, not done):

1. **Make the config authoritative (preferred).** In `experiment_features.py`, replace the
   hardcoded `hidden_size`/`num_layers`/`n_epochs`/… with reads from `experiment_params`,
   and let `experiment` actually select which model(s) run. One config, one source of truth.
2. **Or trim the config** to just `data_directory` until the HPs are wired, so it stops
   promising knobs that do nothing.

Until then this README documents reality (what the code reads) rather than intent.
