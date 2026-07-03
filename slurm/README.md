# SPaRTA SLURM runbook — run + tune all models, with a timing sweep

All jobs are Slurm scripts for VSC (preset to `--clusters=wice --account=lp_verbekelab`;
change those headers + the `miniconda3` path if your setup differs). They select what to run
via `SPARTA_*` env vars (see `src/utils/setup.py`: `resolve_dataset`, `resolve_dynamic`,
`resolve_timing`, `resolve_sequence`, `run_tag`) — **no edits to `config/*.yaml` are needed
for the sweep**. Run from the repo root. Logs land in `slurm/logs/` (created on submit).

## One dataset per submission

Every extraction/training script takes the **dataset as its submission argument**, so the
three datasets run as separate jobs (one whole-sweep submission was too long for the VSC
wall-time limits):

```bash
sbatch slurm/extract_features_dynamic.slurm AMLWorld
sbatch slurm/extract_features_dynamic.slurm AMLSim
sbatch slurm/extract_features_dynamic.slurm Tide
```

Valid names: `AMLWorld`, `AMLSim`, `Tide` (the `DATASETS` list in `sweep_common.sh`).
Omitting or misspelling the argument fails the job immediately with a usage message.
Alternatively pass it as an env var: `sbatch --export=ALL,SPARTA_DATASET=Tide <script>`.
Each dataset is fully independent — submit them all at once, on different days, or only
re-run the one that failed.

## The sweep grid (edit in ONE place: `slurm/sweep_common.sh`)

- **Datasets:** AMLWorld, AMLSim, Tide — chosen per submission, NOT part of the arrays.
- **Per-dataset grids** — each dataset sweeps around its own defaults from
  `config/data/config.yaml`, because the spans and laundering-pattern durations differ
  (measured in `notebooks/pattern_durations.ipynb`). Timing combos are the expensive axis
  (one feature re-extraction each; tag = output namespace); K/task are free (re-windowed
  from the same snapshots):

  | Dataset  | Snapshot grid | Timing combos (tags)                        | K            |
  |----------|---------------|---------------------------------------------|--------------|
  | AMLWorld | 6-hour step   | `h6_echo1`, `h6_echo2`, `h6_w6`, `h6_w24`   | 7, **13**, 17 |
  | AMLSim   | daily step    | `d1_echo3`, `d1_echo7`, `d1_w1`, `d1_w3`    | 3, **7**, 9  |
  | Tide     | 2-day step    | `d2_echo3`, `d2_echo7`, `d2_w2`, `d2_w3`    | 3, **5**, 7  |

  Bold = the YAML default. `task ∈ {nowcast, forecast}` for all. Within a dataset the
  snapshot count `T` is constant across its combos, so its K-windows stay comparable.
  The val/test bands (`n_val/n_test_anchors`) are not swept — they come from the YAML.
- **Axis sizes are identical across datasets** (4 timing × 3 K × 2 tasks), so one
  `--array` range fits all three.

`bash slurm/sweep_common.sh` prints the per-dataset array ranges. After editing the grid,
update each script's `--array` range to match (the dynamic scripts source this file, so the
grid itself stays in lockstep — only the `#SBATCH --array=` line is hand-kept):

- `extract_features_dynamic.slurm`: `0-3` (timing combos)
- `train_lstm.slurm` / `train_baseline.slurm`: `0-23` (timing × K × task)
- `extract_features.slurm`, `train_features.slurm`, `train_cnn.slurm`: no array (one job = one dataset)

## What depends on what

- **Static models** (`experiment_features.py` = LogReg/XGB/NN, `experiment_CNN.py`) are
  **timing-independent** → static features extracted once per dataset.
- **Dynamic models** (`experiment_LSTM.py` = LSTM/Transformer, `experiment_baseline.py` =
  XGB/MLP/IsolationForest) read `results/features/<tag>/` → one run per (tag × K × task),
  per dataset.

## Submission order (per dataset)

```bash
cd $VSC_DATA/SPaRTA
DS=AMLWorld           # then repeat the block with AMLSim, Tide

# 1. Feature extraction (the expensive stage) — static + dynamic can run in parallel.
EXTRACT_STATIC=$(sbatch --parsable slurm/extract_features.slurm $DS)          # 1 task  (static)
EXTRACT_DYN=$(sbatch --parsable slurm/extract_features_dynamic.slurm $DS)     # 4 tasks (dynamic sweep)

# 2. Training — each depends on ITS dataset's extraction (afterok = only if extraction ok).
#    Static models depend on the static extraction:
sbatch --dependency=afterok:$EXTRACT_STATIC slurm/train_features.slurm $DS    # 1 task   (CPU)
sbatch --dependency=afterok:$EXTRACT_STATIC slurm/train_cnn.slurm $DS         # 1 task   (GPU)
#    Dynamic models depend on the dynamic extraction:
LSTM=$(sbatch --parsable --dependency=afterok:$EXTRACT_DYN slurm/train_lstm.slurm $DS)     # 24 (GPU)
BASE=$(sbatch --parsable --dependency=afterok:$EXTRACT_DYN slurm/train_baseline.slurm $DS) # 24 (CPU)
```

Once the training jobs of **all datasets you care about** have finished, aggregate
everything into `results/experiments/summary.csv` (afterany: don't let an infeasible
per-combo cell block the summary — it's reported as a gap instead):

```bash
# collect all LSTM/baseline job ids across the datasets you ran, e.g.:
sbatch --dependency=afterany:$LSTM_AMLWORLD:$BASE_AMLWORLD:$LSTM_AMLSIM:$BASE_AMLSIM:$LSTM_TIDE:$BASE_TIDE slurm/collect.slurm
```

`collect.slurm` needs no dataset argument — it sweeps whatever per-combo outputs exist in
`results/`, so you can also run it after each dataset for a partial summary; missing cells
are reported as gaps. It also picks up the static `train_features` / `train_cnn` outputs if
they have finished; to force it strictly after those too, add their job ids to its
`--dependency`.

Tips: append `%8` to a training script's `--array` (e.g. `0-23%8`) to cap concurrent tasks if
your GPU/CPU allocation is limited. Raise `--time=` or shrink `timeseries.n_trials` in
`config/methods/config.yaml` if the 12 h LSTM wall time is tight.

## Limited disk: stage one dataset at a time

`extract_features_sequential.slurm` is the static extraction with a symlink stage-in/out
from an external `STAGE_ROOT` (`data/` stays read-only — only a transient symlink is created
and removed). It takes the dataset argument like the others; chain submissions with
`--dependency=afterany:<prev>` to guarantee only one dataset is on disk at a time (the
header shows the exact three-liner). For the dynamic sweep under the same constraint, copy
its staging block into `extract_features_dynamic.slurm` and append `%1` to that array.

## Optional VGG16 track (separate — AMLWorld HI-Small only, untuned)

Not part of the timing sweep: `visualising_network.py` hardcodes AMLWorld + a static graph,
and `experiment_pictures.py` has no HPO (fixed 10 epochs). No dataset argument.

```bash
RENDER=$(sbatch --parsable slurm/render_pictures.slurm)                   # CPU, heavy
sbatch --dependency=afterok:$RENDER slurm/train_vgg16.slurm               # GPU, 10 epochs, no HPO
```

VGG16 writes only a model file (no metrics dump), so it does **not** appear in `summary.csv`.

## Outputs

- `results/features/<tag>/{type}_dynamic_{i}_features_echo.parquet` — per-combo dynamic
  features (`_w` combos drop the `_echo` suffix; readers also accept legacy `.csv`).
- `results/features/{type}_static_features.parquet` — static features (flat, shared).
- `results/timeseries/<tag>/…npz` — per-combo windowed cache (the tag kills the stale-cache bug).
- `results/experiments/{type}_{tag}_K{K}_{task}_{timeseries|baselines}.txt|.png` — per-combo metrics.
- `results/tuning/…_best_params.json` — best HPs per combo / model.
- `results/experiments/summary.csv` — the one tidy table for the effect-of-timing analysis.
