# SPaRTA SLURM runbook — run + tune all models, with a timing sweep

All jobs are Slurm array scripts for VSC (preset to `--clusters=wice --account=lp_verbekelab`;
change those headers + the `miniconda3` path if your setup differs). They select what to run
via `SPARTA_*` env vars (see `src/utils/setup.py`: `resolve_dataset`, `resolve_dynamic`,
`resolve_timing`, `resolve_sequence`, `run_tag`) — **no edits to `config/*.yaml` are needed
for the sweep**. Run from the repo root. Logs land in `slurm/logs/` (created on submit).

## The sweep grid (edit in ONE place: `slurm/sweep_common.sh`)

- **Datasets:** AMLWorld, AMLSim, Tide.
- **Timing combos (expensive — one feature re-extraction each); tag = output namespace:**
  `d1_echo3`, `d1_echo7` (echo on, `days_echo` 3/7), `d1_w1`, `d1_w3` (echo off, window 1/3).
  All at daily `step=1`, so the snapshot count `T` is constant per dataset and the K-windows
  are comparable across combos.
- **K / task (free — re-windowed from the same snapshots):** `K ∈ {2,3,5}`, `task ∈ {nowcast, forecast}`.

`bash slurm/sweep_common.sh` prints the array ranges. After editing the grid, update each
script's `--array` range to match (the dynamic scripts source this file, so the grid itself
stays in lockstep — only the `#SBATCH --array=` line is hand-kept).

## What depends on what

- **Static models** (`experiment_features.py` = LogReg/XGB/NN, `experiment_CNN.py`) are
  **timing-independent** → static features extracted once.
- **Dynamic models** (`experiment_LSTM.py` = LSTM/Transformer, `experiment_baseline.py` =
  XGB/MLP/IsolationForest) read `results/features/<tag>/` → one run per (dataset × tag × K × task).

## Submission order

```bash
cd $VSC_DATA/SPaRTA

# 1. Feature extraction (the expensive stage) — static + dynamic can run in parallel.
EXTRACT_STATIC=$(sbatch --parsable slurm/extract_features.slurm)          # 3 tasks  (static)
EXTRACT_DYN=$(sbatch --parsable slurm/extract_features_dynamic.slurm)     # 12 tasks (dynamic sweep)

# 2. Training — each depends on its extraction finishing (afterok = only if extraction ok).
#    Static models depend on the static extraction:
sbatch --dependency=afterok:$EXTRACT_STATIC slurm/train_features.slurm    # 3 tasks  (CPU)
sbatch --dependency=afterok:$EXTRACT_STATIC slurm/train_cnn.slurm         # 3 tasks  (GPU)
#    Dynamic models depend on the dynamic extraction:
LSTM=$(sbatch --parsable --dependency=afterok:$EXTRACT_DYN slurm/train_lstm.slurm)       # 72 (GPU)
BASE=$(sbatch --parsable --dependency=afterok:$EXTRACT_DYN slurm/train_baseline.slurm)   # 72 (CPU)

# 3. Aggregate everything into results/experiments/summary.csv (afterany: don't let an
#    infeasible per-combo cell block the summary — it's reported as a gap instead).
sbatch --dependency=afterany:$LSTM:$BASE slurm/collect.slurm
```

`collect.slurm` also picks up the static `train_features` / `train_cnn` outputs if they have
finished; to force it strictly after those too, add their job ids to its `--dependency`.

Tips: append `%8` to a training script's `--array` (e.g. `0-71%8`) to cap concurrent tasks if
your GPU/CPU allocation is limited. Raise `--time=` or shrink `timeseries.n_trials` in
`config/methods/config.yaml` if the 12 h LSTM wall time is tight.

## Limited disk: extract one dataset at a time

`extract_features_sequential.slurm` runs the **static** extraction one dataset at a time
(`--array=0-2%1` + a symlink stage-in/out from an external `STAGE_ROOT`; `data/` stays
read-only — only a transient symlink is created and removed). For the dynamic sweep under the
same constraint, copy its staging block into a `%1`-throttled `extract_features_dynamic.slurm`.

## Optional VGG16 track (separate — AMLWorld HI-Small only, untuned)

Not part of the timing sweep: `visualising_network.py` hardcodes AMLWorld + a static graph,
and `experiment_pictures.py` has no HPO (fixed 10 epochs).

```bash
RENDER=$(sbatch --parsable slurm/render_pictures.slurm)                   # CPU, heavy
sbatch --dependency=afterok:$RENDER slurm/train_vgg16.slurm               # GPU, 10 epochs, no HPO
```

VGG16 writes only a model file (no metrics dump), so it does **not** appear in `summary.csv`.

## Outputs

- `results/features/<tag>/{type}_dynamic_{i}_features_echo.csv` — per-combo dynamic features.
- `results/features/{type}_static_features.csv` — static features (flat, shared).
- `results/timeseries/<tag>/…npz` — per-combo windowed cache (the tag kills the stale-cache bug).
- `results/experiments/{type}_{tag}_K{K}_{task}_{timeseries|baselines}.txt|.png` — per-combo metrics.
- `results/tuning/…_best_params.json` — best HPs per combo / model.
- `results/experiments/summary.csv` — the one tidy table for the effect-of-timing analysis.
