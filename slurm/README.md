# Slurm job scripts for VSC

These scripts run the SPaRTA pipeline as **job arrays**, one array task per dataset,
so you can launch every dataset at once instead of editing the YAML between runs.

## How the dataset is selected

The scripts export `SPARTA_DATASET` (and optionally `SPARTA_EXPERIMENT`) per array
task. The Python entry points read these env vars via `resolve_dataset` /
`resolve_experiment` in [src/utils/setup.py](../src/utils/setup.py) and fall back to
`config/data/config.yaml` / `config/methods/config.yaml` when they are unset. So:

- **Local, single run:** just run the script as before — it uses the YAML values.
- **VSC sweep:** the array task sets the env var; no YAML edit needed.

The `DATASETS` bash array near the top of each script is the sweep list. The array
range (`--array=0-2`) must match its length (3 datasets → `0-2`).

## Before you submit — edit the header placeholders

Every script has a block marked `# <<< EDIT >>>`. Fill in for your VSC cluster:

- `--account=` your VSC credit account (e.g. `lp_...` / `intro_...`).
- `--clusters=` and `--partition=` (e.g. wICE: `--clusters=wice --partition=gpu`;
  Genius: `--clusters=genius --partition=gpu_p100`). Check `sinfo` / VSC docs.
- `--gpus-per-node=` / `--gres=gpu:1` syntax — clusters differ; use what your cluster expects.
- The `module load` lines and the conda activation path (`CONDA_SH`, `ENV_NAME`).

## Submit

```bash
# from the repo root
sbatch slurm/extract_features.slurm     # CPU: build 3x3 features for all datasets (parallel array)
sbatch slurm/train_features.slurm       # CPU: LogReg + XGBoost + NN on the features
sbatch slurm/train_cnn.slurm            # GPU: CNN + Optuna HPO on the 3x3 pictures
sbatch slurm/train_baseline.slurm       # CPU: XGBoost + IsolationForest on the K-snapshot windows
sbatch slurm/train_lstm.slurm           # GPU: LSTM + Transformer + Optuna HPO on the K-snapshot windows
```

`extract_features` must finish before the training jobs (they read
`results/features/`). For the time-series scripts you also need
`time_dynamic: True` (+ `echo: True`) in `config/data/config.yaml` so the
per-snapshot dynamic CSVs get written.

## Limited disk: extract one dataset at a time

`extract_features.slurm` uses `--array=0-2` (all three tasks run **in parallel**),
so all three raw datasets must sit under `data/` at once. If you cannot hold them
simultaneously, use **`extract_features_sequential.slurm`** instead:

```bash
sbatch slurm/extract_features_sequential.slurm   # CPU: build features, one dataset at a time
```

It differs in two ways:

- `--array=0-2%1` — the `%1` throttle runs a single dataset task at a time, so
  only one dataset is on disk at any moment.
- A **symlink** stage-in/stage-out: each task links one dataset's raw files from an
  external archive into `data/<subdir>`, extracts, then removes the link. Because
  `data/` is read-only (no vendor files or artefacts are ever written there — only
  a transient symlink), this respects the project's hard rule. Set `STAGE_ROOT` to
  your archive path (datasets laid out as `AMLWorld/`, `amlsim/`, `Tide/`).

If you'd rather stage manually, leave `STAGE_ROOT=""`: put one dataset under
`data/<subdir>/` yourself, `sbatch` the script (or just run
`SPARTA_DATASET=<name> python src/methods/measure_calculation.py`), swap in the
next dataset, repeat. Feature CSVs are namespaced by dataset
(`results/features/HI-Small_*`, `AMLSim_*`, `HI_*`), so they never collide and the
downstream training/analysis stays per-dataset.

## Running the time-series experiment and its baselines in parallel

`train_lstm.slurm` and `train_baseline.slurm` consume the **same** windowed
dataset (`build_sequence_dataset` reads `config/data/config.yaml` -> `sequence`
identically in both scripts). Their output filenames are disjoint
(`*_timeseries_*` vs `*_baselines_*`), so they can safely run **at the same time**
on different cluster resources — GPU for the sequence models, CPU for the
baselines. Both depend only on `extract_features.slurm` having finished.

```bash
# 1) extract once (writes results/features/*_dynamic_*_features_echo.csv)
jid=$(sbatch --parsable slurm/extract_features.slurm)

# 2) fan out the two training jobs concurrently (each is itself a 3-task array
#    over AMLWorld/AMLSim/Tide -> 6 array tasks running in parallel total).
sbatch --dependency=afterok:$jid slurm/train_lstm.slurm
sbatch --dependency=afterok:$jid slurm/train_baseline.slurm
```

You can submit the two `sbatch` lines in either order; Slurm schedules them
independently, and the LSTM job sits in the GPU queue while the baseline job
sits in the CPU queue. Each script's array (`--array=0-2`) launches its three
dataset tasks in parallel automatically.

If the LSTM array's 12 h wall time is too tight for the full population on your
cluster, raise `--time=` in `slurm/train_lstm.slurm` or shrink the Optuna budget
via `method_config[dataset].timeseries.n_trials`. The baselines side is
typically much faster (small XGBoost search + IsolationForest fit), hence
the shorter 2 h default.

Logs land in `slurm/logs/` (created on submit).
