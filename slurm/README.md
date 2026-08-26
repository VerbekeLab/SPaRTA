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
- `build_windows.slurm`: `0-3` (timing combos — K/task are free re-slices of one store)
- `train_lstm.slurm` / `train_baseline.slurm` / `train_baseline_nn.slurm`: `0-23` (timing × K × task)
- `extract_features.slurm`, `train_features.slurm`, `train_cnn.slurm`: no array (one job = one dataset)

## What depends on what

- **Static models** (`experiment_features.py` = LogReg/XGB/NN, `experiment_CNN.py`) are
  **timing-independent** → static features extracted once per dataset.
- **Dynamic models** (`experiment_LSTM.py` = LSTM/Transformer, `experiment_baseline.py` =
  XGB/MLP/IsolationForest) read `results/features/<tag>/` → one run per (tag × K × task),
  per dataset. The MLP is split into its own GPU script, `train_baseline_nn.slurm` — on
  these window sizes (millions of rows) mini-batch training is far slower per Optuna trial
  than XGBoost's single tree fit, and needs the GPU + bigger batches that `train_baseline.slurm`
  (CPU, XGBoost + IsolationForest only) doesn't provide. See `train_baseline_nn.slurm`'s header.
  LSTM and Transformer share the same GPU resources, so `train_lstm.slurm` tunes both
  sequentially by default; pass `lstm` or `transformer` as a 2nd arg (or set
  `SPARTA_SEQ_MODELS`) to fan them into separate concurrent tasks instead — see that
  script's header.
- **Stage 1.5, `build_windows.slurm`** sits between them. The stacked snapshot block a
  window build needs depends on the **tag alone**, not on K or the task, so it is built once
  per tag (4 CPU tasks) into a prebuilt *window store* (`src/data/window_store.py`); each of
  that tag's 6 (K × task) training cells then derives its own windows from it by index
  arithmetic in seconds. Without it every cell re-read the feature files and re-assembled the
  block itself — ~40 min of single-threaded pandas, **two of the six on a GPU node**. The GPU
  scripts therefore set `SPARTA_REQUIRE_WINDOWS=1`: a missing store fails them in seconds
  instead of quietly paying that cost with an A100 idle. Store location follows
  `resolve_window_store_dir` (`$VSC_SCRATCH/SPaRTA/windows/<tag>` on VSC);
  `SPARTA_WINDOW_STORE=off` disables it and restores the old in-job build.

## Submission order (per dataset)

```bash
cd $VSC_DATA/SPaRTA
DS=AMLWorld           # then repeat the block with AMLSim, Tide

# 1. Feature extraction (the expensive stage) — static + dynamic can run in parallel.
EXTRACT_STATIC=$(sbatch --parsable slurm/extract_features.slurm $DS)          # 1 task  (static)
EXTRACT_DYN=$(sbatch --parsable slurm/extract_features_dynamic.slurm $DS)     # 4 tasks (dynamic sweep)

# 1.5 Window store — one per tag, on CPU. Every dynamic training cell reads this instead of
#     re-assembling windows from the feature files (~40 min each, two of them on a GPU).
WINDOWS=$(sbatch --parsable --dependency=afterok:$EXTRACT_DYN slurm/build_windows.slurm $DS)  # 4 tasks (CPU)

# 2. Training — each depends on ITS dataset's extraction (afterok = only if extraction ok).
#    Static models depend on the static extraction:
sbatch --dependency=afterok:$EXTRACT_STATIC slurm/train_features.slurm $DS    # 1 task   (CPU)
sbatch --dependency=afterok:$EXTRACT_STATIC slurm/train_cnn.slurm $DS         # 1 task   (GPU)
#    Dynamic models depend on the WINDOW STORE (which already depends on the extraction):
LSTM=$(sbatch --parsable --dependency=afterok:$WINDOWS slurm/train_lstm.slurm $DS)        # 24 (GPU)
BASE=$(sbatch --parsable --dependency=afterok:$WINDOWS slurm/train_baseline.slurm $DS)    # 24 (CPU: XGBoost + IsolationForest)
BASE_NN=$(sbatch --parsable --dependency=afterok:$WINDOWS slurm/train_baseline_nn.slurm $DS) # 24 (GPU: MLP)
```

Once the training jobs of **all datasets you care about** have finished, aggregate
everything into `results/experiments/summary.csv` (afterany: don't let an infeasible
per-combo cell block the summary — it's reported as a gap instead):

```bash
# collect all LSTM/baseline job ids across the datasets you ran, e.g.:
sbatch --dependency=afterany:$LSTM_AMLWORLD:$BASE_AMLWORLD:$BASE_NN_AMLWORLD:$LSTM_AMLSIM:$BASE_AMLSIM:$BASE_NN_AMLSIM:$LSTM_TIDE:$BASE_TIDE:$BASE_NN_TIDE slurm/collect.slurm
```

`collect.slurm` needs no dataset argument — it sweeps whatever per-combo outputs exist in
`results/`, so you can also run it after each dataset for a partial summary; missing cells
are reported as gaps. It also picks up the static `train_features` / `train_cnn` outputs if
they have finished; to force it strictly after those too, add their job ids to its
`--dependency`.

Tips: append `%8` to a training script's `--array` (e.g. `0-23%8`) to cap concurrent tasks if
your GPU/CPU allocation is limited. Raise `--time=` or shrink `timeseries.n_trials` in
`config/methods/config.yaml` if the 24 h LSTM wall time is tight.

## Failed / timed-out tasks: what to resubmit

Don't resubmit a whole training array: tasks that completed deleted their Optuna SQLite
(`cleanup_storage`), so rerunning them retunes from scratch instead of resuming. Instead run

```bash
bash slurm/missing_runs.sh            # all datasets; or: bash slurm/missing_runs.sh AMLSim
```

from the repo root (on the cluster — it audits `results/` on disk, not Slurm exit codes).
For every model it lists, per dataset, the array indices that are **missing** (task crashed
or never ran — no `Model:` block in any `results/experiments/` dump) or need a **top-up**
(metrics written, but the study's `results/tuning/*.db` survived, i.e. the walltime timeout
stopped it short of `n_trials`), and prints the exact `sbatch --array=... <script> <Dataset>
<model>` command for each gap. Resubmitted tasks resume their studies from the per-study
SQLite and rewrite the outputs. To find out *why* a task died, check its
`slurm/logs/*_<jobid>_<idx>.err` or `sacct -M wice -j <jobid> --format=JobID%20,State,ExitCode`.

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
- `results/experiments/{type}_CNN.txt|.png` — static CNN test metrics (timing-independent).
- `results/tuning/…_best_params.json` — best HPs (+ val AUC-PR where tuned) per combo / model.
- `results/models/{LSTM|Transformer}_{type}_{tag}_K{K}_{task}.pt`, `results/models/CNN_{type}.pt` —
  saved checkpoints (state_dict + best params). XGBoost/MLP/IsolationForest are not checkpointed.
- `results/experiments/summary.csv` — the one tidy table for the effect-of-timing analysis.
