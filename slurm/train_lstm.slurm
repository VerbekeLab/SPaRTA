#!/bin/bash -l
# LSTM + Transformer over K-snapshot windows with Optuna HPO — the timing sweep for ONE
# dataset per submission:
#   sbatch slurm/train_lstm.slurm AMLWorld      # then AMLSim, Tide separately
# One array task per (timing-combo x K x task). Each task reads results/features/<run_tag>/
# (produced by extract_features_dynamic.slurm for the SAME dataset) and writes per-combo
# outputs keyed by the {type}_{tag}_K{K}_{task} stem, so combos don't overwrite. GPU job —
# experiment_LSTM.py moves model + tensors to cuda when available.
#
# Both architectures use the SAME GPU resources, so by default one task tunes LSTM then
# Transformer sequentially, summing their Optuna time budgets. Pass the architecture as a 2nd
# arg, or export SPARTA_SEQ_MODELS (lstm|transformer|all), to fan them into separate tasks
# that run CONCURRENTLY instead (wall-clock ~= the slower architecture):
#   sbatch slurm/train_lstm.slurm AMLWorld lstm   # then transformer
# Per-arch tasks suffix their outputs (..._timeseries_LSTM.txt) so they never overwrite each
# other; collect_results.py reads both the combined and per-arch dumps.
#
# Array range = N_TIMING*N_K*N_TASK-1 (4*3*2 -> 0-23). Keep in sync with
# slurm/sweep_common.sh (run `bash slurm/sweep_common.sh` to print the range).
# PREREQUISITE: extract_features_dynamic.slurm has finished FOR THIS DATASET — submit with
#   sbatch --dependency=afterok:<extract_dyn_jobid> slurm/train_lstm.slurm <Dataset> [arch]
# The per-dataset grids in sweep_common.sh keep every (K, task) cell feasible (the largest
# K still leaves 18+ train anchors on every dataset); if a future grid edit breaks that,
# the temporal_split guard fails only its own array task — collect_results.py reports the gap.

#SBATCH --job-name=sparta_lstm
#SBATCH --output=slurm/logs/train_lstm_%A_%a.out
#SBATCH --error=slurm/logs/train_lstm_%A_%a.err
# Append %8 (e.g. 0-23%8) to cap concurrent GPU tasks if your allocation is limited.
#SBATCH --array=0-23
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# Sized from the COMPACT window form the experiments now use (build_sequence_windows +
# per-batch gather): the dense [N, K, 90] tensor that used to drive this request is never
# materialised. AMLWorld's worst cell is forecast K=17 —
#   values 8.6M rows x 90 x 4B = 3.1 GB, its scaled copy another 3.1 GB, the train-cell
#   gather in fit_scaler_compact ~2.5 GB, idx 4.9M x 17 x 4B = 0.3 GB -> under 10 GB, vs the
#   ~74 GB (2 x 37) this used to peak at. 32G leaves generous headroom for pandas/torch
#   overhead. GPU-side residency drops the same way: one shared values_t (~3 GB) instead of
#   the ~26 GB of dense per-band window tensors. Check the first cell with
#   `slurm_jobinfo <jobid>` and tighten further if you want.
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1

#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --account=lp_verbekelab

#SBATCH --mail-type=ALL
#SBATCH --mail-user bruno.deprez@kuleuven.be

set -euo pipefail
mkdir -p slurm/logs

cd $VSC_DATA/SPaRTA

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"

source activate SPaRTA_env

# Match thread pools to the allocation (avoid over-subscribing the node).
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Optuna stops starting trials this long before the wall-time kill so each study ends with its
# refit/metrics written. 30 min covers one in-flight sequence trial (longer than an MLP's) plus
# the final refit. NOTE this job tunes BOTH architectures in 6h by default — if that is tight,
# fan them out into separate tasks (`sbatch slurm/train_lstm.slurm <Dataset> lstm`, then
# `transformer`) rather than raising the wall. Studies resume from SQLite on resubmit.
export SPARTA_WALL_MARGIN_S=1800

# Dataset from the submission argument; (timing x K x task) from the array index.
source slurm/sweep_common.sh
resolve_dataset_arg "${1:-}"
decode_train "$SLURM_ARRAY_TASK_ID"

# Which architectures this task runs: 2nd positional arg, else a pre-set SPARTA_SEQ_MODELS,
# else both (the original single-job behaviour). See the header for the fan-out mode.
# This job holds a GPU, so it must NOT fall back to re-assembling windows from the feature
# files (~40 min of single-threaded pandas with the A100 idle). Require the Stage-1.5 store:
# a missing one fails the job in seconds instead. Build it with
#   sbatch slurm/build_windows.slurm <Dataset>
# and chain with --dependency=afterok. Unset this only to deliberately allow the slow path.
export SPARTA_REQUIRE_WINDOWS=1

export SPARTA_SEQ_MODELS="${2:-${SPARTA_SEQ_MODELS:-all}}"
echo "[$(date)] LSTM+Transformer+Optuna ($SPARTA_SEQ_MODELS): dataset=$SPARTA_DATASET tag=$SPARTA_RUN_TAG"\
     "K=$SPARTA_K task=$SPARTA_TASK on $(nvidia-smi -L 2>/dev/null | head -1)"

python -u scripts/experiment_LSTM.py

echo finish
