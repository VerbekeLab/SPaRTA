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
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# Sized for AMLWorld's largest cells: the K=17 window tensor is ~37 GB float32 and peaks
# at ~2x during scaling/band-tensor construction. Without an explicit --mem the job gets
# only the small per-core default share and is OOM-killed before the first trial. AMLSim
# and Tide need far less — override at submit time (sbatch --mem=32G ...) if queueing matters.
#SBATCH --mem=128G
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

# Dataset from the submission argument; (timing x K x task) from the array index.
source slurm/sweep_common.sh
resolve_dataset_arg "${1:-}"
decode_train "$SLURM_ARRAY_TASK_ID"

# Which architectures this task runs: 2nd positional arg, else a pre-set SPARTA_SEQ_MODELS,
# else both (the original single-job behaviour). See the header for the fan-out mode.
export SPARTA_SEQ_MODELS="${2:-${SPARTA_SEQ_MODELS:-all}}"
echo "[$(date)] LSTM+Transformer+Optuna ($SPARTA_SEQ_MODELS): dataset=$SPARTA_DATASET tag=$SPARTA_RUN_TAG"\
     "K=$SPARTA_K task=$SPARTA_TASK on $(nvidia-smi -L 2>/dev/null | head -1)"

python -u scripts/experiment_LSTM.py

echo finish
