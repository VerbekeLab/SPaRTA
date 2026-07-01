#!/bin/bash -l
# LSTM + Transformer over K-snapshot windows with Optuna HPO — the timing sweep: one array
# task per (dataset x timing-combo x K x task). Each task reads results/features/<run_tag>/
# (produced by extract_features_dynamic.slurm) and writes per-combo outputs keyed by the
# same {type}_{tag}_K{K}_{task} stem, so combos don't overwrite. GPU job — experiment_LSTM.py
# moves model + tensors to cuda when available.
#
# Array range = N_DATA*N_TIMING*N_K*N_TASK-1 (3*4*3*2 -> 0-71). Keep in sync with
# slurm/sweep_common.sh (run `bash slurm/sweep_common.sh` to print the range).
# PREREQUISITE: extract_features_dynamic.slurm has finished — submit with e.g.
#   sbatch --dependency=afterok:<extract_dyn_jobid> slurm/train_lstm.slurm
# Some large-K cells on short datasets (AMLWorld T~9) raise in temporal_split; that fails
# only its own array task — collect_results.py reports the gap.

#SBATCH --job-name=sparta_lstm
#SBATCH --output=slurm/logs/train_lstm_%A_%a.out
#SBATCH --error=slurm/logs/train_lstm_%A_%a.err
# Append %8 (e.g. 0-71%8) to cap concurrent GPU tasks if your allocation is limited.
#SBATCH --array=0-71
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1

#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --account=lp_verbekelab

#SBATCH --mail-type=ALL
#SBATCH --mail-user bruno.deprez@kuleuven.be

set -euo pipefail
mkdir -p slurm/logs

cd $VSC_DATA/SPaRTA
mkdir -p results/tuning results/models results/experiments

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"

source activate SPaRTA_env

# Match thread pools to the allocation (avoid over-subscribing the node).
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Decode (dataset x timing x K x task) -> SPARTA_DATASET + SPARTA_TIME_* + SPARTA_K + SPARTA_TASK.
source slurm/sweep_common.sh
decode_train "$SLURM_ARRAY_TASK_ID"
echo "[$(date)] LSTM+Transformer+Optuna: dataset=$SPARTA_DATASET tag=$SPARTA_RUN_TAG"\
     "K=$SPARTA_K task=$SPARTA_TASK on $(nvidia-smi -L 2>/dev/null | head -1)"

python -u scripts/experiment_LSTM.py

echo finish
