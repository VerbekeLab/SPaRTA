#!/bin/bash -l
# LSTM + Transformer over K-snapshot windows with Optuna HPO, for every dataset
# (one array task each). GPU job — experiment_LSTM.py moves the model + tensors
# to cuda when available (falls back to cpu); on VSC GPU nodes cuda is picked up
# automatically. Prerequisite: results/features/*_dynamic_*_features_echo.csv
# must exist (run extract_features.slurm with time_dynamic: True first).

#SBATCH --job-name=sparta_lstm
#SBATCH --output=slurm/logs/train_lstm_%A_%a.out
#SBATCH --error=slurm/logs/train_lstm_%A_%a.err
#SBATCH --array=0-2
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
mkdir -p slurm/logs results/tuning results/models results/experiments

cd $VSC_DATA/SPaRTA

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"

source activate SPaRTA_env

# Match thread pools to the allocation (avoid over-subscribing the node).
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

DATASETS=(AMLWorld AMLSim Tide)
export SPARTA_DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "[$(date)] LSTM+Transformer+Optuna for SPARTA_DATASET=$SPARTA_DATASET on $(nvidia-smi -L 2>/dev/null | head -1)"

python -u scripts/experiment_LSTM.py

echo finish
