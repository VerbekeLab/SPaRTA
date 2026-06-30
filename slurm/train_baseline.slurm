#!/bin/bash -l
# Non-sequential baselines (XGBoost + IsolationForest) on the SAME K-snapshot
# windows the sequence script uses, for every dataset (one array task each).
# CPU job — neither model uses the GPU; XGBoost benefits from multiple CPU
# threads (OMP_NUM_THREADS matches the allocation). Prerequisite: same dynamic
# feature CSVs as train_lstm.slurm (run extract_features.slurm first).

#SBATCH --job-name=sparta_baseline
#SBATCH --output=slurm/logs/train_baseline_%A_%a.out
#SBATCH --error=slurm/logs/train_baseline_%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --account=lp_verbekelab

#SBATCH --mail-type=ALL
#SBATCH --mail-user bruno.deprez@kuleuven.be

set -euo pipefail
mkdir -p slurm/logs results/tuning results/experiments

cd $VSC_DATA/SPaRTA

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"

source activate SPaRTA_env

# Match thread pools to the allocation so XGBoost / sklearn use the whole node
# without over-subscribing it.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

DATASETS=(AMLWorld AMLSim Tide)
export SPARTA_DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "[$(date)] baselines (XGBoost + IsolationForest) for SPARTA_DATASET=$SPARTA_DATASET"

python -u scripts/experiment_baseline.py

echo finish
