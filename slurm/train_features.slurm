#!/bin/bash -l
# Train the tabular models (LogReg + XGBoost grid search + shallow NN) on the
# 3x3 features for every dataset (one array task each). CPU job — these models
# are small; no GPU requested.
#
# NOTE: scripts/experiment_features.py runs all three models in one go; the
# SPARTA_EXPERIMENT var only selects which params block is read. The sweep
# dimension here is therefore the dataset.

#SBATCH --job-name=sparta_feat
#SBATCH --output=slurm/logs/train_feat_%A_%a.out
#SBATCH --error=slurm/logs/train_feat_%A_%a.err
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
mkdir -p slurm/logs

cd $VSC_DATA/SPaRTA

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"

source activate SPaRTA_env

DATASETS=(AMLWorld AMLSim Tide)
export SPARTA_DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "[$(date)] training tabular models for SPARTA_DATASET=$SPARTA_DATASET"

python -u scripts/experiment_features.py

echo finish
