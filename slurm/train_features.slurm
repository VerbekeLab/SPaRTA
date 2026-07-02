#!/bin/bash -l
# Train the tabular models (LogReg + XGBoost grid search + shallow NN) on the
# 3x3 features for ONE dataset per submission:
#   sbatch slurm/train_features.slurm AMLWorld      # then AMLSim, Tide separately
# CPU job — these models are small; no GPU requested.
#
# NOTE: scripts/experiment_features.py runs all three models in one go; the
# SPARTA_EXPERIMENT var only selects which params block is read.

#SBATCH --job-name=sparta_feat
#SBATCH --output=slurm/logs/train_feat_%j.out
#SBATCH --error=slurm/logs/train_feat_%j.err
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

# Dataset from the submission argument (or SPARTA_DATASET via --export).
source slurm/sweep_common.sh
resolve_dataset_arg "${1:-}"
echo "[$(date)] training tabular models for SPARTA_DATASET=$SPARTA_DATASET"

python -u scripts/experiment_features.py

echo finish
