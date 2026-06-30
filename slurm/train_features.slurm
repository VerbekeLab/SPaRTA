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
# <<< EDIT >>> cluster/partition/account for your VSC allocation:
#SBATCH --clusters=CHANGE_ME
#SBATCH --partition=CHANGE_ME
#SBATCH --account=CHANGE_ME

set -euo pipefail
mkdir -p slurm/logs results/experiments

# <<< EDIT >>> environment setup for your cluster:
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"   # <<< EDIT >>> path to conda.sh
ENV_NAME="SPaRTA_env"                                 # <<< EDIT >>> conda env name
source "$CONDA_SH"
conda activate "$ENV_NAME"

DATASETS=(AMLWorld AMLSim Tide)
export SPARTA_DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "[$(date)] training tabular models for SPARTA_DATASET=$SPARTA_DATASET"

cd "$SLURM_SUBMIT_DIR"
python scripts/experiment_features.py
