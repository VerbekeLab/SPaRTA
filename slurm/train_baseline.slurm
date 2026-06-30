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
# <<< EDIT >>> cluster/partition/account for your VSC CPU allocation:
#SBATCH --clusters=CHANGE_ME
#SBATCH --partition=CHANGE_ME
#SBATCH --account=CHANGE_ME

set -euo pipefail
mkdir -p slurm/logs results/tuning results/experiments

# <<< EDIT >>> environment setup for your cluster:
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"   # <<< EDIT >>> path to conda.sh
ENV_NAME="SPaRTA_env"                                 # <<< EDIT >>> conda env name
source "$CONDA_SH"
conda activate "$ENV_NAME"

# Match thread pools to the allocation so XGBoost / sklearn use the whole node
# without over-subscribing it.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

DATASETS=(AMLWorld AMLSim Tide)
export SPARTA_DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "[$(date)] baselines (XGBoost + IsolationForest) for SPARTA_DATASET=$SPARTA_DATASET"

cd "$SLURM_SUBMIT_DIR"
python scripts/experiment_baseline.py
