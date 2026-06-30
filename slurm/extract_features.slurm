#!/bin/bash -l
# Build the 3x3 ego-network features for every dataset (one array task each).
# CPU job — measure_calculation uses a multiprocessing Pool, not the GPU.

#SBATCH --job-name=sparta_extract
#SBATCH --output=slurm/logs/extract_%A_%a.out
#SBATCH --error=slurm/logs/extract_%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# <<< EDIT >>> cluster/partition/account for your VSC allocation:
#SBATCH --clusters=CHANGE_ME
#SBATCH --partition=CHANGE_ME
#SBATCH --account=CHANGE_ME

set -euo pipefail
mkdir -p slurm/logs results/features

# <<< EDIT >>> environment setup for your cluster:
# module load cluster/wice/...      # e.g. a recent Python/CUDA stack if needed
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"   # <<< EDIT >>> path to conda.sh
ENV_NAME="SPaRTA_env"                                 # <<< EDIT >>> conda env name
source "$CONDA_SH"
conda activate "$ENV_NAME"

# Sweep list — keep --array range in sync with its length (3 -> 0-2).
DATASETS=(AMLWorld AMLSim Tide)
export SPARTA_DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "[$(date)] extracting features for SPARTA_DATASET=$SPARTA_DATASET"

# Run from the repo root (the scripts assume os.chdir('./')).
cd "$SLURM_SUBMIT_DIR"
python src/methods/measure_calculation.py
