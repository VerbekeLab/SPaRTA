#!/bin/bash -l
# CNN over the 3x3 pictures with Optuna HPO, for every dataset (one array task
# each). GPU job — experiment_CNN.py moves the model/data to cuda when available.

#SBATCH --job-name=sparta_cnn
#SBATCH --output=slurm/logs/train_cnn_%A_%a.out
#SBATCH --error=slurm/logs/train_cnn_%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# <<< EDIT >>> GPU request — syntax varies per cluster (pick what yours expects):
#SBATCH --gpus-per-node=1
# #SBATCH --gres=gpu:1
# <<< EDIT >>> cluster/partition/account for your VSC GPU allocation:
#SBATCH --clusters=CHANGE_ME
#SBATCH --partition=CHANGE_ME
#SBATCH --account=CHANGE_ME

set -euo pipefail
mkdir -p slurm/logs results/tuning results/models

# <<< EDIT >>> environment setup for your cluster (load the CUDA-matching modules):
# module load cluster/wice/gpu
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"   # <<< EDIT >>> path to conda.sh
ENV_NAME="SPaRTA_env"                                 # <<< EDIT >>> conda env name
source "$CONDA_SH"
conda activate "$ENV_NAME"

DATASETS=(AMLWorld AMLSim Tide)
export SPARTA_DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "[$(date)] CNN+Optuna for SPARTA_DATASET=$SPARTA_DATASET on $(nvidia-smi -L 2>/dev/null | head -1)"

cd "$SLURM_SUBMIT_DIR"
python scripts/experiment_CNN.py
