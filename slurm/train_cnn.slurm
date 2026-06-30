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
#SBATCH --gpus-per-node=1

#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --account=lp_verbekelab

#SBATCH --mail-type=ALL
#SBATCH --mail-user bruno.deprez@kuleuven.be

set -euo pipefail
mkdir -p slurm/logs results/tuning results/models

cd $VSC_DATA/SPaRTA

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"

source activate SPaRTA_env

DATASETS=(AMLWorld AMLSim Tide)
export SPARTA_DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "[$(date)] CNN+Optuna for SPARTA_DATASET=$SPARTA_DATASET on $(nvidia-smi -L 2>/dev/null | head -1)"

python -u scripts/experiment_CNN.py

echo finish
