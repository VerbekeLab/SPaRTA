#!/bin/bash -l
# CNN over the 3x3 pictures with Optuna HPO, for ONE dataset per submission:
#   sbatch slurm/train_cnn.slurm AMLWorld      # then AMLSim, Tide separately
# GPU job — experiment_CNN.py moves the model/data to cuda when available.

#SBATCH --job-name=sparta_cnn
#SBATCH --output=slurm/logs/train_cnn_%j.out
#SBATCH --error=slurm/logs/train_cnn_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1

#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
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
echo "[$(date)] CNN+Optuna for SPARTA_DATASET=$SPARTA_DATASET on $(nvidia-smi -L 2>/dev/null | head -1)"

python -u scripts/experiment_CNN.py

echo finish
