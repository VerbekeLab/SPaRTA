#!/bin/bash -l
# OPTIONAL VGG16 track (2/2): train CNN_visual_VGG16 on the rendered 224x224 pictures. GPU
# job — experiment_pictures.py moves model + data to cuda when available.
#
# CAVEATS:
#  - TRAINED, NOT TUNED: experiment_pictures.py has no Optuna/HPO — fixed 10 epochs, lr=1e-3,
#    pos_weight*50. So it produces a model but no tuned hyperparameters (unlike CNN/LSTM).
#    Wiring Optuna here is a separate effort, not part of the minimal timing-sweep change.
#  - AMLWorld HI-Small only: NetworkImageDataset defaults to dataset_type='HI-Small' and a
#    hardcoded num_files=422 (must match the batch count render_pictures produced).
#  - It writes a model only (results/models/cnn_visual_vgg16_experiment_pictures.pth), no
#    metrics file, so it does NOT appear in summary.csv.
# Prerequisites: render_pictures.slurm finished (results/pickle/) AND the static extraction
# (results/features/HI-Small_static_labels.csv — NetworkImageDataset reads the labels).
#   sbatch --dependency=afterok:<render_jobid> slurm/train_vgg16.slurm

#SBATCH --job-name=sparta_vgg16
#SBATCH --output=slurm/logs/train_vgg16_%j.out
#SBATCH --error=slurm/logs/train_vgg16_%j.err
# WALL TIME capped at 6h. experiment_pictures.py is a fixed 10-epoch loop with no Optuna and
# no checkpointing, so a kill at the cap loses the run entirely — lower `num_epochs` there if
# 6h is not enough.
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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

echo "[$(date)] training VGG16 on rendered pictures (AMLWorld HI-Small, untuned) on $(nvidia-smi -L 2>/dev/null | head -1)"

python -u scripts/experiment_pictures.py

echo finish
