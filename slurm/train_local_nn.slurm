#!/bin/bash -l
# LOCAL-view MLP baseline (GPU): experiment_baseline.py with SPARTA_BASELINE_VIEW=local and
# SPARTA_BASELINE_MODELS=nn — the SAME rows and temporal split as each window cell, but only
# the final window step's 90 features instead of all K. ONE dataset per submission:
#   sbatch slurm/train_local_nn.slurm AMLWorld      # then AMLSim, Tide separately
# XGBoost + IsolationForest run on CPU in slurm/train_local.slurm.
#
# FORECAST ONLY (see train_local.slurm): --array=1-23:2 (the ODD, forecast indices; 12 cells).
# Writes {type}_{tag}_K{K}_forecast_local_NeuralNetwork.txt and local_nn_<stem>.db studies.
# Packs onto shared GPUs like the other GPU scripts:
#   sbatch --array=0-2 --export=ALL,PACK_CELLS=1,3,5,7,9,11,13,15,17,19,21,23 \
#          slurm/pack_gpu.slurm slurm/train_local_nn.slurm <Dataset>
# PREREQUISITE: the window store for this dataset (build_windows.slurm).

#SBATCH --job-name=sparta_local_nn
#SBATCH --output=slurm/logs/train_local_nn_%A_%a.out
#SBATCH --error=slurm/logs/train_local_nn_%A_%a.err
#SBATCH --array=1-23:2
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# Sized like the NOWCAST baseline MLP (90 features per row): MEASURED 11.29 GiB peak and
# 02:20 walltime for job 62043355_0 (AMLWorld nowcast K=7, 6.19M train rows).
#SBATCH --mem=20G
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

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Same margin as train_baseline_nn.slurm.
export SPARTA_WALL_MARGIN_S=1200

source slurm/sweep_common.sh
resolve_dataset_arg "${1:-}"
decode_train "$SLURM_ARRAY_TASK_ID"

# GPU job: fail fast on a missing window store instead of re-assembling with the A100 idle.
export SPARTA_REQUIRE_WINDOWS=1

export SPARTA_BASELINE_VIEW=local
export SPARTA_BASELINE_MODELS="nn"
echo "[$(date)] local baselines (nn): dataset=$SPARTA_DATASET tag=$SPARTA_RUN_TAG"\
     "K=$SPARTA_K task=$SPARTA_TASK on $(nvidia-smi -L 2>/dev/null | head -1)"

python -u scripts/experiment_baseline.py

echo finish
