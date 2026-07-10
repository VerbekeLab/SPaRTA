#!/bin/bash -l
# Feed-forward MLP baseline (experiment_baseline.py) on the SAME K-snapshot windows as
# train_lstm.slurm / train_baseline.slurm — the timing sweep for ONE dataset per submission:
#   sbatch slurm/train_baseline_nn.slurm AMLWorld      # then AMLSim, Tide separately
# GPU job: the MLP's mini-batch training is far slower per Optuna trial than XGBoost's single
# tree fit, so it's split out of train_baseline.slurm (CPU: XGBoost + IsolationForest) onto a
# GPU, like train_lstm.slurm. Reads results/features/<run_tag>/; writes the
# {type}_{tag}_K{K}_{task}_baselines_NeuralNetwork stem.
#
# Array range = N_TIMING*N_K*N_TASK-1 (4*3*2 -> 0-23). Keep in sync with
# slurm/sweep_common.sh. PREREQUISITE: extract_features_dynamic.slurm has finished FOR THIS
# DATASET — submit with
#   sbatch --dependency=afterok:<extract_dyn_jobid> slurm/train_baseline_nn.slurm <Dataset>

#SBATCH --job-name=sparta_baseline_nn
#SBATCH --output=slurm/logs/train_baseline_nn_%A_%a.out
#SBATCH --error=slurm/logs/train_baseline_nn_%A_%a.err
#SBATCH --array=0-23
#SBATCH --time=24:00:00
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

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

source slurm/sweep_common.sh
resolve_dataset_arg "${1:-}"
decode_train "$SLURM_ARRAY_TASK_ID"

export SPARTA_BASELINE_MODELS="nn"
echo "[$(date)] baselines (nn): dataset=$SPARTA_DATASET tag=$SPARTA_RUN_TAG"\
     "K=$SPARTA_K task=$SPARTA_TASK on $(nvidia-smi -L 2>/dev/null | head -1)"

python -u scripts/experiment_baseline.py

echo finish
