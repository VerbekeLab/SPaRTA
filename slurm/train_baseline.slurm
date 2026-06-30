#!/bin/bash -l
# Non-sequential baselines (XGBoost + MLP + IsolationForest) on the SAME K-snapshot windows
# the sequence script uses — the timing sweep: one array task per (dataset x timing-combo x
# K x task), matching train_lstm.slurm so each cell's baselines and sequence models read the
# SAME windows (the per-combo comparison). Reads results/features/<run_tag>/; writes the
# {type}_{tag}_K{K}_{task}_baselines stem. CPU job (XGBoost/sklearn use OMP threads).
#
# Array range = N_DATA*N_TIMING*N_K*N_TASK-1 (3*4*3*2 -> 0-71). Keep in sync with
# slurm/sweep_common.sh. PREREQUISITE: extract_features_dynamic.slurm has finished — submit
# with `sbatch --dependency=afterok:<extract_dyn_jobid> slurm/train_baseline.slurm`.

#SBATCH --job-name=sparta_baseline
#SBATCH --output=slurm/logs/train_baseline_%A_%a.out
#SBATCH --error=slurm/logs/train_baseline_%A_%a.err
#SBATCH --array=0-71
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
mkdir -p slurm/logs results/tuning results/experiments

cd $VSC_DATA/SPaRTA

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"

source activate SPaRTA_env

# Match thread pools to the allocation so XGBoost / sklearn use the whole node
# without over-subscribing it.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Decode (dataset x timing x K x task) -> SPARTA_DATASET + SPARTA_TIME_* + SPARTA_K + SPARTA_TASK.
source slurm/sweep_common.sh
decode_train "$SLURM_ARRAY_TASK_ID"
echo "[$(date)] baselines (XGBoost + MLP + IsolationForest): dataset=$SPARTA_DATASET"\
     "tag=$SPARTA_RUN_TAG K=$SPARTA_K task=$SPARTA_TASK"

python -u scripts/experiment_baseline.py

echo finish
