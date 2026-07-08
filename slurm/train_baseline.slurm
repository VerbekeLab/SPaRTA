#!/bin/bash -l
# Non-sequential baselines (XGBoost + MLP + IsolationForest) on the SAME K-snapshot windows
# the sequence script uses — the timing sweep for ONE dataset per submission:
#   sbatch slurm/train_baseline.slurm AMLWorld      # then AMLSim, Tide separately
# One array task per (timing-combo x K x task), matching train_lstm.slurm so each cell's
# baselines and sequence models read the SAME windows (the per-combo comparison). Reads
# results/features/<run_tag>/; writes the {type}_{tag}_K{K}_{task}_baselines stem.
# CPU job (XGBoost/sklearn use OMP threads).
#
# --- WHICH MODELS RUN --------------------------------------------------------------------
# One task runs XGBoost + MLP + IsolationForest sequentially by default, so its --time must
# cover BOTH Optuna searches (XGBoost's + the MLP's) plus IsolationForest. On the heavy cells
# (forecast, large K -> up to K*90 features) that sum is what previously hit the 2 h cap, so
# --time is now 12 h. Two ways to run:
#
#   (A) All three in one task (default):
#         bash slurm/submit.sh slurm/train_baseline.slurm AMLWorld
#
#   (B) One model per task, so the two Optuna searches run CONCURRENTLY instead of summed
#       (wall-clock ~= slowest single model). Pass the model as a 2nd arg, or export
#       SPARTA_BASELINE_MODELS (xgboost|nn|iforest|all; comma-separated subsets allowed):
#         sbatch slurm/train_baseline.slurm AMLWorld xgboost   # then nn, then iforest
#         # (nn is the slow one; iforest is untuned and finishes in seconds)
#       Per-model tasks suffix their outputs (..._baselines_XGBoost.txt) so they never
#       overwrite each other; collect_results.py reads both the combined and per-model dumps.
#       Warm the shared window cache first (run xgboost, or the LSTM sweep, before the rest)
#       so the later per-model tasks just load results/timeseries/<tag>/ instead of rebuilding.
#       Per-model tasks finish well under 12 h — request less for faster scheduling, e.g.
#         sbatch --time=03:00:00 slurm/train_baseline.slurm AMLWorld nn
#
# Array range = N_TIMING*N_K*N_TASK-1 (4*3*2 -> 0-23). Keep in sync with
# slurm/sweep_common.sh. PREREQUISITE: extract_features_dynamic.slurm has finished FOR THIS
# DATASET — submit with
#   sbatch --dependency=afterok:<extract_dyn_jobid> slurm/train_baseline.slurm <Dataset> [model]

#SBATCH --job-name=sparta_baseline
#SBATCH --output=slurm/logs/train_baseline_%A_%a.out
#SBATCH --error=slurm/logs/train_baseline_%A_%a.err
#SBATCH --array=0-23
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --account=lp_verbekelab

#SBATCH --mail-type=ALL
#SBATCH --mail-user bruno.deprez@kuleuven.be

set -euo pipefail
mkdir -p slurm/logs

cd $VSC_DATA/SPaRTA

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"

source activate SPaRTA_env

# Match thread pools to the allocation so XGBoost / sklearn use the whole node
# without over-subscribing it.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Dataset from the submission argument; (timing x K x task) from the array index.
source slurm/sweep_common.sh
resolve_dataset_arg "${1:-}"
decode_train "$SLURM_ARRAY_TASK_ID"

# Which baselines this task runs: 2nd positional arg, else a pre-set SPARTA_BASELINE_MODELS
# (e.g. sbatch --export=ALL,SPARTA_BASELINE_MODELS=xgboost ...), else all three. See the
# header for the two run modes.
export SPARTA_BASELINE_MODELS="${2:-${SPARTA_BASELINE_MODELS:-all}}"
echo "[$(date)] baselines ($SPARTA_BASELINE_MODELS): dataset=$SPARTA_DATASET"\
     "tag=$SPARTA_RUN_TAG K=$SPARTA_K task=$SPARTA_TASK"

python -u scripts/experiment_baseline.py

echo finish
