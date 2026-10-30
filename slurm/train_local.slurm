#!/bin/bash -l
# LOCAL-view baselines (XGBoost + IsolationForest, CPU): experiment_baseline.py with
# SPARTA_BASELINE_VIEW=local — the SAME rows and temporal split as each window cell, but only
# the final window step's 90 features instead of all K. ONE dataset per submission:
#   sbatch slurm/train_local.slurm AMLWorld      # then AMLSim, Tide separately
# The MLP runs on GPU in slurm/train_local_nn.slurm, as for the window baselines.
#
# FORECAST ONLY: the nowcast window baseline already reads only the anchor snapshot, so it IS
# the local nowcast (the script refuses a local nowcast cell). decode_train puts task on the
# fastest axis, so forecast = the ODD indices -> --array=1-23:2 (12 cells). Writes
# {type}_{tag}_K{K}_forecast_local[_<models>].txt and local_xgb_<stem>.db studies (own studies:
# never resumes the window baseline's). Same Optuna budget/search space (`baselines` block).
#
# Everything else (model selection via a 2nd arg, resume-on-resubmit) is as in
# train_baseline.slurm — see its header.
#   sbatch slurm/train_local.slurm AMLWorld xgboost     # one model only
# PREREQUISITE: the window store for this dataset (build_windows.slurm).

#SBATCH --job-name=sparta_local
#SBATCH --output=slurm/logs/train_local_%A_%a.out
#SBATCH --error=slurm/logs/train_local_%A_%a.err
#SBATCH --array=1-23:2
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# One 90-column step per row, like a nowcast cell: AMLWorld's ~4.9M rows x 90 x 4B = 1.8 GB of
# band arrays, plus the window store's values block (~3 GB) and XGBoost's hist DMatrix. The
# K*90 forecast matrix that sized train_baseline.slurm's 64G is never built here.
#SBATCH --mem=20G

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

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Same margin as train_baseline.slurm (XGBoost trials are single fits).
export SPARTA_WALL_MARGIN_S=900

source slurm/sweep_common.sh
resolve_dataset_arg "${1:-}"
decode_train "$SLURM_ARRAY_TASK_ID"

export SPARTA_BASELINE_VIEW=local
export SPARTA_BASELINE_MODELS="${2:-${SPARTA_BASELINE_MODELS:-xgboost,iforest}}"
echo "[$(date)] local baselines ($SPARTA_BASELINE_MODELS): dataset=$SPARTA_DATASET"\
     "tag=$SPARTA_RUN_TAG K=$SPARTA_K task=$SPARTA_TASK"

python -u scripts/experiment_baseline.py

echo finish
