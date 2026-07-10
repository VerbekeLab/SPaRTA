#!/bin/bash -l
# Non-sequential CPU baselines (XGBoost + IsolationForest) on the SAME K-snapshot windows
# the sequence script uses — the timing sweep for ONE dataset per submission:
#   sbatch slurm/train_baseline.slurm AMLWorld      # then AMLSim, Tide separately
# One array task per (timing-combo x K x task), matching train_lstm.slurm so each cell's
# baselines and sequence models read the SAME windows (the per-combo comparison). Reads
# results/features/<run_tag>/; writes the {type}_{tag}_K{K}_{task}_baselines stem.
# CPU job (XGBoost/sklearn use OMP threads).
#
# --- THE MLP HAS ITS OWN SCRIPT ----------------------------------------------------------
# The feed-forward MLP baseline is NOT run here. On the training-set sizes these windows
# produce (millions of rows), mini-batch SGD on CPU is orders of magnitude slower per trial
# than XGBoost's single vectorized tree fit — 150 XGBoost trials finish in ~30 min while the
# MLP couldn't clear 20 trials in 24 h. It now runs on GPU via slurm/train_baseline_nn.slurm
# (see that file's header). SPARTA_BASELINE_MODELS here defaults to "xgboost,iforest" so this
# CPU job never runs the MLP by accident.
#
# --- WHICH MODELS RUN --------------------------------------------------------------------
# XGBoost + IsolationForest run sequentially by default (IsolationForest is untuned and
# finishes in seconds, so this barely changes the job's wall time vs. XGBoost alone). Each
# Optuna study persists to its own SQLite file, so a job that still hits the time cap RESUMES
# on resubmit (delete results/tuning/baseline_*_<stem>.db to start a study over instead).
# Override the selection via a 2nd arg, or export SPARTA_BASELINE_MODELS (xgboost|nn|iforest|
# all; comma-separated subsets allowed) — e.g. to run only XGBoost:
#   sbatch slurm/train_baseline.slurm AMLWorld xgboost
# Per-model tasks suffix their outputs (..._baselines_XGBoost.txt) so they never overwrite
# each other; collect_results.py reads both the combined and per-model dumps.
#
# Array range = N_TIMING*N_K*N_TASK-1 (4*3*2 -> 0-23). Keep in sync with
# slurm/sweep_common.sh. PREREQUISITE: extract_features_dynamic.slurm has finished FOR THIS
# DATASET — submit with
#   sbatch --dependency=afterok:<extract_dyn_jobid> slurm/train_baseline.slurm <Dataset> [model]

#SBATCH --job-name=sparta_baseline
#SBATCH --output=slurm/logs/train_baseline_%A_%a.out
#SBATCH --error=slurm/logs/train_baseline_%A_%a.err
#SBATCH --array=0-23
#SBATCH --time=04:00:00
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
# (e.g. sbatch --export=ALL,SPARTA_BASELINE_MODELS=xgboost ...), else XGBoost + IsolationForest
# (the MLP lives in train_baseline_nn.slurm — see this file's header).
export SPARTA_BASELINE_MODELS="${2:-${SPARTA_BASELINE_MODELS:-xgboost,iforest}}"
echo "[$(date)] baselines ($SPARTA_BASELINE_MODELS): dataset=$SPARTA_DATASET"\
     "tag=$SPARTA_RUN_TAG K=$SPARTA_K task=$SPARTA_TASK"

python -u scripts/experiment_baseline.py

echo finish
