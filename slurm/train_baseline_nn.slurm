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
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# Sized from the COMPACT window form the experiments now use (build_sequence_windows +
# per-batch gather): the dense [N, K, 90] tensor that used to drive this request is never
# materialised. AMLWorld's worst cell is forecast K=17 —
#   the three 2D band arrays total 3.8M/0.87M/0.20M rows x 1530 x 4B = 29.7 GB (XGBoost and
#   an MLP both need a REAL dense 2D matrix, so for forecast that block must exist once), and
#   StandardScaler.transform copies the band it is scaling -> +23.2 GB transient on the train
#   band. ~53 GB peak. The nowcast cells now gather ONE column instead of building all K and
#   discarding 16/17 of it, so they need a small fraction of this. Check the first cell with
#   `slurm_jobinfo <jobid>` and tighten further if you want.
#SBATCH --mem=72G
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

# Optuna stops starting trials this long before the wall-time kill so the study ends with its
# refit/metrics written. walltime_budget's 1h default would be a sixth of this 6h job; an MLP
# trial is now ~minutes (permutation batching + the prebuilt window store), so 20 min covers
# the in-flight trial plus the final refit and metric writes. The study resumes from its
# SQLite on resubmit either way — `bash slurm/missing_runs.sh` lists the cells needing a top-up.
export SPARTA_WALL_MARGIN_S=1200

source slurm/sweep_common.sh
resolve_dataset_arg "${1:-}"
decode_train "$SLURM_ARRAY_TASK_ID"

# This job holds a GPU, so it must NOT fall back to re-assembling windows from the feature
# files (~40 min of single-threaded pandas with the A100 idle). Require the Stage-1.5 store:
# a missing one fails the job in seconds instead. Build it with
#   sbatch slurm/build_windows.slurm <Dataset>
# and chain with --dependency=afterok. Unset this only to deliberately allow the slow path.
export SPARTA_REQUIRE_WINDOWS=1

export SPARTA_BASELINE_MODELS="nn"
echo "[$(date)] baselines (nn): dataset=$SPARTA_DATASET tag=$SPARTA_RUN_TAG"\
     "K=$SPARTA_K task=$SPARTA_TASK on $(nvidia-smi -L 2>/dev/null | head -1)"

python -u scripts/experiment_baseline.py

echo finish
