#!/bin/bash -l
# Build the 3x3 ego-network features ONE DATASET AT A TIME.
#
# Use this instead of extract_features.slurm when you cannot hold all three raw
# datasets on disk simultaneously. The array throttle `%1` forces Slurm to run a
# single task at a time, and the stage-in / stage-out hooks copy in one dataset's
# raw files, extract its features, then delete the raw files before the next task
# starts. Peak disk usage is therefore ONE raw dataset (the small per-dataset
# feature CSVs in results/features/ accumulate harmlessly).
#
# CPU job — measure_calculation uses a multiprocessing Pool, not the GPU.

#SBATCH --job-name=sparta_extract_seq
#SBATCH --output=slurm/logs/extract_seq_%A_%a.out
#SBATCH --error=slurm/logs/extract_seq_%A_%a.err
# `%1` = at most one array task running at once (sequential). Keep 0-2 in sync
# with the DATASETS length (3 datasets -> 0-2).
#SBATCH --array=0-2%1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# <<< EDIT >>> cluster/partition/account for your VSC allocation:
#SBATCH --clusters=CHANGE_ME
#SBATCH --partition=CHANGE_ME
#SBATCH --account=CHANGE_ME

set -euo pipefail
mkdir -p slurm/logs results/features

# <<< EDIT >>> environment setup for your cluster:
# module load cluster/wice/...      # e.g. a recent Python/CUDA stack if needed
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"   # <<< EDIT >>> path to conda.sh
ENV_NAME="SPaRTA_env"                                 # <<< EDIT >>> conda env name
source "$CONDA_SH"
conda activate "$ENV_NAME"

cd "$SLURM_SUBMIT_DIR"

# Sweep list — keep --array range in sync with its length (3 -> 0-2).
DATASETS=(AMLWorld AMLSim Tide)
export SPARTA_DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

# Per-dataset on-disk subdirectory under data/ (matches the loader contract).
declare -A DATA_SUBDIR=( [AMLWorld]=AMLWorld [AMLSim]=amlsim [Tide]=Tide )
SUBDIR="${DATA_SUBDIR[$SPARTA_DATASET]}"

# <<< EDIT >>> Where your raw datasets live when NOT staged (archive / $VSC_DATA
# / scratch). Leave STAGE_ROOT empty to disable staging entirely (i.e. you have
# already placed the data under data/<subdir>/ yourself).
STAGE_ROOT=""                                         # e.g. "$VSC_DATA/sparta_raw"

echo "[$(date)] extracting features for SPARTA_DATASET=$SPARTA_DATASET (data/$SUBDIR)"

# --- stage in -------------------------------------------------------------
if [[ -n "$STAGE_ROOT" ]]; then
    echo "[$(date)] staging in $STAGE_ROOT/$SUBDIR -> data/$SUBDIR"
    mkdir -p "data/$SUBDIR"
    cp -r "$STAGE_ROOT/$SUBDIR/." "data/$SUBDIR/"
fi

# --- extract --------------------------------------------------------------
# Run from the repo root (the scripts assume os.chdir('./')).
python src/methods/measure_calculation.py

# --- stage out ------------------------------------------------------------
# Only remove what we staged in; never touch data/ if staging is disabled.
if [[ -n "$STAGE_ROOT" ]]; then
    echo "[$(date)] removing staged raw data data/$SUBDIR to free disk for the next dataset"
    rm -rf "data/$SUBDIR"
fi

echo "[$(date)] done $SPARTA_DATASET; features in results/features/"
