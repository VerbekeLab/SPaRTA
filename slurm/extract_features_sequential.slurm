#!/bin/bash -l
# Build the 3x3 ego-network features ONE DATASET AT A TIME.
#
# Use this instead of extract_features.slurm when you cannot hold all three raw
# datasets on disk simultaneously. The array throttle `%1` forces Slurm to run a
# single task at a time. For each task the stage-in hook SYMLINKS one dataset's
# raw files into data/<subdir> from an external archive, extracts its features,
# then removes the symlink before the next task starts. Peak disk usage is
# therefore ONE raw dataset (the small per-dataset feature CSVs in
# results/features/ accumulate harmlessly).
#
# Why a symlink and not a copy: CLAUDE.md marks data/ read-only ("never write,
# rename, or regenerate inside data/"). A symlink keeps the actual raw files in
# your external archive (STAGE_ROOT) and leaves data/ holding only a transient
# link that this script also removes -- no vendor files or generated artefacts
# are ever written into data/. If data/<subdir> already exists as a real
# directory, staging is skipped and the rule is trivially respected.
#
# CPU job — measure_calculation uses a multiprocessing Pool, not the GPU.

#SBATCH --job-name=sparta_extract_seq
#SBATCH --output=slurm/logs/extract_seq_%A_%a.out
#SBATCH --error=slurm/logs/extract_seq_%A_%a.err
# `%1` = at most one array task running at once (sequential). Keep 0-2 in sync
# with the DATASETS length (3 datasets -> 0-2).
#SBATCH --array=0-2%1
#SBATCH --time=16:00:00
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
mkdir -p results/features

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"

source activate SPaRTA_env

# Sweep list — keep --array range in sync with its length (3 -> 0-2).
DATASETS=(AMLWorld AMLSim Tide)
export SPARTA_DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

# Per-dataset on-disk subdirectory under data/ (matches the loader contract).
declare -A DATA_SUBDIR=( [AMLWorld]=AMLWorld [AMLSim]=amlsim [Tide]=Tide )
SUBDIR="${DATA_SUBDIR[$SPARTA_DATASET]}"

# <<< EDIT >>> Archive that holds your raw datasets as data/<subdir>-shaped
# folders (e.g. $VSC_DATA/sparta_raw/{AMLWorld,amlsim,Tide}). Leave STAGE_ROOT
# empty to disable staging entirely -- then you are responsible for having one
# dataset present under data/<subdir>/ before submitting.
STAGE_ROOT=""                                         # e.g. "$VSC_DATA/sparta_raw"

echo "[$(date)] extracting features for SPARTA_DATASET=$SPARTA_DATASET (data/$SUBDIR)"

# --- stage in (symlink only; never copies into the read-only data/) -------
STAGED_LINK=""
if [[ -n "$STAGE_ROOT" ]]; then
    if [[ -e "data/$SUBDIR" && ! -L "data/$SUBDIR" ]]; then
        echo "[$(date)] data/$SUBDIR already present as a real dir; skipping stage-in"
    else
        echo "[$(date)] linking $STAGE_ROOT/$SUBDIR -> data/$SUBDIR"
        ln -sfn "$STAGE_ROOT/$SUBDIR" "data/$SUBDIR"
        STAGED_LINK="data/$SUBDIR"
    fi
fi

# --- extract --------------------------------------------------------------
# Run from the repo root (the scripts assume os.chdir('./')).
python -u src/methods/measure_calculation.py

# --- stage out (remove only the symlink we created; vendor files untouched)
if [[ -n "$STAGED_LINK" ]]; then
    echo "[$(date)] removing transient symlink $STAGED_LINK"
    rm -f "$STAGED_LINK"
fi

echo "[$(date)] done $SPARTA_DATASET; features in results/features/"
