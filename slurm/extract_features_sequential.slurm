#!/bin/bash -l
# Build the STATIC 3x3 ego-network features for ONE dataset, with disk staging:
#   sbatch slurm/extract_features_sequential.slurm AMLWorld   # then AMLSim, Tide separately
#
# Use this instead of extract_features.slurm when you cannot hold all three raw
# datasets on disk simultaneously. The stage-in hook SYMLINKS the dataset's raw
# files into data/<subdir> from an external archive, extracts its features, then
# removes the symlink at the end. Peak disk usage is therefore ONE raw dataset
# (the small per-dataset feature CSVs in results/features/ accumulate harmlessly).
# To keep runs strictly one-after-another, chain the submissions:
#   J1=$(sbatch --parsable slurm/extract_features_sequential.slurm AMLWorld)
#   J2=$(sbatch --parsable --dependency=afterany:$J1 slurm/extract_features_sequential.slurm AMLSim)
#   sbatch --dependency=afterany:$J2 slurm/extract_features_sequential.slurm Tide
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
#SBATCH --output=slurm/logs/extract_seq_%j.out
#SBATCH --error=slurm/logs/extract_seq_%j.err
# WALL TIME capped at 6h. NOTE: the static branch builds ONE network, so there is nothing to
# resume — a kill at the cap loses the run. (The dynamic extraction skips completed snapshots;
# the static one has no such granularity.) If it does not fit in 6h, reduce the work rather
# than resubmitting.
#SBATCH --time=06:00:00
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

# Dataset from the submission argument (or SPARTA_DATASET via --export).
source slurm/sweep_common.sh
resolve_dataset_arg "${1:-}"

# Per-dataset on-disk subdirectory under data/ (matches the loader contract).
declare -A DATA_SUBDIR=( [AMLWorld]=AMLWorld [AMLSim]=amlsim [Tide]=Tide )
SUBDIR="${DATA_SUBDIR[$SPARTA_DATASET]}"

# <<< EDIT >>> Archive that holds your raw datasets as data/<subdir>-shaped
# folders (e.g. $VSC_DATA/sparta_raw/{AMLWorld,amlsim,Tide}). Leave STAGE_ROOT
# empty to disable staging entirely -- then you are responsible for having the
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
