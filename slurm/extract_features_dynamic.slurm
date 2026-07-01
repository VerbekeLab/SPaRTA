#!/bin/bash -l
# Build the DYNAMIC (snapshot) 3x3 ego-network features for the timing sweep: one array
# task per (dataset x timing-combo). Each task writes results/features/<run_tag>/ so the
# combos coexist (no overwrite). This is the EXPENSIVE Stage-1 axis — feature extraction
# per snapshot, ~199 snapshots for AMLSim and ~370 for Tide at daily step=1. The cheap
# K/task axis is NOT swept here (it re-windows these same files in the training jobs).
#
# Array range = N_DATA*N_TIMING-1 (3*4 -> 0-11). Keep in sync with slurm/sweep_common.sh
# (run `bash slurm/sweep_common.sh` to print the range). CPU job (multiprocessing Pool).

#SBATCH --job-name=sparta_extract_dyn
#SBATCH --output=slurm/logs/extract_dyn_%A_%a.out
#SBATCH --error=slurm/logs/extract_dyn_%A_%a.err
#SBATCH --array=0-11
#SBATCH --time=24:00:00
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

# Decode (dataset x timing) from the array index -> SPARTA_DATASET + SPARTA_TIME_* env.
source slurm/sweep_common.sh
decode_extract "$SLURM_ARRAY_TASK_ID"

echo "[$(date)] extract DYNAMIC: dataset=$SPARTA_DATASET tag=$SPARTA_RUN_TAG"\
     "(echo=$SPARTA_ECHO days_echo=$SPARTA_DAYS_ECHO width=$SPARTA_TIME_WIDTH)"

python -u src/methods/measure_calculation.py

echo finish
