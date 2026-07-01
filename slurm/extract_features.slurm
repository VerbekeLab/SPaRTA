#!/bin/bash -l
# Build the STATIC 3x3 ego-network features for every dataset (one array task each).
# These feed the timing-INDEPENDENT models (experiment_features.py, experiment_CNN.py), so
# they're extracted once — SPARTA_TIME_DYNAMIC=false forces the static branch regardless of
# config/data/config.yaml. The DYNAMIC (snapshot) features for the timing sweep are built by
# extract_features_dynamic.slurm.
# CPU job — measure_calculation uses a multiprocessing Pool, not the GPU.

#SBATCH --job-name=sparta_extract
#SBATCH --output=slurm/logs/extract_%A_%a.out
#SBATCH --error=slurm/logs/extract_%A_%a.err
#SBATCH --array=0-2
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
export SPARTA_TIME_DYNAMIC=false      # static branch -> results/features/{type}_static_*
echo "[$(date)] extracting STATIC features for SPARTA_DATASET=$SPARTA_DATASET"

python -u src/methods/measure_calculation.py

echo finish
