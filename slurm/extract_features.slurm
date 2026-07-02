#!/bin/bash -l
# Build the STATIC 3x3 ego-network features for ONE dataset per submission:
#   sbatch slurm/extract_features.slurm AMLWorld      # then AMLSim, Tide as separate runs
# These feed the timing-INDEPENDENT models (experiment_features.py, experiment_CNN.py), so
# they're extracted once per dataset — SPARTA_TIME_DYNAMIC=false forces the static branch
# regardless of config/data/config.yaml. The DYNAMIC (snapshot) features for the timing
# sweep are built by extract_features_dynamic.slurm.
# CPU job — measure_calculation uses a multiprocessing Pool, not the GPU.

#SBATCH --job-name=sparta_extract
#SBATCH --output=slurm/logs/extract_%j.out
#SBATCH --error=slurm/logs/extract_%j.err
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

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
export SPARTA_TIME_DYNAMIC=false      # static branch -> results/features/{type}_static_*
echo "[$(date)] extracting STATIC features for SPARTA_DATASET=$SPARTA_DATASET"

python -u src/methods/measure_calculation.py

echo finish
