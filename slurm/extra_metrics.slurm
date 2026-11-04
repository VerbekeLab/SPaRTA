#!/bin/bash -l
# F1@0.5, best-F1 and precision@100 for every saved test-prediction table (all datasets, timing
# combos, K, task) -> results/experiments/extra_metrics.csv. Reads results/predictions/*.parquet
# (written by the training jobs, or backfilled with SPARTA_RESCORE=1 -- see slurm/README.md); no
# training. Run after the tables exist:
#   sbatch slurm/extra_metrics.slurm
# Extra flags go to the script, e.g. add P@500 and R-precision:
#   sbatch slurm/extra_metrics.slurm --ks 100,500 --r-precision
# CPU only; memory is set by the largest table (AMLWorld: a few million rows x a few models).

#SBATCH --job-name=sparta_extra_metrics
#SBATCH --output=slurm/logs/extra_metrics_%j.out
#SBATCH --error=slurm/logs/extra_metrics_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

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

echo "[$(date)] extra metrics -> results/experiments/extra_metrics.csv"

python -u scripts/extra_metrics.py "$@"

echo finish
