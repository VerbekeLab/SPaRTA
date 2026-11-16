#!/bin/bash -l
# Aggregate every model's metrics (all datasets, timing combos, K, task) into one tidy
# table results/experiments/summary.csv. Run LAST, after the training arrays finish:
#   sbatch --dependency=afterany:<lstm_jobid>:<baseline_jobid>:<feat_jobid>:<cnn_jobid> slurm/collect.slurm
# afterany (not afterok) so infeasible per-combo cells that failed don't block the summary;
# collect_results.py just reports them as gaps. Tiny CPU job.

#SBATCH --job-name=sparta_collect
#SBATCH --output=slurm/logs/collect_%j.out
#SBATCH --error=slurm/logs/collect_%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

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

echo "[$(date)] aggregating results -> results/experiments/summary.csv"

python -u scripts/collect_results.py

echo finish
