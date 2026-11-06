#!/bin/bash -l
# GraphSAGE GNN baseline (scripts/experiment_GNN.py): a 1- and a 2-layer GraphSAGE with
# log(1 + degree) as the only node feature, on the SAME rows and temporal split as the window
# baselines — the timing sweep for ONE dataset per submission:
#   sbatch slurm/train_gnn.slurm AMLWorld      # then AMLSim, Tide separately
# Both depths train in one task (fixed GARG-AML hyperparameters, no Optuna) and write
# {type}_{tag}_K{K}_{task}_gnn.txt with 'Model: GraphSAGE-1L' / 'Model: GraphSAGE-2L' blocks.
# All 24 cells (nowcast AND forecast): unlike the local MLP, the nowcast GNN is not a copy of
# an existing baseline. Packs onto shared GPUs like the other GPU scripts:
#   sbatch --array=0-5 slurm/pack_gpu.slurm slurm/train_gnn.slurm <Dataset>
# PREREQUISITES: the window store for this dataset (build_windows.slurm) and torch_geometric +
# pyg-lib in SPaRTA_env (see environment.txt).

#SBATCH --job-name=sparta_gnn
#SBATCH --output=slurm/logs/train_gnn_%A_%a.out
#SBATCH --error=slurm/logs/train_gnn_%A_%a.err
#SBATCH --array=0-23
# MEASURED on a laptop CPU (2026-11-06), one epoch per model on the two heaviest cells — the
# smallest K and the widest echo, i.e. the most training rows on the densest graphs:
#   AMLWorld h6_echo2 K=7 (array 6): 7.8M train rows, 17M edges; 1L 15 s, 2L 81 s per epoch
#   Tide     d2_echo7 K=3 (array 6): 4.5M train rows, 48M edges; 1L 28 s, 2L 350 s per epoch
# Neighbour sampling (CPU, also on the cluster) was only ~10% of the Tide 2L epoch; the rest is
# model maths the A100 takes over. Each model stops after 5 epochs without val improvement
# (max 50), so even 50 + 50 epochs should fit in 4 h. Confirm on the first job:
#   sbatch --array=6 slurm/train_gnn.slurm Tide   -> check its log and `seff <jobid>`
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# Peak host RAM in the same runs: 8.7 GB (AMLWorld), 8.3 GB (Tide). 20G leaves room for the CUDA
# context and forecast cells (a few % more nodes). PACK=4 in pack_gpu.slurm's 64G fits too.
#SBATCH --mem=20G
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

source slurm/sweep_common.sh
resolve_dataset_arg "${1:-}"
decode_train "$SLURM_ARRAY_TASK_ID"

# GPU job: fail fast on a missing window store instead of re-assembling with the A100 idle.
export SPARTA_REQUIRE_WINDOWS=1

echo "[$(date)] GraphSAGE: dataset=$SPARTA_DATASET tag=$SPARTA_RUN_TAG"\
     "K=$SPARTA_K task=$SPARTA_TASK on $(nvidia-smi -L 2>/dev/null | head -1)"

python -u scripts/experiment_GNN.py

echo finish
