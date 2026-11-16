#!/bin/bash -l
# Stage 1.5 — build the prebuilt per-TAG window store (src/data/window_store.py), the artefact
# every training job then derives its (K, task) windows from:
#   sbatch slurm/build_windows.slurm AMLWorld      # then AMLSim, Tide separately
#
# WHY THIS JOB EXISTS. The stacked snapshot block a window build needs depends only on the TAG,
# not on K or the task. Without this stage each of a tag's 6 (K x task) training cells
# re-read every feature file and re-assembled that block itself — ~40 min of single-threaded
# pandas, and TWO of those six run on GPU nodes (train_baseline_nn, train_lstm), i.e. ~40 min
# of an A100 sitting idle per cell. One CPU job per tag replaces all of it.
#
# ARRAY = one task per TIMING COMBO, so 0..N_TIMING-1 (= 0-3), NOT the 0-23 the training arrays
# use — K and task are free re-slices of the same store. Keep in sync with slurm/sweep_common.sh.
#
# PREREQUISITE: extract_features_dynamic.slurm has finished FOR THIS DATASET. Chain the whole
# thing together with dependencies:
#   ext=$(sbatch --parsable slurm/extract_features_dynamic.slurm AMLWorld)
#   win=$(sbatch --parsable --dependency=afterok:$ext slurm/build_windows.slurm AMLWorld)
#   sbatch --dependency=afterok:$win slurm/train_baseline.slurm    AMLWorld
#   sbatch --dependency=afterok:$win slurm/train_baseline_nn.slurm AMLWorld
#   sbatch --dependency=afterok:$win slurm/train_lstm.slurm        AMLWorld

#SBATCH --job-name=sparta_build_windows
#SBATCH --output=slurm/logs/build_windows_%A_%a.out
#SBATCH --error=slurm/logs/build_windows_%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# Peak is one snapshot DataFrame plus two copies of the stacked block (the per-snapshot list
# and its concatenation): ~2 x 2.6 GB at AMLWorld scale, plus the node-universe sort. Far below
# the training jobs' cap, because this stage never materialises the dense [N, K, 90] tensor.
#SBATCH --mem=48G

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

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Dataset from the submission argument; the timing combo from the array index.
source slurm/sweep_common.sh
resolve_dataset_arg "${1:-}"
decode_extract "$SLURM_ARRAY_TASK_ID"

echo "[$(date)] build window store: dataset=$SPARTA_DATASET tag=$SPARTA_RUN_TAG"

python -u scripts/build_window_store.py

echo finish
