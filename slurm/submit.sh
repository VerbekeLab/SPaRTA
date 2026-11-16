#!/bin/bash
# Submit a per-dataset SPaRTA job with the dataset name appended to the job name,
# so Slurm notification emails (whose subject quotes Name=<job-name>) say which
# dataset they are about. Equivalent to:
#   sbatch --job-name=<script-basename>_<Dataset> [extra sbatch args] <script> <Dataset>
#
# Usage: bash slurm/submit.sh <script.slurm> <Dataset> [extra sbatch args...]
#   bash slurm/submit.sh slurm/extract_features_dynamic.slurm AMLWorld
#   bash slurm/submit.sh slurm/train_lstm.slurm Tide --parsable --dependency=afterok:12345
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: bash slurm/submit.sh <script.slurm> <Dataset> [extra sbatch args...]" >&2
    exit 1
fi

script=$1
dataset=$2
shift 2

base=$(basename "$script" .slurm)
exec sbatch --job-name="${base}_${dataset}" "$@" "$script" "$dataset"
