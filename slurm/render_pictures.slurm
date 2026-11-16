#!/bin/bash -l
# OPTIONAL VGG16 track (1/2): render the 224x224 RGB network "pictures" for the VGG16 model
# into results/pickle/ (batched nodes_<i>.pkl + images_tensor_<i>.h5). CPU job — the renderer
# uses a multiprocessing Pool + igraph/cairo, not the GPU.
#
# CAVEATS (this track is intentionally NOT part of the timing sweep):
#  - src/methods/visualising_network.py hardcodes construct_network(dataset='AMLWorld') and a
#    STATIC graph -> AMLWorld HI-Small only, timing-independent. Other datasets / dynamic
#    snapshots are NOT supported without code changes (out of scope for the minimal sweep).
#  - Heavy: it renders every node of the (degree-pruned) graph. Give it a long wall time.
# Prerequisite: AMLWorld raw data present under data/amlworld/.

#SBATCH --job-name=sparta_render
#SBATCH --output=slurm/logs/render_%j.out
#SBATCH --error=slurm/logs/render_%j.err
# WALL TIME capped at 6h. NOTE: visualising_network.py has NO skip-existing logic, so a task
# killed at the cap restarts from the first batch on resubmit rather than continuing. Render
# fewer nodes per submission if 6h is not enough — or give it the same skip-existing check the
# dynamic extraction uses (see the `pending` list in measure_calculation.py).
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

echo "[$(date)] rendering 224x224 network pictures (AMLWorld HI-Small) -> results/pickle/"

python -u src/methods/visualising_network.py

echo finish
