#!/bin/bash -l
# PACK several array cells of a GPU training script onto ONE GPU allocation. These jobs are
# single-process, launch-bound loops that keep an A100 at ~5-25% utilisation (VSC flagged job
# 62252588 at 5% / 6 GB VRAM), so one GPU per cell wastes it. Here one job holds one GPU and
# runs PACK cells of the sweep CONCURRENTLY as separate processes, so the sweep needs ~PACK
# times fewer GPU allocations and each GPU is kept busy.
#
#   sbatch --array=0-5 slurm/pack_gpu.slurm <inner-script> <Dataset> [inner args...]
#   sbatch --array=0-5 slurm/pack_gpu.slurm slurm/train_lstm.slurm     AMLWorld lstm
#   sbatch --array=0-5 slurm/pack_gpu.slurm slurm/train_cnn_seq.slurm  AMLWorld transformer
#
# The inner script is run UNCHANGED, once per cell, with SLURM_ARRAY_TASK_ID set to that
# cell's index in the (timing x K x task) grid, so every cell decodes and writes exactly what
# the unpacked array task would (same study names, same result files). Works for the arrayed
# scripts: train_lstm, train_cnn_seq, train_baseline_nn.
#
# Array range: pack job t takes cells [t*PACK, (t+1)*PACK) of the cell list, so
#   --array=0-$(( ceil(cells / PACK) - 1 ))      (24 cells, PACK=4 -> 0-5; the default)
# Environment knobs (pass with --export=ALL,NAME=value):
#   PACK=4            cells per job (default 4). Each cell is its own process.
#   PACK_CELLS=a,b,c  explicit cell indices instead of 0..23 — for top-ups / one task kind:
#                       PACK_CELLS=0,2,4,6,8,10,12,14,16,18,20,22   (EVEN = nowcast only)
#                       PACK_CELLS=1,3,5,7,9,11,13,15,17,19,21,23   (ODD  = forecast only)
#                     (the array range is then ceil(#cells / PACK) - 1)
#   PACK_STAGGER_S=20 seconds between process launches, so the processes do not all load the
#                     window store (a few GB each) at the same instant.
#
# Sizing (override on the sbatch line; --mem/--cpus-per-task are split across the PACK cells):
#  - train_lstm / train_cnn_seq: a cell peaks under ~10 GB host RAM and ~6 GB of VRAM, so the
#    defaults below (96G, 16 cores, PACK=4) leave headroom and stay inside one GPU's share of a
#    wice node (123G -> 18 cores on gpu_a100).
#  - train_baseline_nn: NOWCAST cells ~12 GB each -> PACK=4 with --mem=64G is fine. FORECAST
#    cells peak ~53 GB (a dense [rows, K*90] matrix) -> do NOT pack more than 2 of them
#    (--mem=120G), or use the unpacked script.
#  - Optuna's time budget is the JOB's SLURM_JOB_END_TIME, shared by all PACK processes (each
#    runs the whole wall). Sharing the GPU slows each cell, so each does fewer trials per job;
#    studies resume from their SQLite, so a short cell just gets topped up on resubmit
#    (`bash slurm/missing_runs.sh`). Raise --time and SPARTA_WALL_MARGIN_S if trials get long.
#
# At the end the job prints the GPU's MEAN/MAX utilisation and peak VRAM (sampled every 30 s
# into slurm/logs/gpu_<jobid>.csv) — check it on the first job before submitting the rest.
# Per-cell output goes to slurm/logs/pack_<jobid>_cell<idx>.log.

#SBATCH --job-name=sparta_pack
#SBATCH --output=slurm/logs/pack_%A_%a.out
#SBATCH --error=slurm/logs/pack_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gpus-per-node=1

#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --account=lp_verbekelab

#SBATCH --mail-type=ALL
#SBATCH --mail-user bruno.deprez@kuleuven.be

# No -e: one failing cell must not abort its siblings; failures are collected and reported.
set -uo pipefail
mkdir -p slurm/logs

cd $VSC_DATA/SPaRTA

if [[ $# -lt 2 ]]; then
    echo "Usage: sbatch --array=0-5 slurm/pack_gpu.slurm <inner-script> <Dataset> [inner args...]" >&2
    exit 1
fi
INNER=$1
DS=$2
shift 2
if [[ ! -f "$INNER" ]]; then
    echo "ERROR: inner script '$INNER' not found (run from the repo root)." >&2
    exit 1
fi

source slurm/sweep_common.sh
resolve_dataset_arg "$DS"
TOTAL=$(( N_TIMING * N_K * N_TASK ))

PACK=${PACK:-4}
if [[ -n "${PACK_CELLS:-}" ]]; then
    IFS=, read -r -a ALL <<< "$PACK_CELLS"
else
    ALL=($(seq 0 $(( TOTAL - 1 ))))
fi

start=$(( ${SLURM_ARRAY_TASK_ID:-0} * PACK ))
MINE=("${ALL[@]:$start:$PACK}")
if [[ ${#MINE[@]} -eq 0 ]]; then
    echo "pack task ${SLURM_ARRAY_TASK_ID:-0}: no cells (list has ${#ALL[@]}, PACK=$PACK) — nothing to do."
    exit 0
fi

# Split the cores across the cells; the inner scripts size their thread pools from
# SLURM_CPUS_PER_TASK, so each cell sees only its own share.
PER_CPUS=$(( ${SLURM_CPUS_PER_TASK:-4} / ${#MINE[@]} ))
(( PER_CPUS < 1 )) && PER_CPUS=1

JOBTAG=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${SLURM_ARRAY_TASK_ID:-0}
GPU_LOG=slurm/logs/gpu_${JOBTAG}.csv
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 30 \
    > "$GPU_LOG" 2>/dev/null &
MON=$!

echo "[$(date)] pack task ${SLURM_ARRAY_TASK_ID:-0}: $INNER $DS $* | cells: ${MINE[*]} |" \
     "$PER_CPUS cores/cell | $(nvidia-smi -L 2>/dev/null | head -1)"

pids=()
for idx in "${MINE[@]}"; do
    log=slurm/logs/pack_${JOBTAG}_cell${idx}.log
    echo "[$(date)] launching cell $idx -> $log"
    (
        export SLURM_ARRAY_TASK_ID=$idx SLURM_CPUS_PER_TASK=$PER_CPUS
        exec bash "$INNER" "$DS" "$@"
    ) > "$log" 2>&1 &
    pids+=($!)
    sleep "${PACK_STAGGER_S:-20}"
done

rc=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[$(date)] cell ${MINE[$i]} finished OK"
    else
        echo "[$(date)] cell ${MINE[$i]} FAILED (see slurm/logs/pack_${JOBTAG}_cell${MINE[$i]}.log)" >&2
        rc=1
    fi
done

kill "$MON" 2>/dev/null
wait "$MON" 2>/dev/null
if [[ -s "$GPU_LOG" ]]; then
    awk -F, '{u+=$1; if ($1>mu) mu=$1; if ($2>mm) mm=$2; n++}
             END {printf "GPU utilisation over %d samples: mean %.0f%%, max %d%%, peak VRAM %d MiB\n", n, u/n, mu, mm}' "$GPU_LOG"
fi

echo finish
exit $rc
