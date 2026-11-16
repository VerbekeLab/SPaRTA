#!/bin/bash
# Shared sweep grid + array-index decoders for the SPaRTA timing sweep. SOURCED (not
# executed) by the extraction and training Slurm scripts so the grid is defined ONCE — the
# timing tags the training jobs look for are exactly the ones the extraction jobs produce.
#
# The grid is PER DATASET: each dataset sweeps around its own defaults from
# config/data/config.yaml, because the spans and laundering-pattern durations differ
# (measured in notebooks/pattern_durations.ipynb):
#   AMLWorld : 10 days,  patterns ~3-5d   -> 6-hour grid, echo 1d, default K=13
#   AMLSim   : 200 days, patterns ~7-11d  -> daily grid,  echo 3d, default K=7
#   Tide     : 363 days, chains ~5-14d    -> 2-day grid,  echo 3d, default K=5
# Timing variants per dataset: [default echo, ~2x echo, plain width = step, plain width =
# default echo length]. K variants: [shorter, default, longer lookback]. The val/test
# bands (n_val/n_test_anchors) are NOT swept — they come from the YAML sequence blocks.
#
# The DATASET is NOT part of the array index: every script is submitted PER DATASET
# (`sbatch <script>.slurm <Dataset>`), so the three datasets run as separate jobs — one
# submission never sweeps all three at once. The arrays only cover the remaining axes.
# Every dataset MUST keep the same axis sizes (N_TIMING / N_K / N_TASK) so a single
# --array range fits all three.
#
# Edit the grid HERE and keep every script's --array range in sync with the counts:
#   extract_features_dynamic.slurm        : 0 .. N_TIMING-1              (= 3 for 4 combos)
#   train_lstm.slurm/train_baseline.slurm : 0 .. N_TIMING*N_K*N_TASK-1   (= 23 for 4*3*2)
# Run `bash slurm/sweep_common.sh` to print the ranges and the per-dataset grids.

# Datasets (the valid SPARTA_DATASET values; their type_dataset comes from
# config/data/config.yaml).
DATASETS=(AMLWorld AMLSim Tide)

# Resolve the dataset for a per-dataset submission: first script argument, else a
# pre-set SPARTA_DATASET (e.g. `sbatch --export=ALL,SPARTA_DATASET=Tide <script>`).
# Exits the job when neither is given or the name is unknown.
resolve_dataset_arg() {
    local name="${1:-${SPARTA_DATASET:-}}"
    if [[ -z "$name" ]]; then
        echo "ERROR: no dataset given. Submit per dataset: sbatch <script>.slurm <AMLWorld|AMLSim|Tide>" >&2
        exit 1
    fi
    local d
    for d in "${DATASETS[@]}"; do
        if [[ "$d" == "$name" ]]; then
            export SPARTA_DATASET="$name"
            return 0
        fi
    done
    echo "ERROR: unknown dataset '$name' (expected one of: ${DATASETS[*]})" >&2
    exit 1
}

# Per-dataset snapshot grid + expensive re-extraction axis (one Stage-1 feature build per
# combo). Index i -> a timing combo whose run_tag is {D}_TIMING_LABELS[i] BY CONSTRUCTION
# (must match src/utils/setup.py:run_tag):
#   echo on  : window = [end - DAYS_ECHO days, end], decayed; width ignored -> tag {g}_echo{DAYS}
#   echo off : window = [end - TIME_WIDTH,     end], plain                  -> tag {g}_w{WIDTH}
# where g = {time_type[0]}{time_step}. NOTE: days_echo is always in DAYS regardless of
# time_type; time_width is in units of time_type. Keep step <= window everywhere, or
# transactions between anchors are silently dropped.

AMLWorld_TIME_TYPE=hours
AMLWorld_TIME_STEP=6
AMLWorld_TIMING_LABELS=(h6_echo1 h6_echo2 h6_w6 h6_w24)
AMLWorld_TIMING_ECHO=(true true false false)
AMLWorld_TIMING_DAYS=(1 2 1 1)        # SPARTA_DAYS_ECHO (used when echo=true)
AMLWorld_TIMING_WIDTH=(6 6 6 24)      # SPARTA_TIME_WIDTH in hours (used when echo=false)
AMLWorld_KS=(7 13 17)                 # lookback (K-1)*6h + 1d echo = 2.5 / 4.0 / 5.0 days

AMLSim_TIME_TYPE=days
AMLSim_TIME_STEP=1
AMLSim_TIMING_LABELS=(d1_echo3 d1_echo7 d1_w1 d1_w3)
AMLSim_TIMING_ECHO=(true true false false)
AMLSim_TIMING_DAYS=(3 7 3 3)
AMLSim_TIMING_WIDTH=(1 1 1 3)
AMLSim_KS=(3 7 9)                     # lookback (K-1)*1d + 3d echo = 5 / 9 / 11 days

Tide_TIME_TYPE=days
Tide_TIME_STEP=2
Tide_TIMING_LABELS=(d2_echo3 d2_echo7 d2_w2 d2_w3)
Tide_TIMING_ECHO=(true true false false)
Tide_TIMING_DAYS=(3 7 3 3)
Tide_TIMING_WIDTH=(2 2 2 3)
Tide_KS=(3 5 7)                       # lookback (K-1)*2d + 3d echo = 7 / 11 / 15 days

# Free re-slice axis (NO re-extraction — build_sequence_dataset re-windows existing
# snapshots): task; K is per dataset above.
TASKS=(nowcast forecast)

N_TIMING=4        # identical for every dataset (uniform --array ranges)
N_K=3
N_TASK=${#TASKS[@]}

# Indirect lookup into the ACTIVE dataset's grid: `_grid TIMING_LABELS 2` expands
# ${<SPARTA_DATASET>_TIMING_LABELS[2]}. Plain bash-3.2 indirection (${!ref}), no bash-4
# namerefs/associative arrays, so the sanity check below also runs on macOS.
_grid() {
    local ref="${SPARTA_DATASET}_$1"
    [[ $# -gt 1 ]] && ref="${ref}[$2]"
    printf '%s' "${!ref}"
}

# Export the timing env for timing-combo index $1 of the ACTIVE dataset (used by both
# extraction and training). Always sets every SPARTA_TIME_* knob; resolve_timing/run_tag
# (src/utils/setup.py) pick the ones that matter for this combo's echo setting, so
# SPARTA_RUN_TAG == {D}_TIMING_LABELS[$1].
set_timing_env() {
    local t=$1
    if [[ -z "${SPARTA_DATASET:-}" ]]; then
        echo "ERROR: set_timing_env needs SPARTA_DATASET — call resolve_dataset_arg first." >&2
        exit 1
    fi
    export SPARTA_TIME_DYNAMIC=true
    export SPARTA_TIME_TYPE="$(_grid TIME_TYPE)"
    export SPARTA_TIME_STEP="$(_grid TIME_STEP)"
    export SPARTA_ECHO="$(_grid TIMING_ECHO "$t")"
    export SPARTA_DAYS_ECHO="$(_grid TIMING_DAYS "$t")"
    export SPARTA_TIME_WIDTH="$(_grid TIMING_WIDTH "$t")"
    export SPARTA_RUN_TAG="$(_grid TIMING_LABELS "$t")"   # informational: the dir this run uses
}

# Decode a 1D extraction index (timing combo) from $1; sets the timing env.
# The dataset comes separately from resolve_dataset_arg.
decode_extract() {
    set_timing_env "$1"
}

# Decode a 3D training index (timing × K × task) from $1; sets the timing env,
# SPARTA_K, SPARTA_TASK. Mixed-radix, task fastest then K then timing.
# The dataset comes separately from resolve_dataset_arg.
decode_train() {
    local idx=$1
    local task_i=$(( idx % N_TASK )); idx=$(( idx / N_TASK ))
    local k_i=$(( idx % N_K ));       idx=$(( idx / N_K ))
    set_timing_env "$idx"
    export SPARTA_K="$(_grid KS "$k_i")"
    export SPARTA_TASK="${TASKS[$task_i]}"
}

# When executed directly (not sourced), print the grids and array sizes as a sanity check.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "N_TIMING=$N_TIMING N_K=$N_K N_TASK=$N_TASK (datasets submitted separately: ${DATASETS[*]})"
    echo "extract array range (per dataset) : 0-$(( N_TIMING - 1 ))"
    echo "train   array range (per dataset) : 0-$(( N_TIMING * N_K * N_TASK - 1 ))"
    for SPARTA_DATASET in "${DATASETS[@]}"; do
        tags="${SPARTA_DATASET}_TIMING_LABELS[@]"
        ks="${SPARTA_DATASET}_KS[@]"
        echo "$SPARTA_DATASET: grid=$(_grid TIME_TYPE)/step$(_grid TIME_STEP) | tags: ${!tags} | K: ${!ks}"
    done
fi
