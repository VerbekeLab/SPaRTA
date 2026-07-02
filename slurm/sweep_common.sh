#!/bin/bash
# Shared sweep grid + array-index decoders for the SPaRTA timing sweep. SOURCED (not
# executed) by the extraction and training Slurm scripts so the grid is defined ONCE — the
# timing tags the training jobs look for are exactly the ones the extraction jobs produce.
#
# The DATASET is NOT part of the array index: every script is submitted PER DATASET
# (`sbatch <script>.slurm <Dataset>`), so the three datasets run as separate jobs — one
# submission never sweeps all three at once. The arrays only cover the remaining axes.
#
# Edit the grid HERE and keep every script's --array range in sync with the counts:
#   extract_features_dynamic.slurm        : 0 .. N_TIMING-1              (= 3 for 4 combos)
#   train_lstm.slurm/train_baseline.slurm : 0 .. N_TIMING*N_K*N_TASK-1   (= 23 for 4*3*2)
# Run `bash slurm/sweep_common.sh` to print the ranges.

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

# Expensive re-extraction axis (one Stage-1 feature build per combo). Daily step=1
# throughout, so the snapshot count T is constant per dataset and the K-windows are
# comparable across combos. Index i -> a timing combo whose run_tag is TIMING_LABELS[i]
# BY CONSTRUCTION (must match src/utils/setup.py:run_tag):
#   echo on  : window = [end-DAYS_ECHO, end], width ignored -> tag d1_echo{DAYS_ECHO}
#   echo off : window = [end-TIME_WIDTH, end] (plain)        -> tag d1_w{TIME_WIDTH}
TIMING_LABELS=(d1_echo3 d1_echo7 d1_w1 d1_w3)
TIMING_ECHO=(true     true     false  false)
TIMING_DAYS=(3        7        3      3)      # SPARTA_DAYS_ECHO (used when echo=true)
TIMING_WIDTH=(1       1        1      3)      # SPARTA_TIME_WIDTH (used when echo=false)

# Free re-slice axis (NO re-extraction — build_sequence_dataset re-windows existing
# snapshots): sequence window length K and task.
KS=(2 3 5)
TASKS=(nowcast forecast)

N_TIMING=${#TIMING_LABELS[@]}
N_K=${#KS[@]}
N_TASK=${#TASKS[@]}

# Export the timing env for timing-combo index $1 (used by both extraction and training).
# Always sets every SPARTA_TIME_* knob; resolve_timing/run_tag (src/utils/setup.py) pick
# the ones that matter for this combo's echo setting, so SPARTA_RUN_TAG == TIMING_LABELS[$1].
set_timing_env() {
    local t=$1
    export SPARTA_TIME_DYNAMIC=true
    export SPARTA_TIME_TYPE=days
    export SPARTA_TIME_STEP=1
    export SPARTA_ECHO="${TIMING_ECHO[$t]}"
    export SPARTA_DAYS_ECHO="${TIMING_DAYS[$t]}"
    export SPARTA_TIME_WIDTH="${TIMING_WIDTH[$t]}"
    export SPARTA_RUN_TAG="${TIMING_LABELS[$t]}"   # informational: the dir this run uses
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
    local t=$idx
    set_timing_env "$t"
    export SPARTA_K="${KS[$k_i]}"
    export SPARTA_TASK="${TASKS[$task_i]}"
}

# When executed directly (not sourced), print the array sizes as a sanity check.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "N_TIMING=$N_TIMING N_K=$N_K N_TASK=$N_TASK (datasets submitted separately: ${DATASETS[*]})"
    echo "extract array range (per dataset) : 0-$(( N_TIMING - 1 ))"
    echo "train   array range (per dataset) : 0-$(( N_TIMING * N_K * N_TASK - 1 ))"
fi
