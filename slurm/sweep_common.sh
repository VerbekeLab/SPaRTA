#!/bin/bash
# Shared sweep grid + array-index decoders for the SPaRTA timing sweep. SOURCED (not
# executed) by the extraction and training Slurm scripts so the grid is defined ONCE — the
# timing tags the training jobs look for are exactly the ones the extraction jobs produce.
#
# Edit the grid HERE and keep every script's --array range in sync with the counts:
#   extract_features_dynamic.slurm : 0 .. N_DATA*N_TIMING-1                (= 11 for 3*4)
#   train_lstm.slurm/train_baseline.slurm : 0 .. N_DATA*N_TIMING*N_K*N_TASK-1 (= 71 for 3*4*3*2)
# The asserts at the bottom print the counts; run `bash slurm/sweep_common.sh` to see them.

# Datasets (the SPARTA_DATASET values; their type_dataset comes from config/data/config.yaml).
DATASETS=(AMLWorld AMLSim Tide)

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

N_DATA=${#DATASETS[@]}
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

# Decode a 2D extraction index (dataset × timing) from $1; sets SPARTA_DATASET + timing env.
decode_extract() {
    local idx=$1
    local t=$(( idx % N_TIMING ))
    local d=$(( idx / N_TIMING ))
    export SPARTA_DATASET="${DATASETS[$d]}"
    set_timing_env "$t"
}

# Decode a 4D training index (dataset × timing × K × task) from $1; sets SPARTA_DATASET,
# timing env, SPARTA_K, SPARTA_TASK. Mixed-radix, task fastest then K then timing then data.
decode_train() {
    local idx=$1
    local task_i=$(( idx % N_TASK )); idx=$(( idx / N_TASK ))
    local k_i=$(( idx % N_K ));       idx=$(( idx / N_K ))
    local t=$(( idx % N_TIMING ));    idx=$(( idx / N_TIMING ))
    local d=$idx
    export SPARTA_DATASET="${DATASETS[$d]}"
    set_timing_env "$t"
    export SPARTA_K="${KS[$k_i]}"
    export SPARTA_TASK="${TASKS[$task_i]}"
}

# When executed directly (not sourced), print the array sizes as a sanity check.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "N_DATA=$N_DATA N_TIMING=$N_TIMING N_K=$N_K N_TASK=$N_TASK"
    echo "extract array range : 0-$(( N_DATA * N_TIMING - 1 ))"
    echo "train   array range : 0-$(( N_DATA * N_TIMING * N_K * N_TASK - 1 ))"
fi
