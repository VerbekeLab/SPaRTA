#!/bin/bash
# Which training array tasks still need a (re)run? Audits results/ ON DISK against the full
# (timing x K x task) grid in sweep_common.sh — the ground truth is what got written, not
# Slurm exit codes — and prints ready-to-paste `sbatch --array=...` resubmission commands.
#
# For every (dataset, cell, model) a task is reported as
#   missing : no 'Model: <name>' block in any results/experiments/<stem>_<kind>*.txt
#             (combined and per-model fan-out dumps both count) -> the task crashed,
#             timed out mid-trial, or was never submitted;
#   top-up  : metrics exist but the model's Optuna SQLite (results/tuning/<study>.db) is
#             still on disk -> the study stopped short of its n_trials budget on the
#             walltime timeout (cleanup_storage keeps the file exactly then); resubmit
#             and the study resumes + the metrics are rewritten at the full budget.
# Both lists are safe to resubmit: studies resume from their SQLite. Cells NOT listed are
# complete (their .db was already cleaned up) — rerunning those would retune from scratch.
#
# Models covered and the submit command each maps to:
#   LSTM / Transformer                -> train_lstm.slurm <Dataset> lstm|transformer
#   XGBoost / IsolationForest        -> train_baseline.slurm <Dataset> xgboost|iforest
#   NeuralNetwork (MLP)              -> train_baseline_nn.slurm <Dataset>
#
# Usage (repo root, on the machine that holds results/ — i.e. the cluster):
#   bash slurm/missing_runs.sh                # all datasets
#   bash slurm/missing_runs.sh AMLSim Tide    # subset
#
# Caveats: a cell that fails BY DESIGN (temporal_split guard on an infeasible grid edit)
# keeps showing as missing — check its .err log before resubmitting in a loop. Assumes the
# timeseries and baselines blocks share one storage template (they do; both are
# 'sqlite:///results/tuning/{study_name}.db').

set -euo pipefail
shopt -s nullglob

source slurm/sweep_common.sh

MODELS="LSTM Transformer XGBoost NeuralNetwork IsolationForest"
TOTAL=$((N_TIMING * N_K * N_TASK))

# type_dataset (the filename prefix) for a dataset block in config/data/config.yaml.
_type_dataset() {
    awk -v ds="$1" '
        $0 ~ "^"ds":" {in_ds=1; next}
        in_ds && /^[A-Za-z]/ {in_ds=0}
        in_ds && $1 == "type_dataset:" {print $2; exit}
    ' config/data/config.yaml | tr -d "'\""
}

# Optuna storage path template from config/methods/config.yaml, e.g.
# results/tuning/{study_name}.db; empty -> in-memory studies, no top-up detection.
DB_TMPL=$(grep -m1 -Eo "sqlite:///[^\"' ]*\{study_name\}[^\"' ]*" config/methods/config.yaml || true)
DB_TMPL="${DB_TMPL#sqlite:///}"
_db_path() { printf '%s' "${DB_TMPL%%\{study_name\}*}$1${DB_TMPL##*\{study_name\}}"; }

_kind_of() {
    case $1 in LSTM|Transformer) echo timeseries ;; *) echo baselines ;; esac
}
_study_of() {   # study name for model $1 on stem $2 ('' = untuned, no study)
    case $1 in
        LSTM)            echo "LSTM_$2" ;;
        Transformer)     echo "Transformer_$2" ;;
        XGBoost)         echo "baseline_xgb_$2" ;;
        NeuralNetwork)   echo "baseline_nn_$2" ;;
        IsolationForest) echo "" ;;
    esac
}
_submit_of() {  # script + args resubmitting model $1 for dataset $2
    case $1 in
        LSTM)            echo "slurm/train_lstm.slurm $2 lstm" ;;
        Transformer)     echo "slurm/train_lstm.slurm $2 transformer" ;;
        XGBoost)         echo "slurm/train_baseline.slurm $2 xgboost" ;;
        NeuralNetwork)   echo "slurm/train_baseline_nn.slurm $2" ;;
        IsolationForest) echo "slurm/train_baseline.slurm $2 iforest" ;;
    esac
}
_sorted() { printf '%s\n' $(printf '%s' "$1" | tr ',' ' ') | sort -n | paste -sd, -; }

datasets=("$@")
[ ${#datasets[@]} -eq 0 ] && datasets=("${DATASETS[@]}")

for ds in "${datasets[@]}"; do
    resolve_dataset_arg "$ds"
    type=$(_type_dataset "$ds")
    if [ -z "$type" ]; then
        echo "ERROR: no type_dataset for '$ds' in config/data/config.yaml" >&2
        exit 1
    fi
    echo "=== $ds (files prefixed ${type}_; $TOTAL cells per model) ==="
    gaps=0
    for model in $MODELS; do
        kind=$(_kind_of "$model")
        miss="" topup=""
        for ((i = 0; i < TOTAL; i++)); do
            decode_train "$i"
            stem="${type}_${SPARTA_RUN_TAG}_K${SPARTA_K}_${SPARTA_TASK}"
            found=0
            for f in "results/experiments/${stem}_${kind}"*.txt; do
                if grep -qs "^Model: ${model}\$" "$f"; then found=1; break; fi
            done
            if [ "$found" -eq 0 ]; then
                miss="${miss},$i"
                continue
            fi
            study=$(_study_of "$model" "$stem")
            if [ -n "$study" ] && [ -n "$DB_TMPL" ] && [ -e "$(_db_path "$study")" ]; then
                topup="${topup},$i"
            fi
        done
        miss="${miss#,}" topup="${topup#,}"
        if [ -z "$miss" ] && [ -z "$topup" ]; then
            printf '  %-16s complete\n' "$model"
            continue
        fi
        gaps=1
        printf '  %-16s missing: %-24s top-up: %s\n' \
            "$model" "${miss:-—}" "${topup:-—}"
        rerun=$(_sorted "${miss}${miss:+${topup:+,}}${topup}")
        printf '    -> sbatch --array=%s %s\n' "$rerun" "$(_submit_of "$model" "$ds")"
    done
    [ "$gaps" -eq 0 ] && echo "  all models complete for $ds"
    echo
done
