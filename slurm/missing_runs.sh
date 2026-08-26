#!/bin/bash
# Which training array tasks still need a (re)run? Audits results/ ON DISK against the full
# (timing x K x task) grid in sweep_common.sh — the ground truth is what got written, not
# Slurm exit codes — and prints ready-to-paste `sbatch --array=...` resubmission commands.
#
# For every (dataset, cell, model) a task is reported as
#   missing : no 'Model: <name>' block in any results/experiments/<stem>_<kind>*.txt
#             (combined and per-model fan-out dumps both count) -> the task crashed,
#             timed out mid-trial, or was never submitted;
#   top-up  : metrics exist but the model's Optuna SQLite (results/tuning/<study>.db) holds
#             FEWER than n_trials finished (COMPLETE/PRUNED/FAIL) trials -> the walltime
#             timeout stopped the study short; resubmit and it resumes + the metrics are
#             rewritten at the full budget.
# Both lists are safe to resubmit: studies resume from their SQLite, which is never deleted, so
# a resubmitted cell that is already complete skips tuning and just rewrites its outputs.
# Cells NOT listed are complete (n_trials finished trials in their .db).
#
# A third bucket, 'unknown?', is for cells whose trial count could not be READ (no sqlite3 CLI,
# an unreadable .db, no n_trials in the config) — complete or short can't be told apart, so no
# command is printed; check those by hand (a needless resubmit is cheap, not a re-tune, but it
# does redo the refit). A cell with metrics and NO .db at all is counted complete: only the old
# auto-cleanup ever removed those files, and only once a study had spent its full budget.
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
# 'sqlite:///results/tuning/{study_name}.db'). Trial budgets are read from the dataset's own
# block in config/methods/config.yaml, falling back to the `defaults:` anchors — so LOWERING
# an n_trials after a sweep makes its cells read as complete, and raising it as top-up.

set -euo pipefail
shopt -s nullglob

source slurm/sweep_common.sh

MODELS="LSTM Transformer XGBoost NeuralNetwork IsolationForest"
TOTAL=$((N_TIMING * N_K * N_TASK))

# Reading a study's trial count needs the sqlite3 CLI; without it every tuned cell that has a
# .db lands in 'unknown?' (say so once, up front, rather than 24 times per model).
HAVE_SQLITE3=$(command -v sqlite3 2>/dev/null || true)
if [ -z "$HAVE_SQLITE3" ]; then
    echo "WARNING: no sqlite3 CLI on PATH -> can't count a study's trials, so every cell with a" >&2
    echo "         surviving results/tuning/*.db is reported as 'unknown?'. Load the conda env." >&2
fi

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

# Value of field $3 inside the "$1:" -> 2-space-indented "$2:" block of the methods config
# (e.g. _cfg_field defaults timeseries n_trials); empty when absent.
_cfg_field() {
    awk -v top="$1" -v sec="$2" -v key="$3" '
        $0 ~ "^"top":" {in_top=1; next}
        in_top && /^[^[:space:]#]/ {in_top=0}
        in_top && $1 == sec":" {in_sec=1; next}
        in_sec && /^  [^[:space:]#]/ {in_sec=0}
        in_sec && $1 == key":" {print $2; exit}
    ' config/methods/config.yaml | tr -d "'\""
}
# Trial budget for model $1 on dataset $2: the dataset's own override if it has one, else the
# shared `defaults:` anchor the per-dataset blocks merge in.
_budget_of() {
    local sec key v
    case $1 in
        LSTM|Transformer) sec=timeseries; key=n_trials ;;
        XGBoost)          sec=baselines;  key=n_trials ;;
        NeuralNetwork)    sec=baselines;  key=nn_n_trials ;;
        *)                echo ""; return ;;
    esac
    v=$(_cfg_field "$2" "$sec" "$key")
    [ -z "$v" ] && v=$(_cfg_field defaults "$sec" "$key")
    printf '%s' "$v"
}
# Finished (COMPLETE/PRUNED/FAIL) trials study $2 has in SQLite file $1 — the same states
# src/utils/setup.py:remaining_trials counts, so this matches what a resubmit would top up.
# Empty output = couldn't tell (no sqlite3 CLI, or an unreadable / study-less file).
_finished_trials() {
    [ -n "$HAVE_SQLITE3" ] || return 0
    sqlite3 "$1" "select count(*) from trials t join studies s on s.study_id = t.study_id \
        where s.study_name = '$2' and t.state in ('COMPLETE','PRUNED','FAIL');" 2>/dev/null || true
}

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
        budget=$(_budget_of "$model" "$ds")
        miss="" topup="" unknown=""
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
            # Metrics exist. The study's own trial count decides whether tuning finished:
            # the SQLite is kept either way now, so its mere presence proves nothing.
            study=$(_study_of "$model" "$stem")
            if [ -z "$study" ] || [ -z "$DB_TMPL" ]; then
                continue                                      # untuned model / in-memory studies
            fi
            db=$(_db_path "$study")
            if [ ! -e "$db" ]; then
                continue     # no .db: a run from when completed studies were auto-deleted at
                             # full budget, so its metrics are already at n_trials -> complete
            fi
            done_trials=$(_finished_trials "$db" "$study")
            if [ -z "$done_trials" ] || [ -z "$budget" ]; then
                unknown="${unknown},$i"                       # no sqlite3, or no budget in the config
            elif [ "$done_trials" -lt "$budget" ]; then
                topup="${topup},$i"
            fi
        done
        miss="${miss#,}" topup="${topup#,}" unknown="${unknown#,}"
        if [ -z "$miss" ] && [ -z "$topup" ] && [ -z "$unknown" ]; then
            printf '  %-16s complete\n' "$model"
            continue
        fi
        gaps=1
        printf '  %-16s missing: %-24s top-up: %-16s unknown?: %s\n' \
            "$model" "${miss:-—}" "${topup:-—}" "${unknown:-—}"
        rerun=$(_sorted "${miss}${miss:+${topup:+,}}${topup}")
        if [ -n "$rerun" ]; then
            printf '    -> sbatch --array=%s %s\n' "$rerun" "$(_submit_of "$model" "$ds")"
        fi
    done
    [ "$gaps" -eq 0 ] && echo "  all models complete for $ds"
    echo
done
