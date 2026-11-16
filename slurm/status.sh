#!/bin/bash
# READ-ONLY progress snapshot of the SPaRTA sweep. Submits nothing, deletes nothing, and the
# only file it writes is results/experiments/summary.csv (via collect_results.py, which is
# idempotent). Run from the repo root on the VSC login node:
#   bash slurm/status.sh                 # all datasets
#   bash slurm/status.sh AMLSim Tide     # subset
# and paste the whole output back.
#
# Sections: 1 code state | 2 queue + job outcomes | 3 log errors | 4 feature extraction
#           5 window stores | 6 training audit (missing_runs.sh) | 7 results headline | 8 disk

cd "$(dirname "$0")/.." || exit 1
source slurm/sweep_common.sh

# sqlite3 + pandas live in the conda env the jobs use; load it quietly if it is not active.
if ! command -v sqlite3 >/dev/null 2>&1 || ! python -c "import pandas" >/dev/null 2>&1; then
    export PATH="${VSC_DATA:-$HOME}/miniconda3/bin:${PATH}"
    source activate SPaRTA_env >/dev/null 2>&1 || true
fi

datasets=("$@")
[ ${#datasets[@]} -eq 0 ] && datasets=("${DATASETS[@]}")

hdr() { printf '\n==================== %s ====================\n' "$1"; }

# type_dataset (filename prefix) and expected snapshot count T per dataset.
type_of() { awk -v ds="$1" '$0 ~ "^"ds":" {f=1; next} f && /^[A-Za-z]/ {f=0}
            f && $1=="type_dataset:" {print $2; exit}' config/data/config.yaml | tr -d "'\""; }
expected_T() { case $1 in AMLWorld) echo 39 ;; AMLSim) echo 199 ;; Tide) echo 181 ;; *) echo ? ;; esac; }

hdr "1. CODE STATE  ($(date '+%F %H:%M') on $(hostname -s))"
git rev-parse --abbrev-ref HEAD
git log -3 --format='%h %ad %s' --date=short
echo "uncommitted:"; git status --short | head -10
echo "(this checkout is what the jobs run — compare the latest commit with your laptop's)"

hdr "2. QUEUE + RECENT JOB OUTCOMES (last 60 days, allocations only)"
if command -v squeue >/dev/null 2>&1; then
    echo "-- now (state counts):"
    squeue -M wice -u "$USER" -h -o "%j %T" 2>/dev/null | sort | uniq -c
    echo "-- pending reasons (first 8 rows):"
    squeue -M wice -u "$USER" -t PENDING -h -o "%.14i %.18j %R" 2>/dev/null | head -8
    echo "-- outcomes per job name (state | count):"
    sacct -M wice -u "$USER" -S "$(date -d '60 days ago' +%F 2>/dev/null || echo 2026-08-20)" \
          -X -n -P --format=JobName,State 2>/dev/null \
        | sed -E 's/ by [0-9]+//' | sort | uniq -c | sort -k2
else
    echo "(squeue not found — run this on the VSC login node)"
fi

hdr "3. ERRORS IN slurm/logs (files touched in the last 60 days)"
if [ -d slurm/logs ]; then
    for pat in "Traceback" "OutOfMemory|out of memory|oom_kill|Killed" "DUE TO TIME LIMIT" "CUDA error|CUDA out"; do
        printf '%-45s' "$pat:"
        find slurm/logs -name '*.err' -mtime -60 -print0 2>/dev/null \
            | xargs -0 grep -lE "$pat" 2>/dev/null | wc -l
    done
    echo "-- by script, files with a Traceback:"
    find slurm/logs -name '*.err' -mtime -60 -print0 2>/dev/null | xargs -0 grep -lE "Traceback" 2>/dev/null \
        | sed -E 's#.*/##; s/_[0-9]+_[0-9]+\.err$//; s/_[0-9]+\.err$//' | sort | uniq -c
    echo "-- last line of the 5 newest Tracebacks:"
    for f in $(find slurm/logs -name '*.err' -mtime -60 -print0 2>/dev/null | xargs -0 grep -lE "Traceback" 2>/dev/null \
               | xargs -r ls -t 2>/dev/null | head -5); do
        echo "$f: $(grep -v '^\s*$' "$f" | tail -1 | cut -c1-150)"
    done
fi

hdr "4. FEATURE EXTRACTION  (dynamic tables on disk vs expected snapshots)"
for ds in "${datasets[@]}"; do
    SPARTA_DATASET=$ds; type=$(type_of "$ds"); T=$(expected_T "$ds")
    echo "$ds (prefix ${type}_, expected T~$T):"
    n_static=$(ls results/features/ 2>/dev/null | grep -cE "^${type}_static_features\.(parquet|csv)$")
    echo "  static features present: $([ "$n_static" -gt 0 ] && echo yes || echo NO)"
    for ((t = 0; t < N_TIMING; t++)); do
        tag=$(_grid TIMING_LABELS "$t")
        nf=$(ls "results/features/$tag" 2>/dev/null | grep -E "^${type}_dynamic_[0-9]+_features" | sed -E 's/\.(parquet|csv)$//' | sort -u | wc -l)
        nl=$(ls "results/features/$tag" 2>/dev/null | grep -E "^${type}_dynamic_[0-9]+_labels"   | sed -E 's/\.(parquet|csv)$//' | sort -u | wc -l)
        printf '  %-10s features %4s / labels %4s  of ~%s\n' "$tag" "$nf" "$nl" "$T"
    done
done

hdr "5. WINDOW STORES  (\$VSC_SCRATCH/SPaRTA/windows/<tag> — transient, may have been purged)"
root="${SPARTA_WINDOW_STORE:-}"
if [ -z "$root" ]; then
    if [ -n "${VSC_SCRATCH:-}" ]; then root="$VSC_SCRATCH/SPaRTA/windows"; else root="results/windows"; fi
fi
echo "root: $root"
for ds in "${datasets[@]}"; do
    SPARTA_DATASET=$ds
    for ((t = 0; t < N_TIMING; t++)); do
        tag=$(_grid TIMING_LABELS "$t")
        meta=$(ls "$root/$tag"/*_meta.json 2>/dev/null | head -1)
        if [ -n "$meta" ]; then
            printf '  %-8s %-10s OK   %s\n' "$ds" "$tag" "$(python -c "import json,sys;m=json.load(open('$meta'));print('T=%s rows=%s nodes=%s'%(m['T'],m['n_rows'],m['n_nodes']))" 2>/dev/null)"
        else
            printf '  %-8s %-10s MISSING\n' "$ds" "$tag"
        fi
    done
done

hdr "6. TRAINING AUDIT  (slurm/missing_runs.sh — what is missing / short of n_trials)"
bash slurm/missing_runs.sh "${datasets[@]}" 2>&1 | head -120

hdr "7. RESULTS HEADLINE  (results/experiments/summary.csv, rebuilt now)"
python scripts/collect_results.py 2>&1 | head -4
python - <<'PY'
import os
import pandas as pd
p = "results/experiments/summary.csv"
if not os.path.exists(p):
    raise SystemExit("no summary.csv")
d = pd.read_csv(p)
if d.empty:
    raise SystemExit("summary.csv is empty")
pd.set_option("display.width", 200)
dyn = d[d.run_tag != "static"].copy()
print("\ncells with results (dataset x model x task; 12 per task = complete, 24 per model):")
if len(dyn):
    c = (dyn.groupby(["dataset", "model", "task"]).size().unstack("task").fillna(0).astype(int))
    print(c.to_string())
    print("\nAUC-PR on the test band: best / median over cells (dynamic models):")
    g = dyn.groupby(["dataset", "model", "task"]).AUC_PR
    print(pd.DataFrame({"best": g.max(), "median": g.median(), "worst": g.min()}).round(3).to_string())
    print("\nbest cell per dataset x task:")
    idx = dyn.dropna(subset=["AUC_PR"]).groupby(["dataset", "task"]).AUC_PR.idxmax()
    best = dyn.loc[idx, ["dataset", "task", "model", "run_tag", "K", "AUC_PR", "AUC_ROC"]].copy()
    best["K"] = best["K"].astype(int)
    print(best.round(3).to_string(index=False))
st = d[d.run_tag == "static"]
if len(st):
    print("\nstatic models (test AUC-PR / AUC-ROC):")
    print(st[["dataset", "model", "AUC_PR", "AUC_ROC"]].round(3).to_string(index=False))
PY

hdr "8. DISK / QUOTA"
command -v myquota >/dev/null 2>&1 && myquota 2>&1 | head -12
du -sh results 2>/dev/null
[ -n "${VSC_SCRATCH:-}" ] && du -sh "$VSC_SCRATCH/SPaRTA" 2>/dev/null
echo; echo "done."
