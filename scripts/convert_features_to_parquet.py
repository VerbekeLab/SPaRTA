"""Convert already-extracted feature/label CSVs to zstd Parquet in place.

measure_calculation.py now writes results/features/ tables as Parquet, and all
readers (sequence_data, experiment_features, experiment_CNN, feature_data)
prefer <stem>.parquet with a legacy <stem>.csv fallback. This script migrates
feature sets extracted *before* the switch — typically ~5-10x smaller on disk —
so long dynamic sweeps don't have to be recomputed to free VSC quota.

Run from the repo root:

    python scripts/convert_features_to_parquet.py            # convert, keep CSVs
    python scripts/convert_features_to_parquet.py --delete   # convert, then remove CSVs

Walks results/features/ recursively (run_tag subdirectories included) and
converts every *_features*.csv / *_labels*.csv. Each Parquet file is reloaded
and checked value-identical to its CSV parse before the CSV is (optionally)
deleted; files whose Parquet twin already exists are skipped. Other CSVs in
the tree are left untouched.
"""
import argparse
import os
import sys

os.chdir("./")
sys.path.append("./")

import pandas as pd

from src.utils.feature_io import save_table


def find_csvs(root):
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            stem, ext = os.path.splitext(name)
            if ext == ".csv" and ("_features" in stem or "_labels" in stem):
                yield os.path.join(dirpath, stem)


def convert(stem, delete):
    csv_path = f"{stem}.csv"
    parquet_path = f"{stem}.parquet"
    if os.path.exists(parquet_path):
        print(f"skip (exists)  {parquet_path}")
        return 0, 0

    # Plain read_csv: the same dtype inference every reader applied to these
    # files, so the Parquet serves exactly the values consumers saw before.
    df = pd.read_csv(csv_path)
    save_table(df, stem)

    reloaded = pd.read_parquet(parquet_path)
    pd.testing.assert_frame_equal(reloaded, df, check_exact=True)

    csv_bytes = os.path.getsize(csv_path)
    pq_bytes = os.path.getsize(parquet_path)
    print(f"converted      {csv_path}: {csv_bytes / 1e6:.1f} MB -> "
          f"{pq_bytes / 1e6:.1f} MB ({csv_bytes / pq_bytes:.1f}x smaller)")

    if delete:
        os.remove(csv_path)
    return csv_bytes, pq_bytes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="results/features",
                        help="directory tree to convert (default: results/features)")
    parser.add_argument("--delete", action="store_true",
                        help="remove each CSV after its Parquet passes the integrity check")
    args = parser.parse_args()

    total_csv = total_pq = n = 0
    for stem in find_csvs(args.root):
        csv_bytes, pq_bytes = convert(stem, args.delete)
        if csv_bytes:
            total_csv += csv_bytes
            total_pq += pq_bytes
            n += 1

    if n:
        freed = " (CSVs deleted)" if args.delete else " (CSVs kept; re-run with --delete to free the space)"
        print(f"\n{n} files: {total_csv / 1e6:.0f} MB CSV -> {total_pq / 1e6:.0f} MB Parquet, "
              f"{(total_csv - total_pq) / 1e6:.0f} MB reclaimable{freed}")
    else:
        print("Nothing to convert.")
