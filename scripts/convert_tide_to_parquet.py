"""Pre-process the raw Tide edge CSVs into slim Parquet files.

The raw ``data/Tide/generated_edges_{HI,LI}.csv`` files are ~900 MB each: 11
columns of uncompressed text, 4 of which the loader never reads (two are
verbose, comma-laden strings). After filtering to transaction edges and
USD-converting, the loader only ever uses 5 columns. This script bakes that
canonical frame into a Parquet file (~10x smaller on disk, ~6x less RAM),
which ``load_transactions_tide`` prefers when present.

Run once from the repo root:

    python scripts/convert_tide_to_parquet.py

The original CSVs are left untouched (they remain the reproducible source and
the fallback). Re-running overwrites the Parquet files. Account IDs are stored
as ``category`` so the columns that dominate memory (27k unique IDs repeated
7.6M times) stay compact on load; ``amount`` stays float64 to preserve sum
precision and ``timestamp`` stays datetime64[ns] — so the loaded frame is
value-identical to parsing the CSV.
"""
import os
import sys

os.chdir("./")
sys.path.append("./")

import pandas as pd

from src.data.utils.Tide import _load_transactions_tide_from_csv


def convert(type_dataset):
    csv_path = f'./data/Tide/generated_edges_{type_dataset}.csv'
    parquet_path = f'./data/Tide/generated_edges_{type_dataset}.parquet'

    print(f'[{type_dataset}] parsing {csv_path} ...')
    df = _load_transactions_tide_from_csv(type_dataset=type_dataset)

    # Account IDs: 27k unique values repeated across 7.6M rows. Categorical
    # encoding shrinks the two columns that dominate RAM without changing the
    # string values networkx / groupby see downstream.
    df['from_account'] = df['from_account'].astype('category')
    df['to_account'] = df['to_account'].astype('category')

    df.to_parquet(parquet_path, compression='zstd', index=False)

    csv_mb = os.path.getsize(csv_path) / 1e6
    pq_mb = os.path.getsize(parquet_path) / 1e6
    print(f'[{type_dataset}] wrote {parquet_path}: '
          f'{pq_mb:.0f} MB (from {csv_mb:.0f} MB CSV, {csv_mb / pq_mb:.1f}x smaller)')

    # Integrity check: the values served via Parquet must match a fresh CSV parse.
    reloaded = pd.read_parquet(parquet_path)
    assert reloaded.shape == df.shape, f'shape mismatch: {reloaded.shape} vs {df.shape}'
    assert list(reloaded.columns) == list(df.columns), 'column order changed'
    for col in df.columns:
        left = reloaded[col].astype(object).reset_index(drop=True)
        right = df[col].astype(object).reset_index(drop=True)
        assert left.equals(right), f'value mismatch in column {col!r}'
    print(f'[{type_dataset}] integrity check passed ({df.shape[0]:,} rows).')


if __name__ == '__main__':
    for type_dataset in ('HI', 'LI'):
        convert(type_dataset)
