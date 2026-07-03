"""Disk I/O for the extracted feature/label tables under results/features/.

New tables are written as zstd-compressed Parquet (~5-10x smaller than the
plain CSVs, which eat the VSC disk quota on long dynamic sweeps). Readers
resolve a path *stem* (no extension): they prefer <stem>.parquet and fall
back to a legacy <stem>.csv, so feature sets extracted before the switch
keep working without regeneration. Values are unchanged either way — Parquet
stores the float64 measures losslessly, CSV round-trips them through text.
"""
import os

import pandas as pd


def save_table(df, stem):
    """Write ``df`` to ``<stem>.parquet`` (zstd, no index) and return the path.

    Frames indexed by a meaningful key (e.g. the labels' Account index) must be
    ``reset_index()``-ed by the caller so the key survives as a column.
    """
    # Decategorize before writing (Tide account IDs arrive categorical from the
    # transaction Parquet cache). Parquet round-trips category dtype, but the
    # legacy CSVs served plain values — keep readers' merge/reindex semantics.
    cat_cols = df.select_dtypes(include="category").columns
    if len(cat_cols):
        df = df.copy()
        for col in cat_cols:
            df[col] = df[col].astype(object)

    path = f"{stem}.parquet"
    df.to_parquet(path, compression="zstd", index=False)
    return path


def find_table(stem):
    """Return the existing path for ``stem`` (Parquet preferred, legacy CSV
    fallback) or None if neither file exists."""
    for path in (f"{stem}.parquet", f"{stem}.csv"):
        if os.path.exists(path):
            return path
    return None


def load_table(stem, **csv_kwargs):
    """Load ``<stem>.parquet`` if present, else legacy ``<stem>.csv``.

    Both come back as a plain range-indexed DataFrame (the labels CSVs were
    written with their Account index as the first column, which read_csv
    already surfaces as a regular column). ``csv_kwargs`` (e.g. a ``dtype``
    map) only apply on the CSV path — Parquet already stores exact dtypes,
    so it needs no parse hints.
    """
    path = find_table(stem)
    if path is None:
        raise FileNotFoundError(f"No table found for stem {stem!r} "
                                f"(tried {stem}.parquet and {stem}.csv)")
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, **csv_kwargs)
