import os

import pandas as pd

TIDE_COLUMN_MAP = {
    'src': 'from_account',
    'dest': 'to_account',
    'timestamp': 'timestamp',
    'amount': 'amount',
    'is_fraudulent': 'is_laundering',
}

# Currency converter with predefined exchange rates from original Tide code.
# Exchange rates as of May 30, 2025 (1 USD = X foreign currency)
# Source: Federal Reserve H.10, OANDA, and other financial data providers
USD_EXCHANGE_RATES = {
    'USD': 1.0000,      # Base currency
    'EUR': 0.8814,      # Euro (European Union)
    'GBP': 0.7404,      # British Pound Sterling
    'JPY': 142.61,      # Japanese Yen
    'CHF': 0.8214,      # Swiss Franc
    'AED': 3.6725,      # UAE Dirham
    'HKD': 7.8317,      # Hong Kong Dollar
    'SGD': 1.2843,      # Singapore Dollar
    'BSD': 1.0000,      # Bahamian Dollar (pegged to USD)
    'SCR': 13.7500,     # Seychellois Rupee
    'BBD': 2.0000,      # Barbadian Dollar (pegged to USD)
    'BMD': 1.0000,      # Bermudian Dollar (pegged to USD)
    'BZD': 2.0150,      # Belize Dollar
    'VUV': 118.50,      # Vanuatu Vatu
    'XCD': 2.7000,      # East Caribbean Dollar
}


def load_transactions_tide(type_dataset='HI'):
    """Return the canonical Tide transaction frame.

    The raw ``generated_edges_{type}.csv`` is ~900 MB of mostly-unused columns.
    A pre-processed Parquet (built by ``scripts/convert_tide_to_parquet.py``)
    holds exactly the canonical schema this loader produces and is ~10x smaller
    on disk and in RAM — important on the VSC cluster. Prefer it when present;
    otherwise parse the CSV with the original logic so the repo still works from
    the raw files alone. Both paths return identical values.
    """
    parquet_path = f'./data/Tide/generated_edges_{type_dataset}.parquet'
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    return _load_transactions_tide_from_csv(type_dataset=type_dataset)


def _load_transactions_tide_from_csv(type_dataset='HI'):
    columns = list(TIDE_COLUMN_MAP.keys()) + ['currency', 'edge_type']
    transactions = pd.read_csv(
        f'./data/Tide/generated_edges_{type_dataset}.csv',
        usecols=columns,
        low_memory=False,
    )

    # The Tide edge file interleaves 'transaction' and 'ownership' edges; only
    # transaction edges carry an amount/currency/timestamp, so keep those.
    transactions = transactions[transactions['edge_type'] == 'transaction']
    transactions = transactions.drop(columns='edge_type')

    #Remove transactions where source account is 'individual'.
    transactions = transactions[~transactions['src'].str.startswith('individual')]

    transactions = transactions.rename(columns=TIDE_COLUMN_MAP)
    # Drop self-loops to conform to the canonical loader contract.
    transactions = transactions[transactions['from_account'] != transactions['to_account']]
    # Tide transactions come in multiple currencies; convert them all to USD.
    rate = transactions['currency'].map(USD_EXCHANGE_RATES)
    assert not rate.isna().any(), "unknown currency in Tide transactions"
    transactions['amount'] = transactions['amount'] / rate
    transactions = transactions.drop(columns='currency')
    # Timestamps are mostly second-resolution but a minority carry microseconds
    # (e.g. '...:15.446144'); ISO8601 parses both shapes.
    transactions['timestamp'] = pd.to_datetime(transactions['timestamp'], format='ISO8601')
    # is_laundering reads back as object-dtype Python bools (the ownership rows leave
    # blanks in the column). A bare .astype(bool) is unsafe: a float NaN is truthy, so a
    # blank on a *transaction* row would flip to True. Coerce numerically and fill missing
    # with 0 (fillna MUST precede astype(bool)), mirroring AMLWorld/AMLSim.
    is_laundering = pd.to_numeric(transactions['is_laundering'], errors='coerce')
    n_missing = int(is_laundering.isna().sum())
    if n_missing:
        print(f"[Tide:{type_dataset}] WARNING: {n_missing} transaction rows have a missing "
              f"'is_fraudulent' label; treating them as non-laundering (0).")
    transactions['is_laundering'] = is_laundering.fillna(0).astype(bool)

    return transactions

