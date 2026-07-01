import pandas as pd

AMLWORLD_COLUMN_MAP = {
    'Account': 'from_account',
    'Account.1': 'to_account',
    'Timestamp': 'timestamp',
    'Amount Paid': 'amount',
    'Is Laundering': 'is_laundering',
}

# Currency conversion tables (predefined rates from original Tide code).
# Exchange rates as of May 30, 2025 (1 USD = X foreign currency)
# Source: Federal Reserve H.10, OANDA, and other financial data providers
EXCHANGE_NAME_TO_CODE = {
    "US Dollar": "USD",
    "Euro": "EUR",
    "Australian Dollar": "AUD",
    "Swiss Franc": "CHF",
    "Yuan": "CNY",
    "Shekel": "ILS",
    "Rupee": "INR",
    "UK Pound": "GBP",
    "Yen": "JPY",
    "Ruble": "RUB",
    "Canadian Dollar": "CAD",
    "Mexican Peso": "MXN",
    "Saudi Riyal": "SAR",
    "Brazil Real": "BRL",
    "Bitcoin": "BTC",
}

USD_EXCHANGE_RATES = {
    'USD': 1.0000,      # Base currency
    'AUD': 1.4128,      # Australian Dollar
    'BTC': 0.0001,      # Bitcoin
    'BRL': 5.6465,      # Brazilian Real
    'CAD': 1.3193,      # Canadian Dollar
    'EUR': 0.8534,      # Euro
    'MXN': 21.1431,     # Mexican Peso
    'RUB': 77.804,      # Russian Ruble
    'INR': 73.444,      # Indian Rupee
    'SAR': 3.7511,      # Saudi Riyal
    'ILS': 3.377,       # Israeli Shekel
    'CHF': 0.915,       # Swiss Franc
    'GBP': 0.7742,      # British Pound Sterling
    'JPY': 105.4,       # Japanese Yen
    'CNY': 6.6976       # Chinese Yuan
}

def load_transactions_amlworld(type_dataset='HI-Small'):
    dtype = {
                'Timestamp': 'object',
                'From Bank': 'object',
                'Account': 'object',
                'To Bank': 'object',
                'Account.1': 'object',
                'Amount Received': 'float64',
                'Receiving Currency': 'object',
                'Payment Currency': 'object',
                'Payment Format': 'object',
            }

    transactions = pd.read_csv(
        f'./data/AMLWorld/{type_dataset}_Trans.csv',
        dtype=dtype
    )
    # Coerce the label to a clean boolean. errors='coerce' turns any blank/non-numeric
    # cell into NaN; fillna(0) treats a missing label as non-laundering. The fillna MUST
    # precede astype(bool) — a float NaN is truthy, so casting it directly would flip a
    # missing label to True. print (not warnings.warn) because measure_calculation.py sets
    # warnings.simplefilter('ignore'), which would swallow a warning in the Slurm log.
    is_laundering = pd.to_numeric(transactions['Is Laundering'], errors='coerce')
    n_missing = int(is_laundering.isna().sum())
    if n_missing:
        print(f"[AMLWorld:{type_dataset}] WARNING: {n_missing} rows have a missing "
              f"'Is Laundering' label; treating them as non-laundering (0).")
    transactions['Is Laundering'] = is_laundering.fillna(0).astype(bool)
    transactions = transactions.rename(columns=AMLWORLD_COLUMN_MAP)
    transactions = transactions[transactions['from_account'] != transactions['to_account']]

    # Coerce amount and drop rows with a missing/non-numeric value: a NaN amount would
    # otherwise collapse into a phantom zero-weight, zero-count edge in construct_network's
    # groupby.agg, indistinguishable from a genuine zero-amount transaction.
    transactions['amount'] = pd.to_numeric(transactions['amount'], errors='coerce')
    n_bad_amount = int(transactions['amount'].isna().sum())
    if n_bad_amount:
        print(f"[AMLWorld:{type_dataset}] WARNING: {n_bad_amount} rows have a missing/"
              f"non-numeric 'Amount Paid'; dropping them.")
        transactions = transactions[transactions['amount'].notna()]

    # Convert to USD. Explicit guard rather than assert: an assert is stripped under
    # python -O (letting NaN rates silently corrupt amounts), and a blank/unknown currency
    # should name the offending value(s) instead of failing with a bare message.
    rate = transactions['Payment Currency'].map(EXCHANGE_NAME_TO_CODE).map(USD_EXCHANGE_RATES)
    if rate.isna().any():
        bad = sorted(transactions.loc[rate.isna(), 'Payment Currency'].dropna().unique())
        raise ValueError(f"unmapped Payment Currency values in AMLWorld {type_dataset}: {bad}")
    transactions['amount'] = transactions['amount'] / rate
    transactions['timestamp'] = pd.to_datetime(transactions['timestamp'], format='%Y/%m/%d %H:%M')
    transactions = transactions[transactions['timestamp'] <= '2022-09-11']
    return transactions
