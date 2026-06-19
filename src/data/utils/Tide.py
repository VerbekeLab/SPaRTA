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
    # is_laundering reads back as object-dtype Python bools (the ownership rows
    # leave blanks in the column); normalise to a clean boolean like AMLWorld/AMLSim.
    transactions['is_laundering'] = transactions['is_laundering'].astype(bool)

    return transactions

