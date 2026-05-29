import pandas as pd

TIDE_COLUMN_MAP = {
    'src': 'from_account',
    'dest': 'to_account',
    'timestamp': 'timestamp',
    'amount': 'amount',
    'is_fraudulent': 'is_laundering',
}

def convert_to_usd(amount: float, currency: str) -> float:
    """Currency converter with predefined exchange rates from original Tide code."""

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

    fx_rate = USD_EXCHANGE_RATES[currency]
    return amount / fx_rate


def load_transactions_tide(type_dataset='HI'):
    columns = list(TIDE_COLUMN_MAP.keys()) + ['currency']
    transactions = pd.read_csv(
        f'./data/Tide/generated_transactions_{type_dataset}.csv',
        usecols=columns
    )
    transactions = transactions.rename(columns=TIDE_COLUMN_MAP)
    # Tide transactions come in multiple currencies; convert them all to USD.
    transactions['amount'] = [
        convert_to_usd(amount, currency)
        for amount, currency in zip(transactions['amount'], transactions['currency'])
    ]
    transactions = transactions.drop(columns='currency')
    transactions['timestamp'] = pd.to_datetime(transactions['timestamp'], format='%Y-%m-%d %H:%M:%S')
    return transactions

