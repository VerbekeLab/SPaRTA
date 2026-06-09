import pandas as pd

IBM_COLUMN_MAP = {
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

def load_transactions_ibm(type_dataset='HI-Small'):
    dtype = {
                'Timestamp': 'object',
                'From Bank': 'object',
                'Account': 'object',
                'To Bank': 'object',
                'Account.1': 'object',
                'Amount Received': 'float64',
                'Receiving Currency': 'object',
                'Amount Paid': 'float64',
                'Payment Currency': 'object',
                'Payment Format': 'object',
                'Is Laundering': 'bool'
            }

    transactions = pd.read_csv(
        f'./data/IBM/{type_dataset}_Trans.csv',
        dtype=dtype
    )
    transactions = transactions.rename(columns=IBM_COLUMN_MAP)
    transactions = transactions[transactions['from_account'] != transactions['to_account']]
    rate = transactions['Payment Currency'].map(EXCHANGE_NAME_TO_CODE).map(USD_EXCHANGE_RATES)
    assert not rate.isna().any(), "unknown currency in IBM transactions"
    transactions['amount'] = transactions['amount'] / rate
    transactions['timestamp'] = pd.to_datetime(transactions['timestamp'], format='%Y/%m/%d %H:%M')
    transactions = transactions[transactions['timestamp'] <= '2022-09-11']
    return transactions
