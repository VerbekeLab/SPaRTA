import pandas as pd

AMLSIM_COLUMN_MAP = {
    'SENDER_ACCOUNT_ID': 'from_account',
    'RECEIVER_ACCOUNT_ID': 'to_account',
    'TIMESTAMP': 'timestamp',
    'TX_AMOUNT': 'amount',
    'IS_FRAUD': 'is_laundering',
}

def load_transactions_amlsim():
    columns = list(AMLSIM_COLUMN_MAP.keys()) + ['TX_ID']
    transactions = pd.read_csv(
        './data/amlsim/transactions.csv',
        index_col=0,
        usecols=columns
    )
    transactions = transactions.rename(columns=AMLSIM_COLUMN_MAP)
    # AMLSim TIMESTAMP is an integer step; treat it as days from epoch so the
    # same datetime-based filtering and decay logic works as for AMLWorld.
    transactions['timestamp'] = pd.to_datetime(transactions['timestamp'], unit='D')
    return transactions
