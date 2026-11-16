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
    # Drop self-loops to conform to the canonical loader contract (mirrors AMLWorld/Tide).
    transactions = transactions[transactions['from_account'] != transactions['to_account']]
    # Coerce the label to a clean boolean, mirroring AMLWorld/Tide. IS_FRAUD reads back as
    # bool on the clean file, but a blank cell in a staged copy would leave the column
    # object-dtype with NaN; fillna(0) MUST precede astype(bool) since a float NaN is
    # truthy and would otherwise flip a missing label to True. print (not warnings.warn)
    # so the notice survives measure_calculation.py's warnings.simplefilter('ignore').
    is_laundering = pd.to_numeric(transactions['is_laundering'], errors='coerce')
    n_missing = int(is_laundering.isna().sum())
    if n_missing:
        print(f"[AMLSim] WARNING: {n_missing} rows have a missing 'IS_FRAUD' label; "
              f"treating them as non-laundering (0).")
    transactions['is_laundering'] = is_laundering.fillna(0).astype(bool)
    # AMLSim TIMESTAMP is an integer step; treat it as days from epoch so the
    # same datetime-based filtering and decay logic works as for AMLWorld.
    transactions['timestamp'] = pd.to_datetime(transactions['timestamp'], unit='D')
    return transactions
