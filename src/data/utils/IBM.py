import pandas as pd

IBM_COLUMN_MAP = {
    'Account': 'from_account',
    'Account.1': 'to_account',
    'Timestamp': 'timestamp',
    'Amount Paid': 'amount',
    'Is Laundering': 'is_laundering',
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
    transactions['timestamp'] = pd.to_datetime(transactions['timestamp'], format='%Y/%m/%d %H:%M')
    transactions = transactions[transactions['timestamp'] <= '2022-09-11']
    return transactions
