import numpy as np
import pandas as pd
import networkx as nx

from .utils.dates import exponential_time_decay
from .utils.IBM import load_transactions_ibm
from .utils.AMLSim import load_transactions_amlsim


def load_transactions(dataset, type_dataset=None):
    if dataset == 'IBM':
        if type_dataset is None:
            return load_transactions_ibm()
        return load_transactions_ibm(type_dataset=type_dataset)
    if dataset == 'AMLSim':
        return load_transactions_amlsim()
    raise ValueError(f"Unknown dataset: {dataset}")


def define_ML_labels(transactions):
    transactions_from = transactions[['from_account', 'is_laundering']]
    transactions_to = transactions[['to_account', 'is_laundering']].rename(
        columns={'to_account': 'from_account'}
    )
    transactions_labels = pd.concat([transactions_from, transactions_to], axis=0)

    accounts_labelled = transactions_labels.groupby('from_account').mean()
    accounts_labelled['is_laundering'] = (accounts_labelled['is_laundering'] > 0.1) * 1
    accounts_labelled.index.name = 'Account'
    return accounts_labelled


def load_network(transactions, echo=False):
    G = nx.DiGraph()
    for idx, row in transactions.iterrows():
        from_account = row['from_account']
        to_account = row['to_account']
        amount_trans = row['amount_trans']
        num_trans = row['num_trans']
        if echo:
            decay = row['decay']
            G.add_edge(from_account, to_account, weight=decay, amount_trans=amount_trans, num_trans=num_trans)
        else:
            G.add_edge(from_account, to_account, weight=1, amount_trans=amount_trans, num_trans=num_trans)
    return G


def construct_network(dataset='IBM', type_dataset=None):
    transactions = load_transactions(dataset, type_dataset=type_dataset)
    transactions_agg = transactions.groupby(['from_account', 'to_account']).agg({
        'amount': ['sum', 'count']
        }).reset_index()

    transactions_agg.columns = ['from_account', 'to_account', 'amount_trans', 'num_trans']
    G = load_network(transactions_agg)
    labels = define_ML_labels(transactions)
    return G, labels


def load_network_time(start_date, end_date, dataset='IBM', type_dataset=None, echo=False, days_echo=3):
    transactions = load_transactions(dataset, type_dataset=type_dataset)

    if echo:
        start_date = end_date - np.timedelta64(days_echo, 'D')
        transactions_time_filtered = transactions[
            (transactions['timestamp'] >= start_date) &
            (transactions['timestamp'] < end_date)
        ]

        transactions_time_filtered['decay'] = transactions_time_filtered['timestamp'].apply(
            lambda x: exponential_time_decay(x, end_date, days_echo=days_echo)
        )
        transactions_time_filtered_agg = transactions_time_filtered.groupby(['from_account', 'to_account']).agg({
            'decay': 'max',
            'amount': ['sum', 'count']
            }).reset_index()

        transactions_time_filtered_agg.columns = ['from_account', 'to_account', 'decay', 'amount_trans', 'num_trans']
        G = load_network(transactions_time_filtered_agg, echo=True)

        transactions_time_filtered['is_laundering'] = transactions_time_filtered['is_laundering'] * transactions_time_filtered['decay']
        labels = define_ML_labels(transactions_time_filtered)

    else:
        transactions_time_filtered = transactions[
            (transactions['timestamp'] >= start_date) &
            (transactions['timestamp'] < end_date)
        ]

        transactions_time_filtered_agg = transactions_time_filtered.groupby(['from_account', 'to_account']).agg({
            'amount': ['sum', 'count']
            }).reset_index()

        transactions_time_filtered_agg.columns = ['from_account', 'to_account', 'amount_trans', 'num_trans']

        G = load_network(transactions_time_filtered_agg)
        labels = define_ML_labels(transactions_time_filtered)
    return G, labels


def construct_network_time(start_dates, end_dates, dataset='IBM', type_dataset=None, echo=False, days_echo=3):
    networks = []
    for start_date, end_date in zip(start_dates, end_dates):
        G, labels = load_network_time(start_date, end_date, dataset=dataset, type_dataset=type_dataset, echo=echo, days_echo=days_echo)
        networks.append((G, labels))
    return networks
