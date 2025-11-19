import numpy as np
import pandas as pd
import networkx as nx

from .utils.dates import exponential_time_decay

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
    transactions = transactions[transactions['Account'] != transactions['Account.1']]
    transactions['Timestamp'] = pd.to_datetime(transactions['Timestamp'], format='%Y/%m/%d %H:%M')
    transactions = transactions[transactions['Timestamp'] <= '2022-09-11']
    return transactions

def load_network_ibm(transactions, weight = 'Amount Paid'):
    G = nx.DiGraph()
    for idx, row in transactions.iterrows():
        from_account = row['Account']
        to_account = row['Account.1']
        amount = row[weight]
        G.add_edge(from_account, to_account, weight=amount)
    return G

def define_ML_labels_ibm(transactions):
    transactions_from = transactions[['Account', 'Is Laundering']]
    transactions_to = transactions[['Account.1', 'Is Laundering']]
    transactions_to = transactions_to.rename(columns={'Account.1': 'Account'})
    transactions_labels = pd.concat([transactions_from, transactions_to], axis=0)

    accounts_labelled = transactions_labels.groupby("Account").mean()
    accounts_labelled['Is Laundering'] = (accounts_labelled['Is Laundering'] > 0.1)*1

    return accounts_labelled

def construct_network_ibm(type_dataset='HI-Small'):
    transactions = load_transactions_ibm(type_dataset=type_dataset)
    G = load_network_ibm(transactions)
    labels = define_ML_labels_ibm(transactions)
    return G, labels



def load_network_ibm_time(start_date, end_date, type_dataset='HI-Small', echo=False, days_echo=3):
    transactions = load_transactions_ibm(type_dataset=type_dataset)
    
    if echo:
        start_date = end_date - np.timedelta64(days_echo, 'D')
        transactions_time_filtered = transactions[
            (transactions['Timestamp'] >= start_date) & 
            (transactions['Timestamp'] < end_date)
        ]
        
        transactions_time_filtered['decay'] = transactions_time_filtered['Timestamp'].apply(
            lambda x: exponential_time_decay(x, end_date, days_echo=days_echo)
        )
        

        transactions_time_decay = transactions_time_filtered[[
            'Account', 'Account.1', 'decay'
            ]].groupby(['Account', 'Account.1']).max().reset_index()
        
        G = load_network_ibm(transactions_time_decay, weight='decay')

        transactions_time_filtered['Is Laundering'] = transactions_time_filtered['Is Laundering']*transactions_time_filtered['decay']
        labels = define_ML_labels_ibm(transactions_time_filtered)

    else:
        transactions_time_filtered = transactions[
            (transactions['Timestamp'] >= start_date) & 
            (transactions['Timestamp'] < end_date)
        ]
        G = load_network_ibm(transactions_time_filtered)
        labels = define_ML_labels_ibm(transactions_time_filtered)
    return G, labels

def construct_network_ibm_time(start_dates, end_dates, type_dataset='HI-Small', echo=False, days_echo=3):
    networks = []
    for start_date, end_date in zip(start_dates, end_dates):
        G, labels = load_network_ibm_time(start_date, end_date, type_dataset=type_dataset, echo=echo, days_echo=days_echo)
        networks.append((G, labels))
    return networks