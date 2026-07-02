import numpy as np
import pandas as pd
import networkx as nx

from .utils.dates import exponential_time_decay
from .utils.AMLWorld import load_transactions_amlworld
from .utils.AMLSim import load_transactions_amlsim
from .utils.Tide import load_transactions_tide


def load_transactions(dataset, type_dataset=None):
    if dataset == 'AMLWorld':
        if type_dataset is None:
            return load_transactions_amlworld()
        return load_transactions_amlworld(type_dataset=type_dataset)
    if dataset == 'AMLSim':
        return load_transactions_amlsim()
    if dataset == 'Tide':
        if type_dataset is None:
            return load_transactions_tide()
        return load_transactions_tide(type_dataset=type_dataset)
    raise ValueError(f"Unknown dataset: {dataset}")


def define_ML_labels(transactions):
    transactions_from = transactions[['from_account', 'is_laundering']]
    transactions_to = transactions[['to_account', 'is_laundering']].rename(
        columns={'to_account': 'from_account'}
    )
    transactions_labels = pd.concat([transactions_from, transactions_to], axis=0)

    # observed=True so a time-windowed slice of categorical account IDs (Tide
    # Parquet) labels only the accounts actually present, not every category;
    # a no-op for object-dtype loaders.
    accounts_labelled = transactions_labels.groupby('from_account', observed=True).mean()
    # A node is positive if ANY of its in/out edges are laundering (fraction > 0), not a
    # >10% share. This '> 0' rule is intentional and authoritative (see CLAUDE.md "Node
    # labels"); don't "fix" it to > 0.1 without a paper-level reason.
    accounts_labelled['is_laundering'] = (accounts_labelled['is_laundering'] > 0) * 1
    accounts_labelled.index.name = 'Account'
    return accounts_labelled


def load_network(transactions, echo=False):
    edges = transactions.copy()
    edges['weight'] = edges['decay'] if echo else 1
    return nx.from_pandas_edgelist(
        edges, 'from_account', 'to_account',
        edge_attr=['weight', 'amount_trans', 'num_trans'],
        create_using=nx.DiGraph,
    )


def construct_network(dataset='AMLWorld', type_dataset=None):
    transactions = load_transactions(dataset, type_dataset=type_dataset)
    # observed=True so categorical account IDs (Tide Parquet) don't expand to the
    # full ID×ID cartesian product; a no-op for object-dtype loaders.
    transactions_agg = transactions.groupby(['from_account', 'to_account'], observed=True).agg({
        'amount': ['sum', 'count']
        }).reset_index()

    transactions_agg.columns = ['from_account', 'to_account', 'amount_trans', 'num_trans']
    G = load_network(transactions_agg)
    labels = define_ML_labels(transactions)
    return G, labels


def _window_network(transactions, start_date, end_date, echo=False, days_echo=3):
    """Build one snapshot (G, labels) from an already-loaded transactions frame."""
    if echo:
        start_date = end_date - pd.Timedelta(days=days_echo) # Start date is set to be the end date minus the number of days for the echo effect
        transactions_time_filtered = transactions[
            (transactions['timestamp'] >= start_date) &
            (transactions['timestamp'] < end_date)
        ].copy()

        # Vectorized form of exponential_time_decay: truncate the lag to whole seconds
        # (matching the old `.astype('timedelta64[s]').astype(int)`) then apply the decay.
        gamma = -np.log(0.01) / days_echo
        delta_days = ((end_date - transactions_time_filtered['timestamp']) // pd.Timedelta(seconds=1)) / (3600 * 24)
        transactions_time_filtered['decay'] = np.exp(-gamma * delta_days)
        transactions_time_filtered_agg = transactions_time_filtered.groupby(['from_account', 'to_account'], observed=True).agg({
            'decay': 'max',
            'amount': ['sum', 'count']
            }).reset_index()

        transactions_time_filtered_agg.columns = ['from_account', 'to_account', 'decay', 'amount_trans', 'num_trans']
        G = load_network(transactions_time_filtered_agg, echo=True)

        #transactions_time_filtered['is_laundering'] = transactions_time_filtered['is_laundering'] * transactions_time_filtered['decay']
        labels = define_ML_labels(transactions_time_filtered)

    else:
        transactions_time_filtered = transactions[
            (transactions['timestamp'] >= start_date) &
            (transactions['timestamp'] < end_date)
        ]

        transactions_time_filtered_agg = transactions_time_filtered.groupby(['from_account', 'to_account'], observed=True).agg({
            'amount': ['sum', 'count']
            }).reset_index()

        transactions_time_filtered_agg.columns = ['from_account', 'to_account', 'amount_trans', 'num_trans']

        G = load_network(transactions_time_filtered_agg)
        labels = define_ML_labels(transactions_time_filtered)
    return G, labels


def load_network_time(start_date, end_date, dataset='AMLWorld', type_dataset=None, echo=False, days_echo=3):
    transactions = load_transactions(dataset, type_dataset=type_dataset)
    return _window_network(transactions, start_date, end_date, echo=echo, days_echo=days_echo)


def construct_network_time_iter(start_dates, end_dates, dataset='AMLWorld', type_dataset=None,
                                echo=False, days_echo=3, transactions=None):
    """
    Lazily yield one (G, labels) snapshot per time window.

    Loads the transactions ONCE up front (or reuses a caller-provided frame via
    ``transactions``) instead of re-reading the raw files for every window, and keeps
    only the current snapshot's graph alive — the eager construct_network_time holds
    every snapshot DiGraph simultaneously, which OOMs long dynamic runs (~370
    snapshots for Tide at daily step=1).
    """
    if transactions is None:
        transactions = load_transactions(dataset, type_dataset=type_dataset)
    for start_date, end_date in zip(start_dates, end_dates):
        # Yield without binding to a local name so the generator holds no reference
        # to the previous snapshot's graph while the caller processes it.
        yield _window_network(transactions, start_date, end_date, echo=echo, days_echo=days_echo)


def construct_network_time(start_dates, end_dates, dataset='AMLWorld', type_dataset=None, echo=False, days_echo=3):
    # Eager variant: fine for a handful of windows, but materialises ALL snapshot
    # graphs at once — prefer construct_network_time_iter for long dynamic sweeps.
    return list(construct_network_time_iter(
        start_dates, end_dates, dataset=dataset, type_dataset=type_dataset, echo=echo, days_echo=days_echo
    ))
