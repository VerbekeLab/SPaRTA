"""Vendored networkx Louvain (3.5; byte-identical in 3.6.1) with ONE change: the
node-sweep loop inside ``_one_level`` is capped at ``max_sweeps`` full passes.

Why this exists: ``nx.community.louvain_communities`` can loop FOREVER on graphs with
heterogeneous float edge weights — exactly what echo mode produces (exponential-decay
weights in (0.01, 1]). In exact arithmetic every accepted move strictly increases
modularity, so the ``while nb_moves > 0`` sweep loop must terminate; with floats,
rounding error in the gain formula (``remove_cost + wt/m - resolution*(Stot*deg)/(2m**2)``)
can make a handful of nodes flip-flop between two communities with a spuriously
positive gain each way, so ``nb_moves`` never reaches 0. The bug is upstream and still
present in networkx 3.6.1; with ``seed=1997`` it is fully deterministic — the AMLSim
d1_echo3 feature extraction hung on snapshot 99/199 every single run.

The cap only changes behaviour on inputs where upstream never returns at all: every
terminating run makes the same RNG calls in the same order and yields a bit-identical
partition (normal snapshots converge in well under 100 sweeps; the cap is 1000). When
the cap trips, the partition after the last completed sweep is used — at that point the
sweeps are only shuffling float noise, so it is as converged as the algorithm can get.

Keep this file in lockstep with the upstream code (docstrings stripped, backend
dispatch decorator dropped); the only functional edit is marked with ``SPARTA CAP``.
"""

import itertools
from collections import defaultdict, deque

import networkx as nx
from networkx.algorithms.community import modularity
from networkx.utils import py_random_state

__all__ = ["louvain_communities_capped"]

DEFAULT_MAX_SWEEPS = 1000


@py_random_state("seed")
def louvain_communities_capped(
    G,
    weight="weight",
    resolution=1,
    threshold=0.0000001,
    max_level=None,
    seed=None,
    max_sweeps=DEFAULT_MAX_SWEEPS,
):
    """Drop-in ``nx.community.louvain_communities`` with a bounded inner loop."""
    partitions = _louvain_partitions(G, weight, resolution, threshold, seed, max_sweeps)
    if max_level is not None:
        if max_level <= 0:
            raise ValueError("max_level argument must be a positive integer or None")
        partitions = itertools.islice(partitions, max_level)
    final_partition = deque(partitions, maxlen=1)
    return final_partition.pop()


def _louvain_partitions(G, weight, resolution, threshold, seed, max_sweeps):
    partition = [{u} for u in G.nodes()]
    if nx.is_empty(G):
        yield partition
        return
    mod = modularity(G, partition, resolution=resolution, weight=weight)
    is_directed = G.is_directed()
    if G.is_multigraph():
        graph = _convert_multigraph(G, weight, is_directed)
    else:
        graph = G.__class__()
        graph.add_nodes_from(G)
        graph.add_weighted_edges_from(G.edges(data=weight, default=1))

    m = graph.size(weight="weight")
    partition, inner_partition, improvement = _one_level(
        graph, m, partition, resolution, is_directed, seed, max_sweeps
    )
    improvement = True
    while improvement:
        # gh-5901 protect the sets in the yielded list from further manipulation here
        yield [s.copy() for s in partition]
        new_mod = modularity(
            graph, inner_partition, resolution=resolution, weight="weight"
        )
        if new_mod - mod <= threshold:
            return
        mod = new_mod
        graph = _gen_graph(graph, inner_partition)
        partition, inner_partition, improvement = _one_level(
            graph, m, partition, resolution, is_directed, seed, max_sweeps
        )


def _one_level(G, m, partition, resolution, is_directed, seed, max_sweeps):
    node2com = {u: i for i, u in enumerate(G.nodes())}
    inner_partition = [{u} for u in G.nodes()]
    if is_directed:
        in_degrees = dict(G.in_degree(weight="weight"))
        out_degrees = dict(G.out_degree(weight="weight"))
        Stot_in = list(in_degrees.values())
        Stot_out = list(out_degrees.values())
        # Calculate weights for both in and out neighbors without considering self-loops
        nbrs = {}
        for u in G:
            nbrs[u] = defaultdict(float)
            for _, n, wt in G.out_edges(u, data="weight"):
                if u != n:
                    nbrs[u][n] += wt
            for n, _, wt in G.in_edges(u, data="weight"):
                if u != n:
                    nbrs[u][n] += wt
    else:
        degrees = dict(G.degree(weight="weight"))
        Stot = list(degrees.values())
        nbrs = {u: {v: data["weight"] for v, data in G[u].items() if v != u} for u in G}
    rand_nodes = list(G.nodes)
    seed.shuffle(rand_nodes)
    nb_moves = 1
    improvement = False
    n_sweeps = 0
    while nb_moves > 0:
        # SPARTA CAP: bound the sweep loop (see module docstring). Breaking between
        # sweeps leaves partition/inner_partition/node2com in a consistent state.
        if n_sweeps >= max_sweeps:
            print(
                f"[louvain_capped] sweep cap ({max_sweeps}) hit on a level with "
                f"{G.number_of_nodes()} nodes; accepting the current partition "
                f"(float-weight oscillation, upstream networkx non-termination)."
            )
            break
        n_sweeps += 1
        nb_moves = 0
        for u in rand_nodes:
            best_mod = 0
            best_com = node2com[u]
            weights2com = _neighbor_weights(nbrs[u], node2com)
            if is_directed:
                in_degree = in_degrees[u]
                out_degree = out_degrees[u]
                Stot_in[best_com] -= in_degree
                Stot_out[best_com] -= out_degree
                remove_cost = (
                    -weights2com[best_com] / m
                    + resolution
                    * (out_degree * Stot_in[best_com] + in_degree * Stot_out[best_com])
                    / m**2
                )
            else:
                degree = degrees[u]
                Stot[best_com] -= degree
                remove_cost = -weights2com[best_com] / m + resolution * (
                    Stot[best_com] * degree
                ) / (2 * m**2)
            for nbr_com, wt in weights2com.items():
                if is_directed:
                    gain = (
                        remove_cost
                        + wt / m
                        - resolution
                        * (
                            out_degree * Stot_in[nbr_com]
                            + in_degree * Stot_out[nbr_com]
                        )
                        / m**2
                    )
                else:
                    gain = (
                        remove_cost
                        + wt / m
                        - resolution * (Stot[nbr_com] * degree) / (2 * m**2)
                    )
                if gain > best_mod:
                    best_mod = gain
                    best_com = nbr_com
            if is_directed:
                Stot_in[best_com] += in_degree
                Stot_out[best_com] += out_degree
            else:
                Stot[best_com] += degree
            if best_com != node2com[u]:
                com = G.nodes[u].get("nodes", {u})
                partition[node2com[u]].difference_update(com)
                inner_partition[node2com[u]].remove(u)
                partition[best_com].update(com)
                inner_partition[best_com].add(u)
                improvement = True
                nb_moves += 1
                node2com[u] = best_com
    partition = list(filter(len, partition))
    inner_partition = list(filter(len, inner_partition))
    return partition, inner_partition, improvement


def _neighbor_weights(nbrs, node2com):
    weights = defaultdict(float)
    for nbr, wt in nbrs.items():
        weights[node2com[nbr]] += wt
    return weights


def _gen_graph(G, partition):
    """Generate a new graph based on the partitions of a given graph"""
    H = G.__class__()
    node2com = {}
    for i, part in enumerate(partition):
        nodes = set()
        for node in part:
            node2com[node] = i
            nodes.update(G.nodes[node].get("nodes", {node}))
        H.add_node(i, nodes=nodes)

    for node1, node2, wt in G.edges(data=True):
        wt = wt["weight"]
        com1 = node2com[node1]
        com2 = node2com[node2]
        temp = H.get_edge_data(com1, com2, {"weight": 0})["weight"]
        H.add_edge(com1, com2, weight=wt + temp)
    return H


def _convert_multigraph(G, weight, is_directed):
    """Convert a Multigraph to normal Graph"""
    if is_directed:
        H = nx.DiGraph()
    else:
        H = nx.Graph()
    H.add_nodes_from(G)
    for u, v, wt in G.edges(data=weight, default=1):
        if H.has_edge(u, v):
            H[u][v]["weight"] += wt
        else:
            H.add_edge(u, v, weight=wt)
    return H
