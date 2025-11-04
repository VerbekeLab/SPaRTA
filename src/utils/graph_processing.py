import networkx as nx

def graph_community(G, resolution=10): # large resolution to have smaller communities
    directed = nx.is_directed(G)
    
    if directed:
        G_undirected = G.copy().to_undirected()
    else:
        G_undirected = G.copy()

    community_list = nx.community.louvain_communities(G_undirected, resolution=resolution, seed=1997)

    # Create a dictionary to map nodes to their community
    node_community = {}
    for idx, community in enumerate(community_list):
        for node in community:
            node_community[node] = idx

    # Create a new graph with only intra-community edges
    if directed:
        H = nx.DiGraph()
    else:
        H = nx.Graph()
        
    H.add_nodes_from(G.nodes(data=True))  # Add all nodes with their attributes

    # Add only edges that connect nodes within the same community
    for u, v in G.edges():
        if node_community[u] == node_community[v]:
            H.add_edge(u, v, **G[u][v])
        
    return(H)