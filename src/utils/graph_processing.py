import pandas as pd
import networkx as nx

def graph_degree_rel(G_copy, degree_cutoff=0.01):
    # Delete the hubs
    # The cut-off is defined as a relative number
    
    degree_df = pd.DataFrame(
        dict(
            G_copy.degree()
        ), 
        index = ["Degree"]
    ).transpose()

    degree_threshold = degree_df["Degree"].quantile(1 - degree_cutoff)
    hub_criteria = degree_df["Degree"] >= degree_threshold
    
    hubs_deleted = list(
                degree_df[hub_criteria].reset_index()["index"]
            )

    G_copy.remove_nodes_from(
        hubs_deleted
    )
    print(f"Removed {len(hubs_deleted)} hubs with degree >= {degree_threshold}.")   
    
    return(G_copy)

def graph_degree_abs(G_copy, degree_cutoff=10): 
    # Delete the hubs
    # The cut-off is defined as an absolute number
    
    degree_df = pd.DataFrame(
        dict(
            G_copy.degree()
        ), 
        index = ["Degree"]
    ).transpose()

    hub_criteria = degree_df["Degree"] >= degree_cutoff
    
    hubs_deleted = list(
                degree_df[hub_criteria].reset_index()["index"]
            )

    G_copy.remove_nodes_from(
        hubs_deleted
    )
    print(f"Removed {len(hubs_deleted)} hubs with degree >= {degree_cutoff}.")   
    
    return(G_copy)

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