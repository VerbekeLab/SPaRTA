import networkx as nx

def node_selection(G_ego_second, G_ego_second_und, G_ego_second_rev, node):
    nodes_1 = list(
      nx.ego_graph(G_ego_second_und, node).nodes
      )
    nodes_1.remove(node)
    nodes_2 = list(G_ego_second.nodes)

    nodes_2_s = list(
        nx.ego_graph(G_ego_second, node, radius=2).nodes
        )
    
    nodes_2_rs = list(
        nx.ego_graph(G_ego_second_rev, node, radius=2).nodes
    )
    
    nodes_0 = list(
        set(nodes_2).difference(set(nodes_2_s)).difference(set(nodes_2_rs)).difference(set(nodes_1))
    )

    nodes_0 = [node] + nodes_0

    for n in nodes_0:
        nodes_2.remove(n)
    for n in nodes_1:
        nodes_2.remove(n)

    # For directed network, specific order to obtain scores (in order of "group")
    nodes_ordered = nodes_0 + nodes_1 + nodes_2
        
    return nodes_0, nodes_1, nodes_2, nodes_ordered