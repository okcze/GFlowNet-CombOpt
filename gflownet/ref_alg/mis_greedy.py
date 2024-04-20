import networkx as nx
import dgl
import torch

def convert_nx_to_dgl(nx_graph):
    # Convert a NetworkX graph to a DGL graph, preserving node ordering
    g = dgl.from_networkx(nx_graph)
    return g

def greedy_maximum_independent_set_with_snapshots(nx_graph):
    nodes = list(nx_graph.nodes())
    in_set = [0] * len(nodes)
    node_index_map = {node: index for index, node in enumerate(nodes)}

    working_graph = nx_graph.copy()

    # List to store snapshots of the state of the in_set array
    snapshots = []

    while working_graph.number_of_nodes() > 0:
        # Take a snapshot at the start of the iteration
        snapshots.append(torch.tensor(in_set, dtype=torch.int32))

        min_degree_node = min(working_graph.nodes, key=working_graph.degree)
        in_set[node_index_map[min_degree_node]] = 1
        
        neighbors = list(working_graph.neighbors(min_degree_node))
        working_graph.remove_node(min_degree_node)
        working_graph.remove_nodes_from(neighbors)

    # Final snapshot after the last modification
    snapshots.append(torch.tensor(in_set, dtype=torch.int32))

    return in_set, node_index_map, snapshots

def set_independent_set_to_dgl(dgl_graph, in_set, node_index_map):
    # Initialize node data for 'in_set' in DGL graph
    dgl_graph.ndata['in_set'] = torch.tensor(in_set, dtype=torch.int32)