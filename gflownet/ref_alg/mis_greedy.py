import networkx as nx
import dgl
import torch

class MISGreedy:

    self.name = "MISGreedy"

    def algorithm(nx_graph):
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

        return snapshots
