import networkx as nx
import dgl
import torch

class MISGreedy:

    def __init__(self) -> None:
        self.name = "MISGreedy"

    @torch.no_grad()
    def sample(self, gbatch_rep, state, done, rand_prob=0.):
        actions = []
        batch_num_graphs = gbatch_rep.batch_size
        node_offset = 0

        for i in range(batch_num_graphs):
            subgraph = gbatch_rep.node_subgraph(gbatch_rep.batch_num_nodes == i)
            nx_g = dgl.to_networkx(subgraph)
            if nx_g.number_of_nodes() == 0:
                continue
            min_degree_node = min(nx_g.nodes, key=nx_g.degree)
            actions.append(min_degree_node + node_offset)
            node_offset += subgraph.number_of_nodes()
        
        return torch.tensor(actions, dtype=torch.int64)

    def algorithm(self, nx_graph):
        nodes = list(nx_graph.nodes())
        in_set = [2] * len(nodes)
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
            for neighbor in neighbors:
                neighbor_index = node_index_map[neighbor]
                in_set[neighbor_index] = 0
            working_graph.remove_node(min_degree_node)
            working_graph.remove_nodes_from(neighbors)

        # Final snapshot after the last modification
        snapshots.append(torch.tensor(in_set, dtype=torch.long))

        return snapshots
