import networkx as nx
import dgl
import torch

from ..util import get_decided

class MISGreedy:

    def __init__(self) -> None:
        self.name = "MISGreedy"

    @torch.no_grad()
    def sample(self, gbatch_rep, state, done, rand_prob=0.):
        device = state.device  # Ensure tensors are on the same device

        # Initialize actions with -1 to denote impossible actions for done graphs
        actions = torch.full((gbatch_rep.batch_size,), -1, dtype=torch.long, device=device)

        # Split state into individual graph states
        batch_num_nodes = gbatch_rep.batch_num_nodes().tolist()
        graphs_states = torch.split(state, batch_num_nodes, dim=0)

        for i, (graph_state, num_nodes) in enumerate(zip(graphs_states, batch_num_nodes)):
            # Check if the graph is already done
            if done[i]:
                continue

            # Ensure the subgraph is on the same device
            subgraph_nodes = torch.arange(num_nodes).to(device)
            subgraph = gbatch_rep.subgraph(subgraph_nodes)
            subgraph = subgraph.cpu()  # Move subgraph to CPU for NetworkX conversion
            nx_g = dgl.to_networkx(subgraph)
            if nx_g.number_of_nodes() == 0:
                continue

            # Mask already decided nodes
            decided_mask = get_decided(graph_state)
            nodes_to_consider = [n for n in nx_g.nodes if not decided_mask[n]]
            if not nodes_to_consider:
                continue

            min_degree_node = min(nodes_to_consider, key=nx_g.degree)
            actions[i] = min_degree_node
        
        return actions

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
