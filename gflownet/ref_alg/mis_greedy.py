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

        # Initialize the combined output tensor
        combined_output_size = sum(batch_num_nodes)
        combined_output = torch.zeros(combined_output_size, device=device)

        # Calculate cumulative sums of the number of nodes to use as offsets
        cumulative_nodes = [0] + torch.cumsum(torch.tensor(batch_num_nodes), dim=0).tolist()

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

            # Update the combined output tensor with logits
            offset = cumulative_nodes[i]
            combined_output[offset + min_degree_node] = 1
        
        return actions, combined_output
