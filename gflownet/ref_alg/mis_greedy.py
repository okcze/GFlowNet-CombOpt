import torch
from ..util import get_decided

class MISGreedy:
    def __init__(self) -> None:
        self.name = "MISGreedy"

    @torch.no_grad()
    def sample(self, gbatch_rep, state, done, rand_prob=0., reward_exp=1.0):
        device = state.device  # Ensure tensors are on the same device

        # Initialize actions with -1 to denote impossible actions for done graphs
        actions = torch.full((gbatch_rep.batch_size,), -1, dtype=torch.long, device=device)

        # Get number of nodes per graph and compute cumulative offsets
        batch_num_nodes = gbatch_rep.batch_num_nodes().tolist()
        cumulative_nodes = [0] + torch.cumsum(torch.tensor(batch_num_nodes), dim=0).tolist()

        # Split state into individual graph states for local processing
        graphs_states = torch.split(state, batch_num_nodes, dim=0)

        # Initialize the combined output tensor
        combined_output_size = sum(batch_num_nodes)
        combined_output = torch.full((combined_output_size,), 10**-6, device=device)

        for i, (graph_state, num_nodes) in enumerate(zip(graphs_states, batch_num_nodes)):
            # Compute the start and end indices for the i-th graph in the batched graph
            start_idx = cumulative_nodes[i]
            end_idx = cumulative_nodes[i + 1]

            # Correctly extract the subgraph nodes using global indices
            subgraph_nodes = torch.arange(start_idx, end_idx, device=device)
            subgraph = gbatch_rep.subgraph(subgraph_nodes)
            
            # Mask already decided nodes in the local state
            decided_mask = get_decided(graph_state)
            
            # Get degrees directly from DGL (in-degree + out-degree)
            degrees = subgraph.in_degrees() + subgraph.out_degrees()
            
            # Exclude decided nodes by setting their degree to a very large value
            degrees[decided_mask] = torch.iinfo(degrees.dtype).max

            # Get the node with the minimum degree among undecided nodes
            min_degree_node = torch.argmin(degrees).item()
            
            # Update the combined output tensor at the correct global index
            combined_output[start_idx + min_degree_node] = 1

            # Record the selected local node index for this graph if it is not done
            if not done[i]:
                actions[i] = min_degree_node
        
        return actions, combined_output
