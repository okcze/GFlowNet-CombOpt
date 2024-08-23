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
        combined_output = torch.full((combined_output_size,), 10**-6, device=device)

        # Calculate cumulative sums of the number of nodes to use as offsets
        cumulative_nodes = [0] + torch.cumsum(torch.tensor(batch_num_nodes), dim=0).tolist()

        for i, (graph_state, num_nodes) in enumerate(zip(graphs_states, batch_num_nodes)):
            # Ensure the subgraph is on the same device
            subgraph_nodes = torch.arange(num_nodes).to(device)
            subgraph = gbatch_rep.subgraph(subgraph_nodes)
            
            # Mask already decided nodes
            decided_mask = get_decided(graph_state)
            
            # Get degrees directly from DGL without converting to NetworkX
            degrees = subgraph.in_degrees() + subgraph.out_degrees()
            
            # Set the degrees of decided nodes to a large value (to exclude them)
            degrees[decided_mask] = torch.iinfo(degrees.dtype).max

            # Get the node with the minimum degree that isn't decided
            min_degree_node = torch.argmin(degrees).item()
            
            # Update the combined output tensor with logits
            offset = cumulative_nodes[i]
            combined_output[offset + min_degree_node] = 1

            # Check if the graph is already done
            if not done[i]:
                actions[i] = min_degree_node
        
        return actions, combined_output
