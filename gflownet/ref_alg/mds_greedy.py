import torch
import dgl
from ..util import get_decided

class MDSGreedy:
    def __init__(self) -> None:
        self.name = "MDSGreedy"

    @torch.no_grad()
    def sample(self, gbatch_rep, state, done, rand_prob=0., reward_exp=1.0):
        device = state.device  # Ensure tensors are on the same device

        # Number of nodes per graph and cumulative offsets
        batch_num_nodes = gbatch_rep.batch_num_nodes().tolist()
        cumulative_nodes = [0] + torch.cumsum(torch.tensor(batch_num_nodes, device=device), dim=0).tolist()

        # Initialize actions and logits
        actions = torch.full((len(batch_num_nodes),), -1, dtype=torch.long, device=device)
        combined_output = torch.full((gbatch_rep.num_nodes(),), -1, device=device)

        # Extract decided mask from state
        decided_mask = get_decided(state)

        # Create a mask for uncovered nodes
        uncovered_mask = ~decided_mask

        # Compute coverage for all nodes in the batch
        gbatch_rep.ndata['uncovered'] = uncovered_mask.float()
        gbatch_rep.update_all(
            dgl.function.copy_u('uncovered', 'msg'),
            dgl.function.sum('msg', 'coverage')
        )

        # Extract coverage for all nodes
        coverage = gbatch_rep.ndata['coverage']
        coverage[~uncovered_mask] = -1  # Exclude already decided nodes

        # Select the best node for each graph
        for i, num_nodes in enumerate(batch_num_nodes):
            
            # Get the range of nodes belonging to the i-th graph
            start_idx = cumulative_nodes[i]
            end_idx = cumulative_nodes[i + 1]

            # Compute best node in the subgraph
            graph_coverage = coverage[start_idx:end_idx]

            # Ensure the graph coverage is masked appropriately for this graph's nodes
            graph_decided_mask = decided_mask[start_idx:end_idx]
            graph_coverage[graph_decided_mask] = -1

            best_node_local = torch.argmax(graph_coverage).item()

            # Map local node index to global index
            best_node_global = start_idx + best_node_local

            # Record the selected node and update logits
            combined_output[best_node_global] = 1

            if not done[i]:
                actions[i] = best_node_local

        return actions, combined_output
