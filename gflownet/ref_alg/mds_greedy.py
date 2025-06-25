import torch
import dgl
from ..util import get_decided

class MDSGreedy:
    def __init__(self) -> None:
        self.name = "MDSGreedy"

    @torch.no_grad()
    def sample(self, gbatch_rep, state, done, rand_prob=0., reward_exp=1.0):
        device = state.device

        batch_num_nodes = gbatch_rep.batch_num_nodes().tolist()
        cumulative_nodes = [0] + torch.cumsum(torch.tensor(batch_num_nodes, device=device), dim=0).tolist()

        actions = torch.full((len(batch_num_nodes),), -1, dtype=torch.long, device=device)
        combined_output = torch.full((gbatch_rep.num_nodes(),), -1, device=device)

        decided_mask = get_decided(state)

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

        for i, num_nodes in enumerate(batch_num_nodes):
            
            # Get the range of nodes belonging to the i-th graph
            start_idx = cumulative_nodes[i]
            end_idx = cumulative_nodes[i + 1]

            # Compute best node in the subgraph
            graph_coverage = coverage[start_idx:end_idx]

            # Ensure the graph coverage is masked appropriately for this graph's nodes
            graph_decided_mask = decided_mask[start_idx:end_idx]
            graph_coverage[graph_decided_mask] = -1

            max_value = graph_coverage.max().item()
            if max_value == -1:
                # No valid candidate found for this graph; leave action as -1.
                continue

            best_node_local = torch.argmax(graph_coverage).item()

            best_node_global = start_idx + best_node_local

            combined_output[best_node_global] = 1

            if not done[i]:
                actions[i] = best_node_local

        return actions, combined_output
