import torch

from ..util import get_decided

class MDSGreedy:
    def __init__(self) -> None:
        self.name = "MDSGreedy"

    @torch.no_grad()
    def sample(self, gbatch_rep, state, done, rand_prob=0., reward_exp=1.0):
        device = state.device

        # Initialize actions with -1 for graphs that are already done
        actions = torch.full((gbatch_rep.batch_size,), -1, dtype=torch.long, device=device)

        # Split state into individual graph states
        batch_num_nodes = gbatch_rep.batch_num_nodes().tolist()
        graphs_states = torch.split(state, batch_num_nodes, dim=0)

        # Prepare output tensor for logits
        combined_output_size = sum(batch_num_nodes)
        combined_output = torch.full((combined_output_size,), 10**-6, device=device)
        
        # Calculate cumulative sums of node counts for indexing
        cumulative_nodes = [0] + torch.cumsum(torch.tensor(batch_num_nodes), dim=0).tolist()

        for i, (graph_state, num_nodes) in enumerate(zip(graphs_states, batch_num_nodes)):
            if done[i]:
                continue

            # Create subgraph for this batch element
            subgraph_nodes = torch.arange(num_nodes, device=device)
            subgraph = gbatch_rep.subgraph(subgraph_nodes)

            decided_mask = get_decided(graph_state)
            uncovered_nodes = subgraph_nodes[~decided_mask]

            if len(uncovered_nodes) == 0:
                continue

            # Greedy choice: pick the node that covers the most uncovered nodes.
            best_node = -1
            best_coverage = -1

            for node in uncovered_nodes:
                neighbors = subgraph.successors(node)
                candidates = torch.cat([neighbors, node.unsqueeze(0)])
                coverage = torch.sum(~decided_mask[candidates])

                if coverage > best_coverage:
                    best_coverage = coverage
                    best_node = node.item()

            # Record the chosen node’s logit and action
            offset = cumulative_nodes[i]
            combined_output[offset + best_node] = 1
            actions[i] = best_node

        return actions, combined_output
