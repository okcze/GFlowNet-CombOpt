import torch
import dgl
import dgl.function as fn
from ..util import get_decided

class MaxCutGreedy:
    def __init__(self) -> None:
        self.name = "MaxCutGreedy"

    @torch.no_grad()
    def sample(self, gbatch_rep, state, done, rand_prob=0., reward_exp=1.0):
        """
        Greedy sampler for Max-Cut. For each graph in the batch, among the vertices still undecided (state == 2),
        this sampler selects the vertex that would yield the maximum increase in the cut
        value if added to S. 
        """
        device = state.device

        # Compute the number of nodes per graph and the cumulative offsets.
        batch_num_nodes = gbatch_rep.batch_num_nodes().tolist()  # list of ints, one per graph
        cumulative_nodes = [0] + torch.cumsum(torch.tensor(batch_num_nodes, device=device), dim=0).tolist()

        # Initialize the outputs.
        num_graphs = len(batch_num_nodes)
        actions = torch.full((num_graphs,), -1, dtype=torch.long, device=device)
        combined_output = torch.full((gbatch_rep.num_nodes(),), -1, device=device)

        # In MaxCutMDP, "decided" means the node is not in the pool (i.e. state != 2).
        decided_mask = get_decided(state)
        undecided_mask = ~decided_mask  # True for nodes with state == 2

        # Compute, for every node, the number of neighbors that are not in S.
        # Since evaluation treats both state 0 and state 2 as "not in S", we define:
        #   notS = 1 if state == 0 or state == 2, and 0 if state == 1.
        with gbatch_rep.local_scope():
            notS = (((state == 0) | (state == 2)).float())
            gbatch_rep.ndata['notS'] = notS
            gbatch_rep.update_all(fn.copy_u('notS', 'm'), fn.sum('m', 'gain'))
            gain = gbatch_rep.ndata['gain']

        # Only consider vertices that are still undecided.
        gain[~undecided_mask] = -float('inf')

        # For each graph in the batch, select the undecided vertex with maximum gain.
        for i, num_nodes in enumerate(batch_num_nodes):
            start_idx = cumulative_nodes[i]
            end_idx = cumulative_nodes[i + 1]
            local_gain = gain[start_idx:end_idx]
            if local_gain.numel() == 0 or done[i]:
                continue  # Nothing to choose if there is no undecided vertex or graph is done.
            best_local = torch.argmax(local_gain).item()
            actions[i] = best_local
            best_global = start_idx + best_local
            combined_output[best_global] = 1

        return actions, combined_output
