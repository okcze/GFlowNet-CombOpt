import networkx as nx
import torch
import dgl
from ..util import get_decided

class MISHeuristic:

    def __init__(self) -> None:
        self.name = "MISHeuristic"

    def min_max_param(self, V0, k, m):
        # Select the vertex with the minimum m value and maximum k value
        V0_sorted = sorted(V0, key=lambda v: (m[v], -k[v]))
        return V0_sorted[0]
    
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

            V = list(nx_g.nodes())
            V0 = set(V)
            k = {}
            m = {}

            # Mask already decided nodes
            decided_mask = get_decided(graph_state)
            nodes_to_consider = [n for n in V0 if not decided_mask[n]]
            if not nodes_to_consider:
                continue

            # Compute k[v] and m[v] for each vertex in nodes_to_consider
            for v in nodes_to_consider:
                neighbors = set(nx_g.neighbors(v))
                k[v] = len(neighbors)
                complete_subgraph_edges = k[v] * (k[v] - 1) // 2
                actual_edges = sum(1 for u in neighbors if any(w in neighbors for w in nx_g.neighbors(u)))
                m[v] = complete_subgraph_edges - actual_edges
            
            # Select the vertex with the minimum m value and maximum k value
            v0 = self.min_max_param(nodes_to_consider, k, m)
            actions[i] = v0
        
        return actions
