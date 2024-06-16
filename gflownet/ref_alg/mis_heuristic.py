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

    def algorithm(self, nx_graph):
        V = list(nx_graph.nodes())
        E = list(nx_graph.edges())
        
        V0 = set(V)
        S = set()
        Est = 0

        in_set = [2] * len(V)  # Initialize with 2 meaning no decision yet
        node_index_map = {node: index for index, node in enumerate(V)}
        
        # List to store snapshots of the state of the independent set S
        snapshots = []

        while V0:
            k = {}
            m = {}
            
            # Compute k[v] and m[v] for each vertex in V0
            for v in V0:
                neighbors = set(nx_graph.neighbors(v))
                k[v] = len(neighbors)
                complete_subgraph_edges = k[v] * (k[v] - 1) // 2
                actual_edges = sum(1 for u in neighbors if any(w in neighbors for w in nx_graph.neighbors(u)))
                m[v] = complete_subgraph_edges - actual_edges
            
            # Select the vertex with the minimum m value and maximum k value
            v0 = self.min_max_param(V0, k, m)
            
            # Add the selected vertex to the independent set
            S.add(v0)
            in_set[node_index_map[v0]] = 1  # Mark as in the set
            Est += m[v0]
            
            # Mark the neighbors of the selected vertex as not in the set
            neighbors = set(nx_graph.neighbors(v0))
            for neighbor in neighbors:
                if neighbor in V0:  # Only mark those still in V0
                    in_set[node_index_map[neighbor]] = 0

            # Take a snapshot of the current state of in_set
            snapshots.append(torch.tensor(in_set, dtype=torch.int32))
            
            # Remove the selected vertex and its neighbors from V0
            V0.remove(v0)
            V0 -= neighbors

        # Final snapshot after the last modification
        snapshots.append(torch.tensor(in_set, dtype=torch.long))

        return snapshots