import networkx as nx
import torch

class MISHeuristic:

    self.name = "MISHeuristic"

    def min_max_param(V0, k, m):
        # Select the vertex with the minimum m value and maximum k value
        V0_sorted = sorted(V0, key=lambda v: (m[v], -k[v]))
        return V0_sorted[0]

    def algorithm(nx_graph):
        V = list(nx_graph.nodes())
        E = list(nx_graph.edges())
        
        V0 = set(V)
        S = set()
        Est = 0
        
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
            v0 = min_max_param(V0, k, m)
            
            # Add the selected vertex to the independent set
            S.add(v0)
            Est += m[v0]
            
            # Take a snapshot of the current independent set S
            snapshot = [1 if v in S else 0 for v in V]
            snapshots.append(torch.tensor(snapshot, dtype=torch.int32))
            
            # Remove the selected vertex and its neighbors from V0
            neighbors = set(nx_graph.neighbors(v0))
            V0.remove(v0)
            V0 -= neighbors

        return snapshots