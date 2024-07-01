import networkx as nx
import dgl
import torch
import random

from ..util import get_decided

class MISLocalImprovement:

    def __init__(self) -> None:
        self.name = "MISLocalImprovement"

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

            # Mask already decided nodes
            decided_mask = get_decided(graph_state)
            independent_set = {n for n in nx_g.nodes if decided_mask[n]}

            improved = True
            while improved:
                improved = False
                # Try to add one independent node
                for node in nx_g.nodes:
                    if node not in independent_set and all(neighbor not in independent_set for neighbor in nx_g.neighbors(node)):
                        independent_set.add(node)
                        improved = True
                        break

                # Try to delete one node and add two other independent nodes
                if not improved:
                    for node in list(independent_set):
                        neighbors = set(nx_g.neighbors(node))
                        potential_adds = [n for n in nx_g.nodes if n not in independent_set and all(neighbor not in independent_set for neighbor in nx_g.neighbors(n))]
                        if len(potential_adds) >= 2:
                            for add1 in potential_adds:
                                for add2 in potential_adds:
                                    if add1 != add2 and nx_g.has_edge(add1, add2) == False:
                                        independent_set.remove(node)
                                        independent_set.add(add1)
                                        independent_set.add(add2)
                                        improved = True
                                        break
                                if improved:
                                    break
                        if improved:
                            break
            
            # Select a random action from the improved independent set
            if independent_set:
                actions[i] = random.choice(list(independent_set))
        
        return actions
