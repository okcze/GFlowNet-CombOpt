import torch
import dgl
import copy

from .ref_alg.mis_greedy import MISGreedy, MISGreedyOriginal

# Assume original MISGreedy code is defined as MISGreedyOriginal
# and optimized version as MISGreedyOptimized

def generate_sample_data(num_graphs, num_nodes):
    graphs = []
    states = []
    done = []

    for _ in range(num_graphs):
        g = dgl.rand_graph(num_nodes, num_nodes * 2)
        state = torch.randint(0, 2, (num_nodes,), dtype=torch.float32)
        is_done = torch.randint(0, 2, (1,), dtype=torch.bool).item()
        graphs.append(g)
        states.append(state)
        done.append(is_done)

    gbatch_rep = dgl.batch(graphs)
    state_tensor = torch.cat(states)
    done_tensor = torch.tensor(done, dtype=torch.bool)

    return gbatch_rep, state_tensor, done_tensor

def compare_versions(gbatch_rep, state_tensor, done_tensor):
    # Instantiate both versions of the algorithm
    original_algo = MISGreedyOriginal()
    optimized_algo = MISGreedy()

    # Run both versions
    original_actions, original_output = original_algo.sample(copy.deepcopy(gbatch_rep), state_tensor.clone(), done_tensor.clone())
    optimized_actions, optimized_output = optimized_algo.sample(copy.deepcopy(gbatch_rep), state_tensor.clone(), done_tensor.clone())

    # Compare results
    actions_match = torch.equal(original_actions, optimized_actions)
    output_match = torch.allclose(original_output, optimized_output, atol=1e-6)

    if actions_match and output_match:
        print("Both versions produce identical results.")
    else:
        print("There are differences between the versions.")
        if not actions_match:
            print("Actions mismatch")
            print("Original Actions:", original_actions)
            print("Optimized Actions:", optimized_actions)
        if not output_match:
            print("Combined Output mismatch")
            print("Original Output:", original_output)
            print("Optimized Output:", optimized_output)

# # Generate sample data
# gbatch_rep, state_tensor, done_tensor = generate_sample_data(num_graphs=5, num_nodes=10)

# # Compare the original and optimized versions
# compare_versions(gbatch_rep, state_tensor, done_tensor)
