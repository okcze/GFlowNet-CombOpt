import sys, os
import gzip, pickle
from time import time, sleep
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, open_dict, OmegaConf
import matplotlib.pyplot as plt

import random
import numpy as np
import torch
import dgl
from einops import rearrange, reduce, repeat

from .data import get_data_loaders
from .util import get_reference_alg, seed_torch, TransitionBuffer, get_mdp_class
from .algorithm import DetailedBalanceTransitionBuffer, sample_from_logits

torch.backends.cudnn.benchmark = True


def get_alg_buffer(cfg, device):
    assert cfg.alg in ["db", "fl"]
    buffer = TransitionBuffer(cfg.tranbuff_size, cfg)
    alg = DetailedBalanceTransitionBuffer(cfg, device)
    return alg, buffer

def get_saved_alg_buffer(cfg, device, alg_load_path):
    """Allow to load model from file."""
    assert cfg.alg in ["db", "fl"]
    buffer = TransitionBuffer(cfg.tranbuff_size, cfg)
    alg = DetailedBalanceTransitionBuffer(cfg, device)
    alg.load(alg_load_path)
    return alg, buffer

def get_logr_scaler(cfg, process_ratio=1., reward_exp=None):
    if reward_exp is None:
        reward_exp = float(cfg.reward_exp)

    if cfg.anneal == "linear":
        process_ratio = max(0., min(1., process_ratio)) # from 0 to 1
        reward_exp = reward_exp * process_ratio +\
                     float(cfg.reward_exp_init) * (1 - process_ratio)
    elif cfg.anneal == "none":
        pass
    else:
        raise NotImplementedError

    # (R/T)^beta -> (log R - log T) * beta
    def logr_scaler(sol_size, gbatch=None):
        logr = sol_size
        return logr * reward_exp
    return logr_scaler

def refine_cfg(cfg):
    with open_dict(cfg):
        cfg.device = cfg.d
        cfg.work_directory = os.getcwd()

        if cfg.task in ["mis", "maxindset", "maxindependentset",]:
            cfg.task = "MaxIndependentSet"
            cfg.wandb_project_name = "MIS"
        elif cfg.task in ["mds", "mindomset", "mindominateset",]:
            cfg.task = "MinDominateSet"
            cfg.wandb_project_name = "MDS"
        elif cfg.task in ["mc", "maxclique",]:
            cfg.task = "MaxClique"
            cfg.wandb_project_name = "MaxClique"
        elif cfg.task in ["mcut", "maxcut",]:
            cfg.task = "MaxCut"
            cfg.wandb_project_name = "MaxCut"
        else:
            raise NotImplementedError

        # architecture
        assert cfg.arch in ["gin"]

        # log reward shape
        cfg.reward_exp = cfg.rexp
        cfg.reward_exp_init = cfg.rexpit
        if cfg.anneal in ["lin"]:
            cfg.anneal = "linear"

        # training
        cfg.batch_size = cfg.bs
        cfg.batch_size_interact = cfg.bsit
        cfg.leaf_coef = cfg.lc
        cfg.same_graph_across_batch = cfg.sameg

        # data
        cfg.test_batch_size = cfg.tbs
        if "rb" in cfg.input:
            cfg.data_type = cfg.input.upper()
        elif "ba" in cfg.input:
            cfg.data_type = cfg.input.upper()
        else:
            raise NotImplementedError

    del cfg.d, cfg.rexp, cfg.rexpit, cfg.bs, cfg.bsit, cfg.lc, cfg.sameg, cfg.tbs
    return cfg

@torch.no_grad()
def rollout(gbatch, cfg, alg, ref_alg=None):
    env = get_mdp_class(cfg.task)(gbatch, cfg)
    state = env.state

    ##### sample traj
    reward_exp_eval = None
    traj_s, traj_r, traj_a, traj_d = [], [], [], []
    while not all(env.done):
        if ref_alg:
            action, _ = ref_alg.sample(gbatch, state, env.done, rand_prob=cfg.randp)
        else:
            action = alg.sample(gbatch, state, env.done, rand_prob=cfg.randp, reward_exp=reward_exp_eval)

        traj_s.append(state)
        traj_r.append(env.get_log_reward())
        traj_a.append(action)
        traj_d.append(env.done)
        state = env.step(action)

    ##### save last state
    traj_s.append(state)
    traj_r.append(env.get_log_reward())
    traj_d.append(env.done)
    assert len(traj_s) == len(traj_a) + 1 == len(traj_r) == len(traj_d)

    traj_s = torch.stack(traj_s, dim=1) # (sum of #node per graph in batch, max_traj_len)
    traj_r = torch.stack(traj_r, dim=1) # (batch_size, max_traj_len)
    traj_a = torch.stack(traj_a, dim=1) # (batch_size, max_traj_len-1)
    """
    traj_a is tensor like 
    [ 4, 30, 86, 95, 96, 29, -1, -1],
    [47, 60, 41, 11, 55, 64, 80, -1],
    [26, 38, 13,  5,  9, -1, -1, -1]
    """
    traj_d = torch.stack(traj_d, dim=1) # (batch_size, max_traj_len)
    """
    traj_d is tensor like 
    [False, False, False, False, False, False,  True,  True,  True],
    [False, False, False, False, False, False, False,  True,  True],
    [False, False, False, False, False,  True,  True,  True,  True]
    """
    traj_len = 1 + torch.sum(~traj_d, dim=1) # (batch_size, )

    ##### graph, state, action, done, reward, trajectory length
    batch = gbatch.cpu(), traj_s.cpu(), traj_a.cpu(), traj_d.cpu(), traj_r.cpu(), traj_len.cpu()
    return batch, env.batch_metric(state)

@torch.no_grad()
def combine_batches_random(batch_normal, batch_ref, metric_ls_normal, metric_ls_ref, cfg):
    """
    Combines two batches by randomly replacing a fraction of elements
    from batch_normal with elements from batch_ref.
    """
    gbatch_normal, traj_s_normal, traj_a_normal, traj_d_normal, traj_r_normal, traj_len_normal = batch_normal
    gbatch_ref, traj_s_ref, traj_a_ref, traj_d_ref, traj_r_ref, traj_len_ref = batch_ref

    traj_r_ref = traj_r_ref * cfg.reward_boost

    gbatch_combined = gbatch_normal
    batch_size = traj_s_normal.size(0)
    num_replace = int(cfg.frac_replaced * batch_size)
    replace_indices = torch.randperm(batch_size)[:num_replace]
    mask = torch.zeros(batch_size, dtype=torch.bool)
    mask[replace_indices] = True

    traj_s_combined = traj_s_normal.clone()
    traj_s_combined[mask] = traj_s_ref[mask]

    traj_a_combined = traj_a_normal.clone()
    traj_a_combined[mask] = traj_a_ref[mask]

    traj_d_combined = traj_d_normal.clone()
    traj_d_combined[mask] = traj_d_ref[mask]

    traj_r_combined = traj_r_normal.clone()
    traj_r_combined[mask] = traj_r_ref[mask]

    traj_len_combined = traj_len_normal.clone()
    traj_len_combined[mask] = traj_len_ref[mask]

    metric_ls_combined = []
    for i in range(batch_size):
        if mask[i]:
            metric_ls_combined.append(metric_ls_ref[i])
        else:
            metric_ls_combined.append(metric_ls_normal[i])

    # Create the combined batch
    batch_combined = (
        gbatch_combined,
        traj_s_combined,
        traj_a_combined,
        traj_d_combined,
        traj_r_combined,
        traj_len_combined,
    )

    return batch_combined, metric_ls_combined

@hydra.main(config_path="configs", config_name="main") # for hydra-core==1.1.0
# @hydra.main(version_base=None, config_path="configs", config_name="main") # for newer hydra
def main(cfg: DictConfig):
    cfg = refine_cfg(cfg)
    device = torch.device(f"cuda:{cfg.device:d}" if torch.cuda.is_available() and cfg.device>=0 else "cpu")
    print(f"Device: {device}")
    alg, buffer = get_alg_buffer(cfg, device)

    ### Additional buffer for reference algorithm
    _, ref_buffer = get_alg_buffer(cfg, device)

    ref_alg = get_reference_alg(cfg)
    seed_torch(cfg.seed)
    print(str(cfg))
    print(f"Work directory: {os.getcwd()}")

    ### Create directory for saving plots
    if cfg.plot_loss and not os.path.exists(f"/content/plots/{cfg.run_name}"):
        os.makedirs(f"/content/plots/{cfg.run_name}")
    elif cfg.plot_loss and os.path.exists(f"/content/{cfg.run_name}"):
        import shutil
        shutil.rmtree(f"/content/plots/{cfg.run_name}")

    train_loader, test_loader = get_data_loaders(cfg)
    trainset_size = len(train_loader.dataset)
    print(f"Trainset size: {trainset_size}")
    alg_save_path = os.path.abspath("./alg.pt")
    alg_save_path_best = os.path.abspath("./alg_best.pt")
    train_data_used = 0
    train_step = 0
    train_logr_scaled_ls = []
    train_metric_ls = []
    metric_best = 0.
    result = {"set_size": {}, "logr_scaled": {}, "train_data_used": {}, "train_step": {}, }

    @torch.no_grad()
    def evaluate(ep, train_step, train_data_used, logr_scaler):
        torch.cuda.empty_cache()
        num_repeat = 20
        mis_ls, mis_top20_ls = [], []
        logr_ls = []
        pbar = tqdm(enumerate(test_loader))
        pbar.set_description(f"Test Epoch {ep:2d} Data used {train_data_used:5d}")
        for batch_idx, gbatch in pbar:
            gbatch = gbatch.to(device)
            gbatch_rep = dgl.batch([gbatch] * num_repeat)

            env = get_mdp_class(cfg.task)(gbatch_rep, cfg)
            state = env.state
            while not all(env.done):
                action = alg.sample(gbatch_rep, state, env.done, rand_prob=0.)
                state = env.step(action)

            logr_rep = logr_scaler(env.get_log_reward())
            logr_ls += logr_rep.tolist()
            curr_mis_rep = torch.tensor(env.batch_metric(state))
            curr_mis_rep = rearrange(curr_mis_rep, "(rep b) -> b rep", rep=num_repeat).float()
            mis_ls += curr_mis_rep.mean(dim=1).tolist()
            mis_top20_ls += curr_mis_rep.max(dim=1)[0].tolist()
            pbar.set_postfix({"Metric": f"{np.mean(mis_ls):.2f}+-{np.std(mis_ls):.2f}"})

        print(f"Test Epoch{ep:2d} Data used{train_data_used:5d}: "
              f"Metric={np.mean(mis_ls):.2f}+-{np.std(mis_ls):.2f}, "
              f"top20={np.mean(mis_top20_ls):.2f}, "
              f"LogR scaled={np.mean(logr_ls):.2e}+-{np.std(logr_ls):.2e}")

        result["set_size"][ep] = np.mean(mis_ls)
        result["logr_scaled"][ep] = np.mean(logr_ls)
        result["train_step"][ep] = train_step
        result["train_data_used"][ep] = train_data_used
        pickle.dump(result, gzip.open("./result.json", 'wb'))

    @torch.no_grad()
    def evaluate_regularization(ep, train_step, train_data_used, logr_scaler, ref_alg):
        torch.cuda.empty_cache()
        num_repeat = 1
        mis_ls, mis_top20_ls = [], []
        logr_ls = []
        pbar = tqdm(enumerate(test_loader))
        pbar.set_description(f"Test Epoch {ep:2d} Data used {train_data_used:5d}")
        
        # Custom regularized evaluation
        intersections = []

        for batch_idx, gbatch in pbar:
            gbatch = gbatch.to(device)
            gbatch_rep = dgl.batch([gbatch] * num_repeat)

            env = get_mdp_class(cfg.task)(gbatch_rep, cfg)
            state = env.state

            env_pref = get_mdp_class(cfg.task)(gbatch_rep, cfg)
            state_pref = env_pref.state

            while not all(env.done):
                action = alg.sample(gbatch_rep, state, env.done, rand_prob=0.)
                state = env.step(action)

            while not all(env_pref.done): 
                actions_pref, _ = ref_alg.sample(gbatch, state_pref, env_pref.done, rand_prob=0.)
                state_pref = env_pref.step(actions_pref)

            # Summarize results
            def state_per_graph(state):
                state_per_graph = torch.split(state, env.numnode_per_graph, dim=0)
                state_per_graph = [state.cpu().numpy() for state in state_per_graph]
                return state_per_graph
            
            states = state_per_graph(state)
            states_pref = state_per_graph(state_pref)

            results_intersection = [set(np.where(states[i]==1)[0].tolist()).intersection(set(np.where(states_pref[i]==1)[0].tolist())) for i in range(len(states))]
            intersections += results_intersection

            logr_rep = logr_scaler(env.get_log_reward())
            logr_ls += logr_rep.tolist()
            curr_mis_rep = torch.tensor(env.batch_metric(state))
            curr_mis_rep = rearrange(curr_mis_rep, "(rep b) -> b rep", rep=num_repeat).float()
            mis_ls += curr_mis_rep.mean(dim=1).tolist()
            mis_top20_ls += curr_mis_rep.max(dim=1)[0].tolist()
            pbar.set_postfix({"Metric": f"{np.mean(mis_ls):.2f}+-{np.std(mis_ls):.2f}"})

        print(f"Test Epoch{ep:2d} Data used{train_data_used:5d}: "
              f"Metric={np.mean(mis_ls):.2f}+-{np.std(mis_ls):.2f}, "
              f"top20={np.mean(mis_top20_ls):.2f}, "
              f"LogR scaled={np.mean(logr_ls):.2e}+-{np.std(logr_ls):.2e}")
        
        avg_intersection = np.mean([len(i) for i in intersections])
        max_intersection = np.max([len(i) for i in intersections])
        print(f"Average intersection: {avg_intersection:.2f}")
        print(f"Max intersection: {max_intersection:.2f}")

        return avg_intersection, max_intersection
        
    # Store loss to plot
    reg_ratio = []
    avg_intersections = []
    max_intersections = []

    for ep in range(cfg.epochs):
        for batch_idx, gbatch in enumerate(train_loader):
            reward_exp = None
            process_ratio = max(0., min(1., train_data_used / cfg.annend))
            logr_scaler = get_logr_scaler(cfg, process_ratio=process_ratio, reward_exp=reward_exp)

            train_logr_scaled_ls = train_logr_scaled_ls[-5000:]
            train_metric_ls = train_metric_ls[-5000:]
            gbatch = gbatch.to(device)
            if cfg.same_graph_across_batch:
                gbatch = dgl.batch([gbatch] * cfg.batch_size_interact)
            train_data_used += gbatch.batch_size

            ###### rollout
            batch_normal, metric_ls_normal = rollout(gbatch, cfg, alg)
            batch_ref, metric_ls_ref = rollout(gbatch, cfg, alg, ref_alg)
            batch, metric_ls = combine_batches_random(batch_normal, batch_ref, metric_ls_normal, metric_ls_ref, cfg)
            
            buffer.add_batch(batch)
            ref_buffer.add_batch(batch_ref)

            logr = logr_scaler(batch[-2][:, -1])
            train_logr_scaled_ls += logr.tolist()
            train_logr_scaled = logr.mean().item()
            train_metric_ls += metric_ls
            train_traj_len = batch[-1].float().mean().item()

            ##### train
            batch_size = min(len(buffer), cfg.batch_size)
            indices = list(range(len(buffer)))
            for _ in range(cfg.tstep):
                if len(indices) == 0:
                    break
                curr_indices = random.sample(indices, min(len(indices), batch_size))
                batch = buffer.sample_from_indices(curr_indices)

                ### Additional batch of ref alg
                batch_ref = ref_buffer.sample_from_indices(curr_indices) 

                train_info = alg.train_step(*batch, reward_exp=reward_exp, logr_scaler=logr_scaler)

                ### Get actions of GFN
                actions_gfn = sample_from_logits(train_info["logits"], gb=batch[0], state=batch[1], done=batch[-1], rand_prob=0.)
                print(actions_gfn)
                print(batch_ref[3])
                reg_ratio.append(len(set(actions_gfn.tolist()).intersection(set(batch_ref[3].tolist())))/len(actions_gfn))

                indices = [i for i in indices if i not in curr_indices]

            if cfg.onpolicy:
                buffer.reset()

            if train_step % cfg.print_freq == 0:
                print(f"Epoch {ep:2d} Data used {train_data_used:.3e}: loss={train_info['train/loss']:.2e}, "
                      + (f"LogZ={train_info['train/logZ']:.2e}, " if cfg.alg in ["tb", "tbbw"] else "")
                      + f"metric size={np.mean(train_metric_ls):.2f}+-{np.std(train_metric_ls):.2f}, "
                      + f"LogR scaled={train_logr_scaled:.2e} traj_len={train_traj_len:.2f}")

            train_step += 1

            ##### eval
            if batch_idx == 0 or train_step % cfg.eval_freq == 0:
                alg.save(alg_save_path)
                metric_curr = np.mean(train_metric_ls[-1000:])
                if metric_curr > metric_best:
                    metric_best = metric_curr
                    print(f"best metric: {metric_best:.2f} at step {train_data_used:.3e}")
                    alg.save(alg_save_path_best)
                if cfg.eval:
                    evaluate(ep, train_step, train_data_used, logr_scaler)
                if cfg.eval_reg:
                    curr_avg, curr_max = evaluate_regularization(ep, train_step, train_data_used, logr_scaler, ref_alg)
                    avg_intersections.append(curr_avg)
                    max_intersections.append(curr_max)
                    
        # Plot loss
        if cfg.plot_loss and (ep % cfg.plot_freq == 0):
            
            # # Reg ratio plot
            plt.plot(reg_ratio, label='Regularization Ratio')
            plt.legend()
            plt.savefig(f"/content/plots/{cfg.run_name}/{ep}_reg_ratio.png")
            plt.close()

            # Intersection plot
            plt.plot(avg_intersections, label='Average Intersection')
            plt.plot(max_intersections, label='Max Intersection')
            plt.legend()
            plt.savefig(f"/content/plots/{cfg.run_name}/{ep}_intersection.png")
            plt.close()

    evaluate(cfg.epochs, train_step, train_data_used, logr_scaler)
    alg.save(alg_save_path)


if __name__ == "__main__":
    main()