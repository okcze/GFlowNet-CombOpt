# Main training script for regularized GFN with heuristic algorithms guidance
# Parameters:
# - task
# - ref_alg
# - frac_replaced
# - reward_boost
# - epochs

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
from .util import get_reference_alg, seed_torch, TransitionBuffer, get_mdp_class, manage_metrics_dir
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
def rollout(gbatch, cfg, alg, ref_alg, frac_replaced):
    env = get_mdp_class(cfg.task)(gbatch, cfg)
    num_replaced = int(frac_replaced * len(env.done))
    state = env.state

    ##### Select N trajectories for replacement
    batch_size = len(env.done)
    replace_indices = torch.randperm(batch_size)[:num_replaced]
    replace_flags = torch.zeros(batch_size, dtype=torch.bool)
    replace_flags[replace_indices] = True

    ##### Initialize trajectory storage
    traj_s, traj_r, traj_a, traj_d = [], [], [], []

    ##### Sample trajectories
    reward_exp_eval = None
    while not all(env.done):
        # Sample actions for the whole batch using both algorithms
        actions_alg = alg.sample(gbatch, state, env.done, rand_prob=cfg.randp, reward_exp=reward_exp_eval)
        actions_preferred, _ = ref_alg.sample(gbatch, state, env.done, rand_prob=cfg.randp)
        
        # Initialize the action tensor for this step
        action = torch.empty_like(env.done, dtype=actions_alg.dtype)
        
        # Use actions from `ref_alg` for the trajectories marked for replacement, else use `alg`
        action[replace_flags] = actions_preferred[replace_flags]
        action[~replace_flags] = actions_alg[~replace_flags] 
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

    ### Boost rewards
    traj_r[replace_flags] = traj_r[replace_flags] * cfg.reward_boost
        
    traj_len = 1 + torch.sum(~traj_d, dim=1) # (batch_size, )

    ##### graph, state, action, done, reward, trajectory length
    batch = gbatch.cpu(), traj_s.cpu(), traj_a.cpu(), traj_d.cpu(), traj_r.cpu(), traj_len.cpu()
    return batch, env.batch_metric(state)

@hydra.main(config_path="configs", config_name="main") # for hydra-core==1.1.0
# @hydra.main(version_base=None, config_path="configs", config_name="main") # for newer hydra
def main(cfg: DictConfig):
    cfg = refine_cfg(cfg)
    device = torch.device(f"cuda:{cfg.device:d}" if torch.cuda.is_available() and cfg.device>=0 else "cpu")
    print(f"Device: {device}")

    if len(cfg.alg_load_path) > 0:
        alg, buffer = get_saved_alg_buffer(cfg, device, cfg.alg_load_path)
        print(f"Loaded model from {cfg.alg_load_path}")
    else:
        alg, buffer = get_alg_buffer(cfg, device)
        print(f"New model {alg.__class__.__name__}")

    ref_alg = get_reference_alg(cfg)
    seed_torch(cfg.seed)
    print(str(cfg))
    print(f"Work directory: {os.getcwd()}")

    ### Create directory for saving regularization metrics
    if cfg.plot_loss:
        path = f"/content/results/{cfg.run_name}"
        manage_metrics_dir(path)

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
            gbatch = dgl.batch([gbatch] * num_repeat)
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

            results_intersection = [
                (
                    len(set(np.where(ref_sol == 1)[0]) & set(np.where(candidate == 1)[0]))
                    / len(np.where(ref_sol == 1)[0])
                    if np.any(ref_sol == 1) else 0.0
                )
                for ref_sol, candidate in zip(states_pref, states)
            ]
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
        
        avg_intersection = np.mean(intersections)
        max_intersection = np.max(intersections)
        print(f"Average intersection: {avg_intersection:.2f}")
        print(f"Max intersection: {max_intersection:.2f}")

        return avg_intersection, max_intersection, intersections
    
    @torch.no_grad()
    def measure_actions(batch, alg, ref_alg):
        """Measure the ratio of GFN to reference algorithm overlap in current batch."""
        gb, s, logr, a, s_next, logr_next, d = batch
        gb, s, logr, a, s_next, logr_next, d = gb.to(alg.device), s.to(alg.device), logr.to(alg.device), \
                    a.to(alg.device), s_next.to(alg.device), logr_next.to(alg.device), d.to(alg.device)
        actions_gfn = alg.sample(gb, s, d)
        actions_ref, _ = ref_alg.sample(gbatch_rep=gb, state=s, done=d)
        curr_reg_ration = torch.mean((actions_gfn == actions_ref).float()).item()
        return curr_reg_ration
    
    # Store loss to plot
    reg_ratio = []
    avg_intersections = []
    max_intersections = []
    intersections = []
    avg_metric = []

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
            batch, metric_ls = rollout(gbatch, cfg, alg, ref_alg, cfg.frac_replaced)
            
            buffer.add_batch(batch)
            
            logr = logr_scaler(batch[-2][:, -1])
            train_logr_scaled_ls += logr.tolist()
            train_logr_scaled = logr.mean().item()
            train_metric_ls += metric_ls
            train_traj_len = batch[-1].float().mean().item()

            ##### train
            batch_size = min(len(buffer), cfg.batch_size)
            indices = list(range(len(buffer)))
            for tstep in range(cfg.tstep):
                if len(indices) == 0:
                    break
                curr_indices = random.sample(indices, min(len(indices), batch_size))
                batch = buffer.sample_from_indices(curr_indices)
                train_info = alg.train_step(*batch, reward_exp=reward_exp, logr_scaler=logr_scaler)

                ### Get actions of GFN
                if cfg.eval_reg and tstep % cfg.reg_ratio_freq == 0:
                    curr_reg_ration = measure_actions(batch=batch, alg=alg, ref_alg=ref_alg)
                    reg_ratio.append(curr_reg_ration)

                indices = [i for i in indices if i not in curr_indices]

            if cfg.onpolicy:
                buffer.reset()

            if train_step % cfg.print_freq == 0:
                print(f"Epoch {ep:2d} Data used {train_data_used:.3e}: loss={train_info['train/loss']:.2e}, "
                      + (f"LogZ={train_info['train/logZ']:.2e}, " if cfg.alg in ["tb", "tbbw"] else "")
                      + f"metric size={np.mean(train_metric_ls):.2f}+-{np.std(train_metric_ls):.2f}, "
                      + f"LogR scaled={train_logr_scaled:.2e} traj_len={train_traj_len:.2f}, "
                      + f"avg reg ratio={np.mean(reg_ratio):.2f}")
                avg_metric.append(np.mean(train_metric_ls))

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
                    curr_avg, curr_max, curr_intersections = evaluate_regularization(ep, train_step, train_data_used, logr_scaler, ref_alg)
                    avg_intersections.append(curr_avg)
                    max_intersections.append(curr_max)
                    intersections += curr_intersections
                    
        # Plot loss
        if cfg.plot_loss and (ep % cfg.plot_freq == 0):
            
            # Reg ratio plot
            plt.plot(reg_ratio, label='Regularization Ratio')
            plt.xlabel('Train Step')
            plt.ylabel('Regularization Ratio')
            plt.legend()
            plt.savefig(f"{path}/plots/{ep}_reg_ratio.png")
            plt.close()

            # Intersection plot
            plt.plot(avg_intersections, label='Average Intersection')
            plt.plot(max_intersections, label='Max Intersection')
            plt.xlabel('Train Step')
            plt.legend()
            plt.savefig(f"{path}/plots/{ep}_intersection.png")
            plt.close()

            # Metric plot
            plt.plot(avg_metric, label='Metric')
            plt.xlabel('Train Step')
            plt.ylabel('Metric')
            plt.legend()
            plt.savefig(f"{path}/plots/{ep}_metric.png")
            plt.close()

        # Save metrics each epoch
        np.array(reg_ratio).dump(f"{path}/metrics/reg_ratio.npy")
        np.array(intersections).reshape(cfg.epochs, -1).dump(f"{path}/metrics/intersections.npy")
        np.array(avg_metric).dump(f"{path}/metrics/avg_metric.npy")

    evaluate(cfg.epochs, train_step, train_data_used, logr_scaler)
    alg.save(alg_save_path)


if __name__ == "__main__":
    main()