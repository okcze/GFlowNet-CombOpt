import sys, os
import gzip, pickle
from time import time, sleep
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, open_dict, OmegaConf

import random
import numpy as np
import torch
import dgl
from einops import rearrange, reduce, repeat

from .data import get_data_loaders
from .util import seed_torch, TransitionBuffer, get_mdp_class
from .algorithm import DetailedBalanceTransitionBuffer
from .ref_alg.mis_greedy import MISGreedy
from .ref_alg.mis_heuristic import MISHeuristic

torch.backends.cudnn.benchmark = True


def get_alg_buffer(cfg, device):
    assert cfg.alg in ["db", "fl"]
    buffer = TransitionBuffer(cfg.tranbuff_size, cfg)
    alg = DetailedBalanceTransitionBuffer(cfg, device)
    return alg, buffer

def get_saved_alg_buffer(cfg, device):
    assert cfg.alg in ["db", "fl"]
    buffer = TransitionBuffer(cfg.tranbuff_size, cfg)
    alg = DetailedBalanceTransitionBuffer(cfg, device)
    alg.load(cfg.alg_load_path)
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
def rollout(gbatch, cfg, alg):
    # Return state
    env = get_mdp_class(cfg.task)(gbatch, cfg)
    state = env.state
    return state   


@hydra.main(config_path="configs", config_name="main") # for hydra-core==1.1.0
# @hydra.main(version_base=None, config_path="configs", config_name="main") # for newer hydra
def sample(cfg: DictConfig):

    cfg = refine_cfg(cfg)
    device = torch.device(f"cuda:{cfg.device:d}" if torch.cuda.is_available() and cfg.device>=0 else "cpu")
    print(f"Device: {device}")
    alg, buffer = get_saved_alg_buffer(cfg, device)
    seed_torch(cfg.seed)
    print(str(cfg))
    print(f"Work directory: {os.getcwd()}")

    _, test_loader = get_data_loaders(cfg)
    
    @torch.no_grad()
    def evaluate(ep, logr_scaler):
        
        torch.cuda.empty_cache()
        pbar = tqdm(enumerate(test_loader))        
                
        for batch_idx, gbatch in pbar:

            env = get_mdp_class(cfg.task)(gbatch, cfg)
            gbatch = gbatch.to(device)

            # Itared over each graph in the batch
            for graph, dgl_g in enumerate(dgl.unbatch(gbatch)):
                if torch.cuda.is_available():
                    nx_g = dgl_g.cpu().to_networkx()
                else:
                    nx_g = dgl_g.to_networkx()
                for ref_alg in [MISHeuristic, MISGreedy]:
                    alg = ref_alg()
                    # Check if store directory exists
                    if not os.path.exists(f'/content/GFlowNet-CombOpt/states/{ref_alg.__name__}'):
                        os.makedirs(f'/content/GFlowNet-CombOpt/states/{ref_alg.__name__}')
                    snapshots = alg.algorithm(nx_g)
                    np.save(f'/content/GFlowNet-CombOpt/states/{ref_alg.__name__}/{batch_idx}_{graph}', snapshots)
            
        return

    ##### sample
    process_ratio = 1
    reward_exp = None
    logr_scaler = get_logr_scaler(cfg, process_ratio=process_ratio, reward_exp=reward_exp)
    evaluate(cfg.epochs, logr_scaler)
    return


if __name__ == "__main__":
    sample()