import random
import networkx as nx
import copy
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from einops import rearrange, reduce, repeat

from .util import get_decided, pad_batch, get_parent, get_reference_alg, normalize_tuple
from .network import GIN


def sample_from_logits(pf_logits, gb, state, done, rand_prob=0.):
    numnode_per_graph = gb.batch_num_nodes().tolist()
    pf_logits[get_decided(state)] = -np.inf
    pf_logits = pad_batch(pf_logits, numnode_per_graph, padding_value=-np.inf)
    # use -1 to denote impossible action (e.g. for done graphs)
    action = torch.full([gb.batch_size,], -1, dtype=torch.long, device=gb.device)
    pf_undone = pf_logits[~done].softmax(dim=1)
    action[~done] = torch.multinomial(pf_undone, num_samples=1).squeeze(-1)
    if rand_prob > 0.:
        unif_pf_undone = torch.isfinite(pf_logits[~done]).float()
        rand_action_unodone = torch.multinomial(unif_pf_undone, num_samples=1).squeeze(-1)
        rand_mask = torch.rand_like(rand_action_unodone.float()) < rand_prob
        action[~done][rand_mask] = rand_action_unodone[rand_mask]
    return action


class DetailedBalance(object):
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.task = cfg.task
        self.device = device

        assert cfg.arch in ["gin"]
        gin_dict = {"hidden_dim": cfg.hidden_dim, "num_layers": cfg.hidden_layer,
                    "dropout": cfg.dropout, "learn_eps": cfg.learn_eps,
                    "aggregator_type": cfg.aggr}
        self.model = GIN(3, 1, graph_level_output=0, **gin_dict).to(device)
        self.model_flow = GIN(3, 0, graph_level_output=1, **gin_dict).to(device)
        self.params = [
            {"params": self.model.parameters(), "lr": cfg.lr},
            {"params": self.model_flow.parameters(), "lr": cfg.zlr},
        ]
        self.optimizer = torch.optim.Adam(self.params)
        self.leaf_coef = cfg.leaf_coef

    def parameters(self):
        return list(self.model.parameters()) + list(self.model_flow.parameters())

    @torch.no_grad()
    def sample(self, gb, state, done, rand_prob=0., temperature=1., reward_exp=None):
        self.model.eval()
        pf_logits = self.model(gb, state, reward_exp)[..., 0]
        return sample_from_logits(pf_logits / temperature, gb, state, done, rand_prob=rand_prob)

    def save(self, path):
        save_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        save_dict.update({"model_flow": self.model_flow.state_dict()})
        torch.save(save_dict, path)
        print(f"Saved to {path}")

    def load(self, path):
        save_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(save_dict["model"])
        self.model_flow.load_state_dict(save_dict["model_flow"])
        self.optimizer.load_state_dict(save_dict["optimizer"])
        print(f"Loaded from {path}")

    def train_step(self, *batch):
        raise NotImplementedError
    
    @torch.no_grad()
    def compute_logits(self, gb, state, done, reward_exp=None):
        """Compute logits for all actions to calculate likelihoods."""
        self.model.eval()
        pf_logits = self.model(gb, state, reward_exp)[..., 0]
        numnode_per_graph = gb.batch_num_nodes().tolist()
        pf_logits[get_decided(state)] = -np.inf
        pf_logits = pad_batch(pf_logits, numnode_per_graph, padding_value=-np.inf)

        # use -1 to denote impossible action (e.g. for done graphs)
        # action = torch.full([gb.batch_size,], -1, dtype=torch.long, device=gb.device)
        # pf_undone = pf_logits[~done].softmax(dim=0)
        pf_undone = pf_logits[~done]
        return pf_logits, pf_undone

class DetailedBalanceTransitionBuffer(DetailedBalance):
    def __init__(self, cfg, device):
        assert cfg.alg in ["db", "fl"]
        self.forward_looking = (cfg.alg == "fl")
        super(DetailedBalanceTransitionBuffer, self).__init__(cfg, device)

    def train_step(self, *batch, reward_exp=None, logr_scaler=None):
        self.model.train()
        self.model_flow.train()
        torch.cuda.empty_cache()

        gb, s, logr, a, s_next, logr_next, d = batch
        gb, s, logr, a, s_next, logr_next, d = gb.to(self.device), s.to(self.device), logr.to(self.device), \
                    a.to(self.device), s_next.to(self.device), logr_next.to(self.device), d.to(self.device)
        logr, logr_next = logr_scaler(logr), logr_scaler(logr_next)
        numnode_per_graph = gb.batch_num_nodes().tolist()
        batch_size = gb.batch_size

        total_num_nodes = gb.num_nodes()
        gb_two = dgl.batch([gb, gb])
        s_two = torch.cat([s, s_next], dim=0)
        logits = self.model(gb_two, s_two, reward_exp)
        _, flows_out = self.model_flow(gb_two, s_two, reward_exp) # (2 * num_graphs, 1)
        flows, flows_next = flows_out[:batch_size, 0], flows_out[batch_size:, 0]

        pf_logits = logits[:total_num_nodes, ..., 0]
        pf_logits[get_decided(s)] = -np.inf
        pf_logits = pad_batch(pf_logits, numnode_per_graph, padding_value=-np.inf)
        log_pf = F.log_softmax(pf_logits, dim=1)[torch.arange(batch_size), a]

        log_pb = torch.tensor([torch.log(1 / get_parent(s_, self.task).sum())
         for s_ in torch.split(s_next, numnode_per_graph, dim=0)]).to(self.device)

        if self.forward_looking:
            flows_next.masked_fill_(d, 0.) # \tilde F(x) = F(x) / R(x) = 1, log 1 = 0
            lhs = logr + flows + log_pf # (bs,)
            rhs = logr_next + flows_next + log_pb
            loss = (lhs - rhs).pow(2)
            loss = loss.mean()
        else:
            flows_next = torch.where(d, logr_next, flows_next)
            lhs = flows + log_pf # (bs,)
            rhs = flows_next + log_pb
            losses = (lhs - rhs).pow(2)
            loss = (losses[d].sum() * self.leaf_coef + losses[~d].sum()) / batch_size

        return_dict = {"train/loss": loss.item()}
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return return_dict
    
class RegularizedDetailedBalanceTransitionBuffer(DetailedBalance):
    def __init__(self, cfg, device):
        assert cfg.alg in ["db", "fl"]
        self.forward_looking = (cfg.alg == "fl")
        assert len(cfg.ref_alg) > 0
        self.ref_alg = get_reference_alg(cfg)
        super(RegularizedDetailedBalanceTransitionBuffer, self).__init__(cfg, device)

    def train_step(self, *batch, reward_exp=None, logr_scaler=None):
        
        self.model.train()
        self.model_flow.train()
        torch.cuda.empty_cache()

        gb, s, logr, a, s_next, logr_next, d = batch
        gb, s, logr, a, s_next, logr_next, d = gb.to(self.device), s.to(self.device), logr.to(self.device), \
                    a.to(self.device), s_next.to(self.device), logr_next.to(self.device), d.to(self.device)
        logr, logr_next = logr_scaler(logr), logr_scaler(logr_next)
        numnode_per_graph = gb.batch_num_nodes().tolist()
        batch_size = gb.batch_size

        total_num_nodes = gb.num_nodes()
        gb_two = dgl.batch([gb, gb])
        s_two = torch.cat([s, s_next], dim=0)
        logits = self.model(gb_two, s_two, reward_exp)
        _, flows_out = self.model_flow(gb_two, s_two, reward_exp) # (2 * num_graphs, 1)
        flows, flows_next = flows_out[:batch_size, 0], flows_out[batch_size:, 0]      
        pf_logits = logits[:total_num_nodes, ..., 0]
          
        # ACTION FROM REFERENCE ALGORITHM
        _, ref_action_logits = self.ref_alg.sample(gb, s, d)
        
        ref_action_logits_splitted = torch.split(ref_action_logits, numnode_per_graph, dim=0)
        pf_logits_splitted = torch.split(pf_logits, numnode_per_graph, dim=0)
        
        # Scale and weighted sum
        pf_logits_splitted, ref_action_logits_splitted = normalize_tuple(pf_logits_splitted, ref_action_logits_splitted)
        pf_logits = torch.cat(pf_logits_splitted)
        ref_action_logits = torch.cat(ref_action_logits_splitted)

        # gfn, ref = [], []
        # for x, y in zip(pf_logits_splitted, ref_action_logits_splitted):
        #     gfn.append(torch.argmax(x))
        #     ref.append(torch.argmax(y))
        
        # print('GFN action')
        # print(gfn)
        # print('Reference action')
        # print(ref)

        # pf_logits = (1-self.cfg.ref_reg_weight) * pf_logits + self.cfg.ref_reg_weight * ref_action_logits
        
        # Regularization loss
        pf_logits_clone = pf_logits.clone()
        loss_reg = F.mse_loss(pf_logits_clone, ref_action_logits)
        loss_reg = self.cfg.ref_reg_weight * loss_reg

        pf_logits[get_decided(s)] = -np.inf
        pf_logits = pad_batch(pf_logits, numnode_per_graph, padding_value=-np.inf)
        
        # print('Final action')
        # print(torch.max(pf_logits, dim=1))
        
        log_pf = F.log_softmax(pf_logits, dim=1)[torch.arange(batch_size), a]

        log_pb = torch.tensor([torch.log(1 / get_parent(s_, self.task).sum())
         for s_ in torch.split(s_next, numnode_per_graph, dim=0)]).to(self.device)

        if self.forward_looking:
            flows_next.masked_fill_(d, 0.) # \tilde F(x) = F(x) / R(x) = 1, log 1 = 0
            lhs = logr + flows + log_pf # (bs,)
            rhs = logr_next + flows_next + log_pb
            loss = (lhs - rhs).pow(2)
            loss = loss.mean()
            loss +=  loss_reg
            # print(self.cfg.ref_reg_weight * loss_reg/loss)
        else:
            flows_next = torch.where(d, logr_next, flows_next)
            lhs = flows + log_pf # (bs,)
            rhs = flows_next + log_pb
            losses = (lhs - rhs).pow(2)
            loss = (losses[d].sum() * self.leaf_coef + losses[~d].sum()) / batch_size

        return_dict = {"train/loss": loss.item(), 'train/reg_loss_scaled': loss_reg.item()}
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return return_dict
    
    # Overwrite sample method to include reference algorithm
    # @torch.no_grad()
    # def sample(self, gb, state, done, rand_prob=0., temperature=1., reward_exp=None):
        
    #     numnode_per_graph = gb.batch_num_nodes().tolist()

    #     # GFN ACTION
    #     self.model.eval()
    #     pf_logits = self.model(gb, state, reward_exp)[..., 0]
    #     pf_logits_splitted = torch.split(pf_logits, numnode_per_graph, dim=0)
                
    #     # ACTION FROM REFERENCE ALGORITHM
    #     _, ref_action_logits = self.ref_alg.sample(gb, state, done)
    #     ref_action_logits_splitted = torch.split(ref_action_logits, numnode_per_graph, dim=0)
        
    #     # Scale and weighted sum
    #     pf_logits_splitted, ref_action_logits_splitted = normalize_tuple(pf_logits_splitted, ref_action_logits_splitted)
    #     pf_logits = torch.cat(pf_logits_splitted)
    #     ref_action_logits = torch.cat(ref_action_logits_splitted)

    #     # gfn, ref = [], []
    #     # for x, y in zip(pf_logits_splitted, ref_action_logits_splitted):
    #     #     gfn.append(torch.argmax(x))
    #     #     ref.append(torch.argmax(y))
        
    #     # print('GFN action')
    #     # print(gfn)
    #     # print('Reference action')
    #     # print(ref)

    #     # Weighted sum of the logits
    #     pf_logits = (1-self.cfg.ref_reg_weight) * pf_logits + self.cfg.ref_reg_weight * ref_action_logits
        
    #     pf_logits[get_decided(state)] = -np.inf
    #     pf_logits = pad_batch(pf_logits, numnode_per_graph, padding_value=-np.inf)
        
    #     # print('Weighted action')
    #     # print(torch.max(pf_logits, dim=1))
                
    #     return self.sample_from_logits(pf_logits / temperature, gb, state, done, rand_prob=rand_prob)

    # def sample_from_logits(self, pf_logits, gb, state, done, rand_prob=0.):
    #     # use -1 to denote impossible action (e.g. for done graphs)
    #     action = torch.full([gb.batch_size,], -1, dtype=torch.long, device=gb.device)
    #     pf_undone = pf_logits[~done].softmax(dim=1)
    #     action[~done] = torch.multinomial(pf_undone, num_samples=1).squeeze(-1)

    #     # print('Sampled action')
    #     # print(action)

    #     if rand_prob > 0.:
    #         unif_pf_undone = torch.isfinite(pf_logits[~done]).float()
    #         rand_action_unodone = torch.multinomial(unif_pf_undone, num_samples=1).squeeze(-1)
    #         rand_mask = torch.rand_like(rand_action_unodone.float()) < rand_prob
    #         action[~done][rand_mask] = rand_action_unodone[rand_mask]
    #     return action
        
