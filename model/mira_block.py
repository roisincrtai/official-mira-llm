"""model/mira_block.py -- MiraBlock, one MIRA layer: RMSNorm, the MIRA mixer (MiraMixer of
model/mira_mixer.py) on the residual stream, RMSNorm, and the feed-forward network on the residual
stream. model/mira_llm.py assembles L of them and the head."""
from __future__ import annotations

import importlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# THROUGH importlib, as everywhere in model/: this module is imported while the `model` package is
# still being initialised.
mira_mixer = importlib.import_module("model.mira_mixer")

EXPERT_INIT_STD = 0.02
BALANCE_BIAS_RATE = 1e-3


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-8, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(int(width)))
        self.bias = nn.Parameter(torch.zeros(int(width))) if bias else None
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        z = x.float()
        z = z * torch.rsqrt(z.square().mean(dim=-1, keepdim=True) + self.eps)
        z = z * self.weight.float()
        if self.bias is not None:
            z = z + self.bias.float()
        return z.to(dtype)


class FeedForward(nn.Module):
    def __init__(self, width: int, dropout: float = 0.0, hidden: int = None):
        super().__init__()
        # THE SCHEME'S ffn_width (config/default/config_mira.yaml); 4 x width when there is none.
        hidden = 4 * int(width) if not hidden else int(hidden)
        self.up = nn.Linear(width, hidden)
        self.down = nn.Linear(hidden, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.up(x))))


class MoEFeedForward(nn.Module):
    def __init__(self, width: int, experts: int, active: int, expert_width: int,
                 shared: int = 0, dropout: float = 0.0):
        super().__init__()
        width, experts, active = int(width), int(experts), int(active)
        expert_width, shared = int(expert_width), int(shared)
        if experts < 2 or not 1 <= active <= experts or expert_width < 1 or shared < 0:
            raise ValueError(f"MoE FFN needs 2 <= experts, 1 <= active <= experts, "
                             f"expert_width >= 1 and shared >= 0; got experts={experts}, "
                             f"active={active}, expert_width={expert_width}, shared={shared}")
        self.width, self.experts, self.active = width, experts, active
        self.expert_width, self.shared = expert_width, shared
        self.router = nn.Parameter(torch.empty(experts, width))
        self.expert_up = nn.Parameter(torch.empty(experts, expert_width, width))
        self.expert_up_bias = nn.Parameter(torch.zeros(experts, expert_width))
        self.expert_down = nn.Parameter(torch.empty(experts, width, expert_width))
        self.expert_down_bias = nn.Parameter(torch.zeros(experts, width))
        if shared:
            self.shared_up = nn.Parameter(torch.empty(shared, expert_width, width))
            self.shared_up_bias = nn.Parameter(torch.zeros(shared, expert_width))
            self.shared_down = nn.Parameter(torch.empty(shared, width, expert_width))
            self.shared_down_bias = nn.Parameter(torch.zeros(shared, width))
        else:
            for name in ("shared_up", "shared_up_bias", "shared_down", "shared_down_bias"):
                self.register_parameter(name, None)
        self.register_buffer("balance_bias", torch.zeros(experts))
        self.dropout = nn.Dropout(dropout)
        self.down_bias_here = True
        self.expert_load = []
        self.reset_parameters()

    def reset_parameters(self, n_layer: int = None):
        bound = EXPERT_INIT_STD * math.sqrt(3.0)
        down_bound = (bound / math.sqrt(2 * int(n_layer)) if n_layer else bound)
        with torch.no_grad():
            self.router.uniform_(-bound, bound)
            self.expert_up.uniform_(-bound, bound)
            self.expert_down.uniform_(-down_bound, down_bound)
            self.expert_up_bias.zero_()
            self.expert_down_bias.zero_()
            if self.shared:
                self.shared_up.uniform_(-bound, bound)
                self.shared_down.uniform_(-down_bound, down_bound)
                self.shared_up_bias.zero_()
                self.shared_down_bias.zero_()
        return self

    def _expert(self, x, up, up_bias, down, down_bias):
        hidden = F.silu(F.linear(x, up, up_bias))
        return F.linear(hidden, down, down_bias if self.down_bias_here else down_bias * 0.0)

    def forward(self, x):
        shape = x.shape
        tokens = x.reshape(-1, shape[-1])
        scores = torch.sigmoid(F.linear(tokens.float(), self.router.float()))
        chosen = torch.topk(scores.detach() + self.balance_bias.float(), self.active,
                            dim=-1).indices
        gates = scores.gather(-1, chosen)
        gates = gates / gates.sum(dim=-1, keepdim=True)
        flat = chosen.reshape(-1)
        load = torch.bincount(flat, minlength=self.experts)
        if self.training and torch.is_grad_enabled():
            self.expert_load.append(load.detach())
        order = torch.argsort(flat)
        rows = order // self.active
        sorted_gates = gates.reshape(-1)[order]
        out = torch.zeros_like(tokens)
        start = 0
        for expert, count in enumerate(load.tolist()):
            if count:
                picked = rows[start:start + count]
                y = self._expert(tokens.index_select(0, picked), self.expert_up[expert],
                                 self.expert_up_bias[expert], self.expert_down[expert],
                                 self.expert_down_bias[expert])
                out = out.index_add(0, picked,
                                    (y * sorted_gates[start:start + count, None]).to(out.dtype))
            start += count
        for index in range(self.shared):
            out = out + self._expert(tokens, self.shared_up[index], self.shared_up_bias[index],
                                     self.shared_down[index],
                                     self.shared_down_bias[index]).to(out.dtype)
        return self.dropout(out.reshape(shape))

    def update_balance(self):
        if not self.expert_load:
            return
        load = torch.stack([count.to(self.balance_bias.device)
                            for count in self.expert_load]).sum(dim=0).float()
        self.expert_load.clear()
        from model import parallel
        if parallel.dist_active():
            import torch.distributed as dist
            staged = load.to(parallel._coll_device())
            dist.all_reduce(staged)
            load = staged.to(self.balance_bias.device)
        with torch.no_grad():
            self.balance_bias.add_(BALANCE_BIAS_RATE * torch.sign(load.mean() - load))


def update_moe_balance(model):
    for module in model.modules():
        if isinstance(module, MoEFeedForward):
            module.update_balance()


class MiraBlock(nn.Module):
    def __init__(
        self,
        width: int,
        dropout: float = 0.0,
        lattice_order: int = None,
        d_e: int = None,
        coefficient_bottleneck: int = None,
        memory_dimension: int = None,
        ffn_hidden: int = None,
        positive_poles: bool = False,
        lattice_initializer: str = "W_uniform_b_jeffreys",
        lattice_init_horizon=None,
        arch: str = "cauchy_x_hyperbolic",
        moe: dict = None,
    ):
        super().__init__()
        if lattice_order is None:
            lattice_order = mira_mixer._configured("lattice_order")
        if d_e is None:
            d_e = mira_mixer._configured("lattice_d_e")
        if coefficient_bottleneck is None:
            coefficient_bottleneck = mira_mixer._configured("coefficient_bottleneck")
        if memory_dimension is None:
            memory_dimension = mira_mixer._configured("memory_dimension")
        self.rn1 = RMSNorm(width)
        # EVERY ARGUMENT PAST THE GEOMETRY IS PASSED BY KEYWORD. Positionally, a new parameter
        # inserted anywhere but the end of the mixer's signature silently shifts the rest.
        self.mixer = mira_mixer.MiraMixer(
            width, lattice_order, d_e, coefficient_bottleneck, memory_dimension,
            dropout=dropout, positive_poles=positive_poles,
            lattice_initializer=lattice_initializer,
            lattice_init_horizon=lattice_init_horizon, arch=arch)
        self.rn2 = RMSNorm(width)
        self.ffn = (MoEFeedForward(width, dropout=dropout, **moe) if moe
                    else FeedForward(width, dropout, hidden=ffn_hidden))

    def post_init(self, n_layer: int):
        self.mixer.initialize_lattice_parameters()
        std = 0.02 / math.sqrt(2 * int(n_layer))
        nn.init.normal_(self.mixer.out_proj.weight, mean=0.0, std=std)
        if isinstance(self.ffn, MoEFeedForward):
            self.ffn.reset_parameters(n_layer)
        else:
            nn.init.normal_(self.ffn.down.weight, mean=0.0, std=std)
        return self

    def forward(self, x, attn_mask=None, cache=None, layer=None):
        x = x + self.mixer(self.rn1(x), attn_mask, cache, layer)
        return x + self.ffn(self.rn2(x))
