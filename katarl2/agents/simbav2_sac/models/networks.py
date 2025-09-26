import torch
import torch.nn as nn
from tensordict import from_modules
from copy import deepcopy
import math
from katarl2.agents.simbav2_sac.models.layers import (
    HyperDense,
    HyperEmbedder,
    HyperLERPBlock,
    HyperNormalPolicyHead,
    HyperDiscreteValueHead,
)
from katarl2.agents.simbav2_sac.models.utils import l2normalize
from dataclasses import dataclass


@dataclass
class ModelConfig:
    num_blocks: int
    hidden_dim: int
    c_shift: float
    scaler_init: str | float = 'auto'   # sqrt(2 / hidden_dim)
    scaler_scale: str | float = 'auto'  # sqrt(2 / hidden_dim)
    alpha_init: str | float = 'auto'    # 1 / (num_blocks + 1)
    alpha_scale: str | float = 'auto'   # 1 / sqrt(hidden_dim)

def update_model_config(cfg: ModelConfig):
    if cfg.scaler_init == 'auto':
        cfg.scaler_init = math.sqrt(2 / cfg.hidden_dim)
    if cfg.scaler_scale == 'auto':
        cfg.scaler_scale = math.sqrt(2 / cfg.hidden_dim)
    if cfg.alpha_init == 'auto':
        cfg.alpha_init = 1 / (cfg.num_blocks + 1)
    if cfg.alpha_scale == 'auto':
        cfg.alpha_scale = 1 / math.sqrt(cfg.hidden_dim)

LOG_STD_MAX = 2
LOG_STD_MIN = -10

class SimbaV2Actor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        action_dim: int,
        cfg: ModelConfig,
    ):
        super().__init__()
        self.embedder = HyperEmbedder(
            in_dim=in_dim,
            hidden_dim=cfg.hidden_dim,
            scaler_init=cfg.scaler_init,
            scaler_scale=cfg.scaler_scale,
            c_shift=cfg.c_shift,
        )
        self.encoder = nn.Sequential(
            *[
                HyperLERPBlock(
                    in_dim=cfg.hidden_dim,
                    hidden_dim=cfg.hidden_dim,
                    scaler_init=cfg.scaler_init,
                    scaler_scale=cfg.scaler_scale,
                    alpha_init=cfg.alpha_init,
                    alpha_scale=cfg.alpha_scale,
                )
                for _ in range(cfg.num_blocks)
            ]
        )
        self.predictor = HyperNormalPolicyHead(
            in_dim=cfg.hidden_dim,
            hidden_dim=cfg.hidden_dim,
            action_dim=action_dim,
            scaler_init=1.0,
            scaler_scale=1.0,
        )

    def forward(self, obs):
        y = self.embedder(obs)
        z = self.encoder(y)
        mean, log_std = self.predictor(z)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std
    
    def get_action(self, x, train: bool = True):
        mean, log_std = self(x)
        std = log_std.exp()
        if train:
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
            log_prob = normal.log_prob(x_t)
        else:
            x_t = mean
            log_prob = torch.zeros_like(x_t)
        action = torch.tanh(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - action.pow(2)).clamp(1e-6))
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob


class SimbaV2Critic(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_bins: int,
        min_v: float,
        max_v: float,
        cfg: ModelConfig,
    ):
        super().__init__()
        self.embedder = HyperEmbedder(
            in_dim=in_dim,
            hidden_dim=cfg.hidden_dim,
            scaler_init=cfg.scaler_init,
            scaler_scale=cfg.scaler_scale,
            c_shift=cfg.c_shift,
        )
        self.encoder = nn.Sequential(
            *[
                HyperLERPBlock(
                    in_dim=cfg.hidden_dim,
                    hidden_dim=cfg.hidden_dim,
                    scaler_init=cfg.scaler_init,
                    scaler_scale=cfg.scaler_scale,
                    alpha_init=cfg.alpha_init,
                    alpha_scale=cfg.alpha_scale,
                )
                for _ in range(cfg.num_blocks)
            ]
        )
        self.predictor = HyperDiscreteValueHead(
            in_dim=cfg.hidden_dim,
            hidden_dim=cfg.hidden_dim,
            num_bins=num_bins,
            scaler_init=1.0,
            scaler_scale=1.0,
        )
        self.bin_values = torch.linspace(min_v, max_v, num_bins).reshape(1, num_bins)
        self.bin_values = nn.Parameter(self.bin_values, requires_grad=False)
    
    def forward(self, obs, action):
        x = torch.concat([obs, action], dim=-1)
        y = self.embedder(x)
        z = self.encoder(y)
        logist = self.predictor(z)
        log_prob = torch.log_softmax(logist, dim=-1)
        value = (self.bin_values * log_prob.exp()).sum(-1)
        return value, log_prob


class Temperature(nn.Module):
    def __init__(self, initial_value=0.01):
        super().__init__()
        self.log_temp = nn.Parameter(torch.ones([]) * math.log(initial_value))

    def __call__(self):
        return torch.exp(self.log_temp)


class Ensemble(nn.Module):
    """
    Vectorized ensemble of modules.
    """

    def __init__(self, modules, **kwargs):
        super().__init__()
        # combine_state_for_ensemble causes graph breaks
        self.params = from_modules(*modules, as_module=True)
        with self.params[0].data.to("meta").to_module(modules[0]):
            self.module = deepcopy(modules[0])
        self._repr = str(modules[0])
        self._n = len(modules)

    def __len__(self):
        return self._n

    def _call(self, params, *args, **kwargs):
        with params.to_module(self.module):
            return self.module(*args, **kwargs)

    def forward(self, *args, **kwargs):
        # (0, None, None): for **two-args** forward call, like value_net(obs, action)
        return torch.vmap(self._call, (0, None, None), randomness="different")(
            self.params, *args, **kwargs
        )

    def __repr__(self):
        return f"Vectorized {len(self)}x " + self._repr


@torch.no_grad()
def l2normalize_network(network):
    """Apply L2 normalization to all hyper-dense layers in the network"""

    def norm_ensemble(name, tensor):
        if "hyper_dense" in name:
            assert tensor.ndim == 3
            tensor.set_(l2normalize(tensor, dim=1))

    def norm(m):
        if isinstance(m, HyperDense):
            assert m.hyper_dense.weight.ndim == 2
            m.hyper_dense.weight.set_(l2normalize(m.hyper_dense.weight, dim=0))

    if isinstance(
        network, Ensemble
    ):  # Params of Ensemble cannot be accessed via nn.Module.apply; apply manually instead.
        network.params.named_apply(norm_ensemble, nested_keys=True)
    else:
        network.apply(norm)
