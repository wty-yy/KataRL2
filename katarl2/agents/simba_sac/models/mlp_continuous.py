""" SAC 处理连续动作输入 """
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as td

from gymnasium.spaces import Box
import numpy as np
from katarl2.agents.common.networks.residual_encoder import ResidualEncoder, init_orthogonal_linear

class SoftQNetwork(nn.Module):
    def __init__(
            self,
            observation_space: Box, action_space: Box,
            num_blocks: int, hidden_dim: int
        ):
        super().__init__()
        self.encoder = ResidualEncoder(
            in_dim=np.array(observation_space.shape).prod() + np.prod(action_space.shape),
            hidden_dim=hidden_dim,
            num_blocks=num_blocks
        )
        self.fc = nn.Linear(hidden_dim, 1)
        init_orthogonal_linear(self.fc, gain=1.0)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.encoder(x)
        x = self.fc(x)
        return x

LOG_STD_MAX = 2
LOG_STD_MIN = -10

class Actor(nn.Module):
    def __init__(
            self,
            observation_space: Box, action_space: Box,
            num_blocks: int, hidden_dim: int
        ):
        super().__init__()
        self.encoder = ResidualEncoder(
            in_dim=np.array(observation_space.shape).prod(),
            hidden_dim=hidden_dim,
            num_blocks=num_blocks
        )
        self.fc_mean = nn.Linear(hidden_dim, np.prod(action_space.shape))
        self.fc_logstd = nn.Linear(hidden_dim, np.prod(action_space.shape))
        init_orthogonal_linear(self.fc_mean, gain=1.0)
        init_orthogonal_linear(self.fc_logstd, gain=1.0)

    def forward(self, x):
        x = self.encoder(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
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