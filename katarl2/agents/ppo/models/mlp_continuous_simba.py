import torch
from torch import nn
import numpy as np
import gymnasium as gym
from torch.distributions.normal import Normal
from katarl2.agents.common.networks.residual_encoder import init_orthogonal_linear, ResidualEncoder

class Agent(nn.Module):
    def __init__(
            self, obs_space: gym.spaces.Box, act_space: gym.spaces.Box,
            critic_num_blocks: int, critic_hidden_dim: int,
            actor_num_blocks: int, actor_hidden_dim: int
        ):
        super().__init__()
        fc_value = nn.Linear(critic_hidden_dim, 1)
        init_orthogonal_linear(fc_value, gain=1.0)
        self.critic = nn.Sequential(
            ResidualEncoder(
                in_dim=np.array(obs_space.shape).prod(),
                hidden_dim=critic_hidden_dim,
                num_blocks=critic_num_blocks
            ),
            fc_value
        )

        fc_mean = nn.Linear(actor_hidden_dim, np.prod(act_space.shape))
        init_orthogonal_linear(fc_mean, gain=1.0)
        self.actor_mean = nn.Sequential(
            ResidualEncoder(
                in_dim=np.array(obs_space.shape).prod(),
                hidden_dim=actor_hidden_dim,
                num_blocks=actor_num_blocks
            ),
            fc_mean
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(act_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, train: bool = True):
        action_mean = self.actor_mean(x)
        if not train:
            return action_mean, None, None, None
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
