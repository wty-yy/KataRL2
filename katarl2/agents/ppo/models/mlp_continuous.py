import torch
from torch import nn
import numpy as np
import gymnasium as gym
from torch.distributions.normal import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, obs_space: gym.spaces.Box, act_space: gym.spaces.Box, policy_layers: int = 7):
        super().__init__()
        obs_dim = np.array(obs_space.shape).prod()
        act_dim = np.prod(act_space.shape)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        if policy_layers == 3:
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(obs_dim, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, act_dim), std=0.01),
            )
        elif policy_layers == 7:
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(obs_dim, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 128)),
                nn.Tanh(),
                layer_init(nn.Linear(128, 128)),
                nn.Tanh(),
                layer_init(nn.Linear(128, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, act_dim), std=0.01),
            )
        else:
            raise ValueError(f"Unsupported policy_layers={policy_layers}, expected 3 or 7.")
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, train: bool = True):
        action_mean = self.actor_mean(x)
        if not train:
            return action_mean, None, None, None, None
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x), {
            "mean": action_mean,
            "std": action_std,
        }
