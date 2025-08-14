import torch
from torch import nn
import gymnasium as gym
from torch.distributions.categorical import Categorical
from katarl2.agents.ppo.ppo_cfg import PPOConfig
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    network: nn.Module
    actor: nn.Module
    critic: nn.Module

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action = None, train: bool = True):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        if not train:
            action = logits.argmax(dim=-1)
            return action, None, None, None
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

class Agent(ActorCritic):
    def __init__(self, action_space: gym.spaces.Discrete):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

class Agent_GN_LN(ActorCritic):
    def __init__(self, action_space: gym.spaces.Discrete):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.Mish(),
            nn.GroupNorm(4, 32),
            
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.LeakyReLU(0.1),
            nn.GroupNorm(8, 64),
            
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.GroupNorm(8, 64),
            nn.Flatten(),
            
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
            nn.LayerNorm(512)
        )

        self.actor = nn.Sequential(
            nn.ReLU(),
            layer_init(nn.Linear(512, action_space.n), std=0.01)
        )

        self.critic = nn.Sequential(
            nn.Mish(),
            layer_init(nn.Linear(512, 1), std=1)
        )
        