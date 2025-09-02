import torch
from torch import nn
import gymnasium as gym
from torch.distributions.categorical import Categorical
from katarl2.agents.ppo.ppo_cfg import PPODiscreteConfig
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
        
class Agent_IN(ActorCritic):
    def __init__(self, action_space: gym.spaces.Discrete):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.Mish(),
            nn.GroupNorm(1, 32),
            
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.Mish(),
            nn.GroupNorm(1, 64),
            
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.Mish(),
            nn.GroupNorm(1, 64),
            nn.Flatten(),
            
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
            nn.LayerNorm(512),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(512, action_space.n), std=0.01)
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(512, 1), std=1)
        )
        
class Agent_LN(ActorCritic):
    def __init__(self, action_space: gym.spaces.Discrete):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),  # (4, 84, 84) -> (32, 20, 20)
            nn.Mish(),
            nn.LayerNorm([32, 20, 20]),
            
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),  # (64, 9, 9)
            nn.Mish(),
            nn.LayerNorm([64, 9, 9]),
            
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),  # (64, 7, 7)
            nn.Mish(),
            nn.LayerNorm([64, 7, 7]),
            nn.Flatten(),
            
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
            nn.LayerNorm(512),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(512, action_space.n), std=0.01)
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(512, 1), std=1)
        )
        
class Agent_IN_before_norm(ActorCritic):
    def __init__(self, action_space: gym.spaces.Discrete):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.GroupNorm(1, 32),
            nn.Mish(),
            
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.GroupNorm(1, 64),
            nn.Mish(),
            
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.GroupNorm(1, 64),
            nn.Mish(),
            nn.Flatten(),
            
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.LayerNorm(512),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(512, action_space.n), std=0.01)
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(512, 1), std=1)
        )
        
class Agent_LN_before_norm(ActorCritic):
    def __init__(self, action_space: gym.spaces.Discrete):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),  # (4, 84, 84) -> (32, 20, 20)
            nn.LayerNorm([32, 20, 20]),
            nn.Mish(),
            
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),  # (64, 9, 9)
            nn.LayerNorm([64, 9, 9]),
            nn.Mish(),
            
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),  # (64, 7, 7)
            nn.LayerNorm([64, 7, 7]),
            nn.Mish(),
            nn.Flatten(),
            
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.LayerNorm(512),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(512, action_space.n), std=0.01)
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(512, 1), std=1)
        )
        



