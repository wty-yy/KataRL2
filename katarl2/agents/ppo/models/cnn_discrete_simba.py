import torch
from torch import nn
import gymnasium as gym
from torch.distributions.categorical import Categorical
from katarl2.agents.common.networks.residual_encoder import ResidualEncoder
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.output_dim = 64 * 7 * 7
    
    def forward(self, x):
        return self.network(x)

class Actor(nn.Module):
    def __init__(self, action_space: gym.spaces.Discrete, hidden_dim: int, num_blocks: int):
        super().__init__()
        self.cnn_encoder = CNNEncoder()
        self.res_encoder = ResidualEncoder(self.cnn_encoder.output_dim, hidden_dim, num_blocks)
        self.fc_out = layer_init(nn.Linear(hidden_dim, action_space.n), std=0.01)
    
    def forward(self, x):
        x = self.cnn_encoder(x)
        x = self.res_encoder(x)
        return self.fc_out(x)

class Critic(nn.Module):
    def __init__(self, hidden_dim: int, num_blocks: int):
        super().__init__()
        self.cnn_encoder = CNNEncoder()
        self.res_encoder = ResidualEncoder(self.cnn_encoder.output_dim, hidden_dim, num_blocks)
        self.fc_out = layer_init(nn.Linear(hidden_dim, 1), std=1.0)
    
    def forward(self, x):
        x = self.cnn_encoder(x)
        x = self.res_encoder(x)
        return self.fc_out(x)

class Agent(nn.Module):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            actor_hidden_dim: int,
            actor_num_blocks: int,
            critic_hidden_dim: int,
            critic_num_blocks: int,
        ):
        super().__init__()
        self.actor = Actor(action_space, hidden_dim=actor_hidden_dim, num_blocks=actor_num_blocks)
        self.critic = Critic(hidden_dim=critic_hidden_dim, num_blocks=critic_num_blocks)

    def get_value(self, x):
        return self.critic(x / 255.0)

    def get_action_and_value(self, x, action = None, train: bool = True):
        x = x / 255.0
        logits = self.actor(x)
        if not train:
            action = logits.argmax(dim=-1)
            return action, None, None, None
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    