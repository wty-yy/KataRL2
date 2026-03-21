import torch
from torch import nn
import gymnasium as gym
from torch.distributions.categorical import Categorical


def layer_init(layer, std=2**0.5, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, down_sample=None):
        super().__init__()
        self.conv1 = layer_init(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        )
        self.conv2 = layer_init(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.down_sample = down_sample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        if self.down_sample is not None:
            identity = self.down_sample(x)
        out = self.relu(out + identity)
        return out


def make_layer(in_channels, out_channels, blocks, stride=1):
    down_sample = None
    if stride != 1 or in_channels != out_channels:
        down_sample = nn.Sequential(
            layer_init(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )
        )
    layers = [BasicBlock(in_channels, out_channels, stride, down_sample)]
    for _ in range(1, blocks):
        layers.append(BasicBlock(out_channels, out_channels))
    return nn.Sequential(*layers)


class ResNetDeep(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = layer_init(nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False))
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = make_layer(64, 64, 2)
        self.layer2 = make_layer(64, 128, 2, stride=2)
        self.layer3 = make_layer(128, 256, 2, stride=2)
        self.layer4 = make_layer(256, 512, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = layer_init(nn.Linear(512, 512))

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class Agent(nn.Module):
    def __init__(self, action_space: gym.spaces.Discrete):
        super().__init__()
        self.network = ResNetDeep()
        self.actor = layer_init(nn.Linear(512, action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None, train: bool = True):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        if not train:
            action = logits.argmax(dim=-1)
            return action, None, None, None
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
