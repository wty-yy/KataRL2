import numpy as np
from functools import partial
from typing import Optional, Any
from dataclasses import dataclass
from katarl2.envs import GymAtariEnvConfig, GymMujocoEnvConfig, DMCEnvConfig, EnvpoolAtariEnvConfig

@dataclass
class PPOEnvpoolAtariEnvConfig(EnvpoolAtariEnvConfig):
    num_envs: int = 8

@dataclass
class PPOGymAtariEnvConfig(GymAtariEnvConfig):
    num_envs: int = 8

@dataclass
class PPOGymMujocoEnvConfig(GymMujocoEnvConfig):
    num_envs: int = 1

    normalize_observation: bool = True
    transform_observation: Optional[Any] = partial(np.clip, a_min=-10, a_max=10)
    clip_action: bool = True
    normalize_reward: bool = True
    normalize_reward_gamma: float = 0.99
    transform_reward: Optional[Any] = partial(np.clip, a_min=-10, a_max=10)

@dataclass
class PPODMCEnvConfig(DMCEnvConfig):
    num_envs: int = 1

    normalize_observation: bool = True
    transform_observation: Optional[Any] = partial(np.clip, a_min=-10, a_max=10)
    clip_action: bool = True
    normalize_reward: bool = True
    normalize_reward_gamma: float = 0.99
    transform_reward: Optional[Any] = partial(np.clip, a_min=-10, a_max=10)

@dataclass
class SimbaPPOGymMujocoEnvConfig(GymMujocoEnvConfig):
    num_envs: int = 1

    rescale_action: bool = True
    normalize_reward: bool = True
    normalize_reward_gamma: float = 0.99
    transform_reward: Optional[Any] = partial(np.clip, a_min=-10, a_max=10)

@dataclass
class SimbaPPODMCEnvConfig(DMCEnvConfig):
    num_envs: int = 1

    action_repeat_wrapper: bool = True
    action_repeat: int = 2
    rescale_action: bool = True
