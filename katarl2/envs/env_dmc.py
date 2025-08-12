""" From simba/scale_rl/envs/dmc.py """
import gymnasium as gym
from dm_control import suite
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation
from shimmy import DmControlCompatibilityV0 as DmControltoGymnasium

from typing import Literal
from dataclasses import dataclass
from katarl2.common import path_manager
from katarl2.envs.env_cfg import EnvConfig

# 20 tasks
DMC_EASY_MEDIUM = [
    "acrobot-swingup",
    "cartpole-balance",
    "cartpole-balance_sparse",
    "cartpole-swingup",
    "cartpole-swingup_sparse",
    "cheetah-run",
    "finger-spin",
    "finger-turn_easy",
    "finger-turn_hard",
    "fish-swim",
    "hopper-hop",
    "hopper-stand",
    "pendulum-swingup",
    "quadruped-walk",
    "quadruped-run",
    "reacher-easy",
    "reacher-hard",
    "walker-stand",
    "walker-walk",
    "walker-run",
]

# 8 tasks (sparse reward)
DMC_SPARSE = [
    "cartpole-balance_sparse",
    "cartpole-swingup_sparse",
    "ball_in_cup-catch",
    "finger-spin",
    "finger-turn_easy",
    "finger-turn_hard",
    "reacher-easy",
    "reacher-hard",
]

# 7 tasks
DMC_HARD = [
    "humanoid-stand",
    "humanoid-walk",
    "humanoid-run",
    "dog-stand",
    "dog-walk",
    "dog-run",
    "dog-trot",
]

@dataclass
class DMCEnvConfig(EnvConfig):
    env_type: Literal['dmc'] = 'dmc'
    env_name: Literal[  # Only continuous control tasks
        'acrobot-swingup',
        'cartpole-balance',
        'cartpole-balance_sparse',
        'cartpole-swingup',
        'cartpole-swingup_sparse',
        'cheetah-run',
        'finger-spin',
        'finger-turn_easy',
        'finger-turn_hard',
        'fish-swim',
        'hopper-hop',
        'hopper-stand',
        'pendulum-swingup',
        'quadruped-walk',
        'quadruped-run',
        'reacher-easy',
        'reacher-hard',
        'walker-stand',
        'walker-walk',
        'walker-run',

        "cartpole-balance_sparse",
        "cartpole-swingup_sparse",
        "ball_in_cup-catch",
        "finger-spin",
        "finger-turn_easy",
        "finger-turn_hard",
        "reacher-easy",
        "reacher-hard",

        'humanoid-stand',
        'humanoid-walk',
        'humanoid-run',
        'dog-stand',
        'dog-walk',
        'dog-run',
        'dog-trot'
    ] = 'walker-walk'


def make_dmc_env(
    env_name: str,
    seed: int,
    flatten: bool = True,
) -> gym.Env:
    domain_name, task_name = env_name.split("-")
    env = suite.load(
        domain_name=domain_name,
        task_name=task_name,
        task_kwargs={"random": seed},
    )
    env = DmControltoGymnasium(env, render_mode="rgb_array")
    if flatten and isinstance(env.observation_space, spaces.Dict):
        env = FlattenObservation(env)

    return env

def make_dmc_env_from_cfg(cfg: EnvConfig):
    if cfg.env_name not in DMC_EASY_MEDIUM + DMC_SPARSE + DMC_HARD:
        raise ValueError(f"Unsupported environment name: {cfg.env_name}. Supported names: {DMC_EASY_MEDIUM + DMC_SPARSE + DMC_HARD}")
    if cfg.capture_video:
        PATH_VIDEOS = path_manager.PATH_LOGS / 'videos'
    
    env = make_dmc_env(cfg.env_name, cfg.seed)
    if cfg.capture_video:
        env = gym.wrappers.RecordVideo(env, str(PATH_VIDEOS))
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env
