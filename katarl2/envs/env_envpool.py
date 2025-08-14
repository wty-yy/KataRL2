import gym
from gym.spaces import Discrete as GymDiscrete
from gymnasium.spaces import Discrete
from dataclasses import dataclass
from typing import Literal
from katarl2.envs.common.env_cfg import EnvConfig
import numpy as np
from katarl2.common import path_manager
from gymnasium.core import ObsType
from typing import Any
import envpool

ATARI_NAME = [
    'Assault-v5',
    'Asterix-v5',
    'Boxing-v5',
    'Breakout-v5',
    'Phoenix-v5',
    'Pong-v5',
    'Qbert-v5',
    'Seaquest-v5',
    'UpNDown-v5',
    'WizardOfWor-v5',
]

ENV_NAME = [
    *ATARI_NAME
]

# 并行版Wrapper, clearnl/ppo_atari_envpool.py (并修改为Gymnasium>=1.0版本)
class RecordEpisodeStatistics:
    def __init__(self, envs):
        self.envs = envs
        self.num_envs = len(envs.all_env_ids)
        self.episode_returns = None
        self.episode_lengths = None

    @property
    def single_observation_space(self):
        if isinstance(self.envs.observation_space, GymDiscrete):
            return Discrete(self.envs.observation_space.n)
        return self.envs.observation_space

    @property
    def single_action_space(self):
        if isinstance(self.envs.action_space, GymDiscrete):
            return Discrete(self.envs.action_space.n)
        return self.envs.action_space

    def reset(self, **kwargs):
        observations, infos = self.envs.reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations, infos

    def step(self, action):
        observations, rewards, terminations, truncations, infos = self.envs.step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        # 当一局游戏结束是所有的生命值全部消耗完, lives=0或者terminated=True
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        episodic_info = {
            'r': self.returned_episode_returns,
            'l': self.returned_episode_lengths,
        }
        final_info = {
            'episode': episodic_info,
            '_episode': terminations | truncations
        }
        infos['episodic_info'] = episodic_info  # 在强制中断游戏也可以获取当前episode的r,l
        if infos['terminated'].sum():
            infos['final_info'] = final_info
        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )
    
    def close(self):
        self.envs.close()

@dataclass
class EnvpoolConfig(EnvConfig):
    env_type: Literal['envpool'] = 'envpool'
    env_name: Literal[
        'Assault-v5',
        'Asterix-v5',
        'Boxing-v5',
        'Breakout-v5',
        'Phoenix-v5',
        'Pong-v5',
        'Qbert-v5',
        'Seaquest-v5',
        'UpNDown-v5',
        'WizardOfWor-v5',
    ] = 'Breakout-v5'

def make_envpool_envs_from_cfg(cfg: EnvConfig, train: bool):
    """ envpool直接创建并行环境 """
    if cfg.env_name not in ENV_NAME:
        raise ValueError(f"Unsupported environment name: {cfg.env_name}. Supported names: {ENV_NAME}")
    if cfg.env_name in ATARI_NAME:
        # 默认支持 ClipReward, EpisodicLife, MaxAndSkip=4, FireReset
        num_envs = cfg.num_envs if train else cfg.num_eval_envs
        envs = envpool.make(
            task_id=cfg.env_name,
            env_type='gym',
            num_envs=num_envs,
            episodic_life=True,
            reward_clip=True,
            seed=cfg.seed,
        )
    else:
        raise ValueError(f"Unsupported environment name: {cfg.env_name}. Supported names: {ATARI_NAME}")
    envs = RecordEpisodeStatistics(envs)
    return envs
