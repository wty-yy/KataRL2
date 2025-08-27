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
    'Alien-v5', 'Amidar-v5', 'Assault-v5', 'Asterix-v5', 'Asteroids-v5',
    'Atlantis-v5', 'BankHeist-v5', 'BattleZone-v5', 'BeamRider-v5', 'Blinky-v5',
    'Bowling-v5', 'Boxing-v5', 'Breakout-v5', 'Carnival-v5', 'Centipede-v5',
    'ChopperCommand-v5', 'CrazyClimber-v5', 'DemonAttack-v5', 'DoubleDunk-v5',
    'Enduro-v5', 'FishingDerby-v5', 'Freeway-v5', 'Frostbite-v5', 'Gopher-v5',
    'Gravitar-v5', 'Hero-v5', 'IceHockey-v5', 'Jamesbond-v5', 'JourneyEscape-v5',
    'Kangaroo-v5', 'Krull-v5', 'KungFuMaster-v5', 'MontezumaRevenge-v5',
    'MsPacman-v5', 'NameThisGame-v5', 'Phoenix-v5', 'Pitfall-v5', 'Pong-v5',
    'PrivateEye-v5', 'Qbert-v5', 'Riverraid-v5', 'RoadRunner-v5', 'Robotank-v5',
    'Seaquest-v5', 'Skiing-v5', 'Solaris-v5', 'SpaceInvaders-v5', 'StarGunner-v5',
    'Tennis-v5', 'TimePilot-v5', 'Tutankham-v5', 'UpNDown-v5', 'Venture-v5',
    'VideoPinball-v5', 'WizardOfWor-v5', 'YarsRevenge-v5', 'Zaxxon-v5'
]

ENV_NAME = [
    *ATARI_NAME
]

# 并行版Wrapper, clearnl/ppo_atari_envpool.py (并修改为Gymnasium>=1.0版本)
class RecordEpisodeStatisticsAndTimeLimit:
    def __init__(self, envs, time_limit: int):
        self.envs = envs
        self.time_limit = time_limit
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
        # 超时
        infos["terminated"] |= self.episode_lengths > self.time_limit
        truncations |= self.episode_lengths > self.time_limit

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
        # infos['episodic_info'] = episodic_info  # 在强制中断游戏也可以获取当前episode的r,l
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
class EnvpoolAtariConfig(EnvConfig):
    max_episode_steps: int = 108000
    env_type: Literal['envpool'] = 'envpool'
    env_name: Literal[
        'Alien-v5', 'Amidar-v5', 'Assault-v5', 'Asterix-v5', 'Asteroids-v5',
        'Atlantis-v5', 'BankHeist-v5', 'BattleZone-v5', 'BeamRider-v5', 'Blinky-v5',
        'Bowling-v5', 'Boxing-v5', 'Breakout-v5', 'Carnival-v5', 'Centipede-v5',
        'ChopperCommand-v5', 'CrazyClimber-v5', 'DemonAttack-v5', 'DoubleDunk-v5',
        'Enduro-v5', 'FishingDerby-v5', 'Freeway-v5', 'Frostbite-v5', 'Gopher-v5',
        'Gravitar-v5', 'Hero-v5', 'IceHockey-v5', 'Jamesbond-v5', 'JourneyEscape-v5',
        'Kangaroo-v5', 'Krull-v5', 'KungFuMaster-v5', 'MontezumaRevenge-v5',
        'MsPacman-v5', 'NameThisGame-v5', 'Phoenix-v5', 'Pitfall-v5', 'Pong-v5',
        'PrivateEye-v5', 'Qbert-v5', 'Riverraid-v5', 'RoadRunner-v5', 'Robotank-v5',
        'Seaquest-v5', 'Skiing-v5', 'Solaris-v5', 'SpaceInvaders-v5', 'StarGunner-v5',
        'Tennis-v5', 'TimePilot-v5', 'Tutankham-v5', 'UpNDown-v5', 'Venture-v5',
        'VideoPinball-v5', 'WizardOfWor-v5', 'YarsRevenge-v5', 'Zaxxon-v5'
    ] = 'Breakout-v5'

def make_envpool_envs_from_cfg(cfg: EnvConfig, train: bool):
    """ envpool直接创建并行环境 """
    if cfg.env_name not in ENV_NAME:
        raise ValueError(f"Unsupported environment name: {cfg.env_name}. Supported names: {ENV_NAME}")
    if cfg.env_name in ATARI_NAME:
        # 默认支持 ClipReward, EpisodicLife, MaxAndSkip=4, FireReset
        num_envs = cfg.num_envs if train else cfg.num_eval_envs
        # Reference: https://envpool.readthedocs.io/en/latest/env/atari.html
        envs = envpool.make(
            task_id=cfg.env_name,
            env_type='gymnasium',
            num_envs=num_envs,
            seed=cfg.seed,
            max_episode_steps=cfg.max_episode_steps,
            repeat_action_probability=0.25,

            # Default wrappers
            noop_max=30,
            stack_num=4,
            episodic_life=True,
            use_fire_reset=True,
            reward_clip=True,
            img_height=84,
            img_width=84,
            gray_scale=True,
            frame_skip=4,
        )
    else:
        raise ValueError(f"Unsupported environment name: {cfg.env_name}. Supported names: {ATARI_NAME}")
    # 这里的time_limit是经过frame_skip后的步数
    envs = RecordEpisodeStatisticsAndTimeLimit(envs, time_limit=cfg.max_episode_steps // 4)
    return envs
