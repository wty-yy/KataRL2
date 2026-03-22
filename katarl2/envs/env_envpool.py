from gym.spaces import Discrete as GymDiscrete
from gymnasium.spaces import Box, Discrete
from dataclasses import dataclass
from typing import Literal
from katarl2.envs.common.env_cfg import AtariEnvConfig
from katarl2.envs.env_gymnasium import GymMujocoEnvConfig
from katarl2.envs.common.running_mean_std import RunningMeanStd
import numpy as np
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

MUJOCO_NAME = [
    'Ant-v4',
    'HalfCheetah-v4',
    'Hopper-v4',
    'HumanoidStandup-v4',
    'Humanoid-v4',
    'InvertedPendulum-v4',
    'Walker2d-v4',
]

ENV_NAME = [
    *ATARI_NAME,
    *MUJOCO_NAME,
]


def cast_observation(observations):
    return np.asarray(observations, dtype=np.float32)


def cast_box_space(space):
    if isinstance(space, Box) and space.dtype != np.float32:
        return Box(
            low=np.asarray(space.low, dtype=np.float32),
            high=np.asarray(space.high, dtype=np.float32),
            shape=space.shape,
            dtype=np.float32,
        )
    return space

# 并行版Wrapper, clearnl/ppo_atari_envpool.py (并修改为Gymnasium>=1.0版本)
class RecordEpisodeStatistics:
    def __init__(self, envs, action_repeat: int):
        self.envs = envs
        self.action_repeat = action_repeat
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
        return cast_box_space(self.envs.action_space)

    def reset(self, **kwargs):
        observations, infos = self.envs.reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations, infos

    def step(self, action):
        observations, rewards, terminations, truncations, infos = self.envs.step(action)
        dones = terminations | truncations
        self.episode_returns += rewards
        self.episode_lengths += self.action_repeat

        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - dones
        self.episode_lengths *= 1 - dones
        episodic_info = {
            'r': self.returned_episode_returns,
            'l': self.returned_episode_lengths,
        }
        final_info = {
            'episode': episodic_info,
            '_episode': dones
        }
        if dones.sum():
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


class EnvpoolWrapper:
    def __init__(self, envs):
        self.envs = envs

    def __getattr__(self, name):
        return getattr(self.envs, name)

    @property
    def num_envs(self):
        return self.envs.num_envs

    @property
    def single_observation_space(self):
        return self.envs.single_observation_space

    @property
    def single_action_space(self):
        return cast_box_space(self.envs.single_action_space)

    def reset(self, **kwargs):
        return self.envs.reset(**kwargs)

    def step(self, action):
        return self.envs.step(action)

    def close(self):
        self.envs.close()


class TransformObservation(EnvpoolWrapper):
    def __init__(self, envs, transform_observation):
        super().__init__(envs)
        self.transform_observation = transform_observation

    @property
    def single_observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=self.envs.single_observation_space.shape,
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        observations, infos = self.envs.reset(**kwargs)
        observations = self.transform_observation(observations)
        return cast_observation(observations), infos

    def step(self, action):
        observations, rewards, terminations, truncations, infos = self.envs.step(action)
        observations = self.transform_observation(observations)
        return cast_observation(observations), rewards, terminations, truncations, infos


class NormalizeObservation(EnvpoolWrapper):
    def __init__(self, envs, rms: RunningMeanStd, update_rms: bool):
        super().__init__(envs)
        self.obs_rms = rms
        self.update_rms = update_rms

    @property
    def single_observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=self.envs.single_observation_space.shape,
            dtype=np.float32,
        )

    def _normalize(self, observations):
        if self.update_rms:
            self.obs_rms.update(observations)
        return cast_observation(self.obs_rms.normalize(observations))

    def reset(self, **kwargs):
        observations, infos = self.envs.reset(**kwargs)
        return self._normalize(observations), infos

    def step(self, action):
        observations, rewards, terminations, truncations, infos = self.envs.step(action)
        return self._normalize(observations), rewards, terminations, truncations, infos


class ClipAction(EnvpoolWrapper):
    def __init__(self, envs):
        super().__init__(envs)
        self.low = self.single_action_space.low
        self.high = self.single_action_space.high

    def step(self, action):
        action = np.clip(action, self.low, self.high)
        return self.envs.step(action)


class RescaleAction(EnvpoolWrapper):
    def __init__(self, envs, min_action: float = -1.0, max_action: float = 1.0):
        super().__init__(envs)
        self.min_action = min_action
        self.max_action = max_action
        self.low = self.single_action_space.low
        self.high = self.single_action_space.high

    def step(self, action):
        action = self.low + (action - self.min_action) * (self.high - self.low) / (self.max_action - self.min_action)
        action = np.clip(action, self.low, self.high)
        return self.envs.step(action)


class TransformReward(EnvpoolWrapper):
    def __init__(self, envs, transform_reward):
        super().__init__(envs)
        self.transform_reward = transform_reward

    def step(self, action):
        observations, rewards, terminations, truncations, infos = self.envs.step(action)
        return observations, self.transform_reward(rewards), terminations, truncations, infos


class NormalizeReward(EnvpoolWrapper):
    def __init__(self, envs, gamma: float, rms: RunningMeanStd, update_rms: bool, epsilon: float = 1e-8):
        super().__init__(envs)
        self.return_rms = rms
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_rms = update_rms
        self.discounted_reward = np.zeros(self.num_envs, dtype=np.float32)

    def reset(self, **kwargs):
        self.discounted_reward[:] = 0.0
        return self.envs.reset(**kwargs)

    def step(self, action):
        observations, rewards, terminations, truncations, infos = self.envs.step(action)
        dones = terminations | truncations
        self.discounted_reward = self.discounted_reward * self.gamma * (1 - dones) + rewards
        if self.update_rms:
            self.return_rms.update(self.discounted_reward)
        rewards = rewards / np.sqrt(self.return_rms.var + self.epsilon)
        return observations, rewards, terminations, truncations, infos

@dataclass
class EnvpoolAtariEnvConfig(AtariEnvConfig):
    max_episode_env_steps: int = 108000
    action_repeat: int = 4  # max_and_skip
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


@dataclass
class EnvpoolMujocoEnvConfig(GymMujocoEnvConfig):
    env_type: Literal['envpool'] = 'envpool'
    env_name: Literal[
        'Ant-v4',
        'HalfCheetah-v4',
        'Hopper-v4',
        'HumanoidStandup-v4',
        'Humanoid-v4',
        'InvertedPendulum-v4',
        'Walker2d-v4',
    ] = 'Hopper-v4'

def make_envpool_envs_from_cfg(cfg: EnvpoolAtariEnvConfig | EnvpoolMujocoEnvConfig, train: bool):
    """ envpool直接创建并行环境 """
    if cfg.env_name not in ENV_NAME:
        raise ValueError(f"Unsupported environment name: {cfg.env_name}. Supported names: {ENV_NAME}")
    if cfg.env_name in ATARI_NAME:
        # 默认支持 ClipReward, EpisodicLife, MaxAndSkip=4, FireReset
        num_envs = cfg.num_envs if train else cfg.num_eval_envs
        # Reference: https://envpool.readthedocs.io/en/latest/env/atari.html
        atari_cfg = cfg.atari_wrapper_cfg
        envs = envpool.make(
            task_id=cfg.env_name,
            env_type='gymnasium',
            num_envs=num_envs,
            seed=cfg.seed,
            max_episode_steps=cfg.max_episode_env_steps//4,
            repeat_action_probability=0.25,

            # Default wrappers
            noop_max=atari_cfg.noop_max,
            stack_num=atari_cfg.stack_num,
            episodic_life=atari_cfg.episodic_life,
            use_fire_reset=atari_cfg.use_fire_reset,
            reward_clip=atari_cfg.reward_clip,
            img_height=atari_cfg.img_height,
            img_width=atari_cfg.img_width,
            gray_scale=atari_cfg.gray_scale,
            frame_skip=atari_cfg.max_and_skip,
        )
    elif cfg.env_name in MUJOCO_NAME:
        num_envs = cfg.num_envs if train else cfg.num_eval_envs
        envs = envpool.make(
            task_id=cfg.env_name,
            env_type='gymnasium',
            num_envs=num_envs,
            seed=cfg.seed,
            max_episode_steps=cfg.max_episode_env_steps,
        )
    else:
        raise ValueError(f"Unsupported environment name: {cfg.env_name}. Supported names: {ENV_NAME}")
    envs = RecordEpisodeStatistics(envs, action_repeat=cfg.action_repeat)
    if cfg.normalize_observation:
        if cfg.rms_observation is None:
            cfg.rms_observation = RunningMeanStd(shape=envs.single_observation_space.shape)
        envs = NormalizeObservation(envs, rms=cfg.rms_observation, update_rms=train)
    if cfg.transform_observation is not None:
        envs = TransformObservation(envs, cfg.transform_observation)
    if cfg.rescale_action:
        envs = RescaleAction(envs)
    elif cfg.clip_action:
        envs = ClipAction(envs)
    if cfg.normalize_reward:
        if cfg.rms_return is None:
            cfg.rms_return = RunningMeanStd(shape=())
        envs = NormalizeReward(
            envs,
            gamma=cfg.normalize_reward_gamma,
            rms=cfg.rms_return,
            update_rms=train,
        )
    if cfg.transform_reward is not None:
        envs = TransformReward(envs, cfg.transform_reward)
    return envs
