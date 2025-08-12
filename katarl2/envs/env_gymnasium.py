import gymnasium as gym
from dataclasses import dataclass
from typing import Literal
from katarl2.envs.env_cfg import EnvConfig
from katarl2.common import path_manager
from gymnasium.core import ObsType
from typing import Any

MUJOCO_PY_NAME = [
    'Hopper-v4', 'Ant-v4', 'HalfCheetah-v4', 'HumanoidStandup-v4', 'Humanoid-v4',
]

ENV_NAME = [
    *MUJOCO_PY_NAME
]

@dataclass
class GymEnvConfig(EnvConfig):
    env_type: Literal['gymnasium'] = 'gymnasium'
    env_name: Literal[
        'Hopper-v4', 'Ant-v4', 'HalfCheetah-v4', 'HumanoidStandup-v4', 'Humanoid-v4'
    ] = 'Hopper-v4'

class SeedWrapper(gym.Wrapper):
    """ 自动将随机种子填入reset中 """
    def __init__(self, env, seed):
        super().__init__(env)
        self.first_reset = True
        self.seed = seed
    
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        if self.first_reset:
            obs, info = super().reset(seed=self.seed, options=options)
        else:
            obs, info = super().reset(seed=seed, options=options)
        self.first_reset = False
        return obs, info

def make_gymnasium_env_fn(cfg: EnvConfig):
    if cfg.env_name not in ENV_NAME:
        raise ValueError(f"Unsupported environment name: {cfg.env_name}. Supported names: {ENV_NAME}")
    if cfg.capture_video:
        PATH_VIDEOS = path_manager.PATH_LOGS / 'videos'
        # PATH_VIDEOS.mkdir(parents=True, exist_ok=True)

    def thunk():
        if cfg.capture_video:
            env = gym.make(cfg.env_name, render_mode='rgb_array')
            env = gym.wrappers.RecordVideo(env, str(PATH_VIDEOS))
        else:
            env = gym.make(cfg.env_name)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = SeedWrapper(env, cfg.seed)
        env.action_space.seed(cfg.seed)
        return env

    return thunk

def update_gymnasium_env_config(cfg: EnvConfig):
    if cfg.env_name in MUJOCO_PY_NAME:
        cfg.max_episode_steps = 1000
        cfg.action_repeat = 1
    else:
        raise Exception(f"[ERROR] Unsupported environment name: {cfg.env_name} to update env_config.")
