from pathlib import Path
import gymnasium as gym
from typing import Literal, Optional, Annotated, Any
from dataclasses import dataclass
from katarl2.envs.common.running_mean_std import RunningMeanStd

@dataclass
class EnvConfig:
    # Environment type
    env_type: str
    # Environment name
    env_name: str
    # Number of parallel environments (Train)
    num_envs: int = 1
    # Number of parallel environments (Evaluate)
    num_eval_envs: int = 1
    # Whether to capture video (checkout `PATH_LOGS/video` folder)
    capture_video: bool = False
    # Random seed for environment
    seed: int = 42
    """ Wrappers Parameters (Update by Algo) """
    # Max episode steps (optional)
    max_episode_steps: Optional[int] = None
    """ Observation related """
    # Flatten observation (deal with cm_control's Dict observation space)
    flatten_observation: bool = False
    # Normalize observation by a shared Running Mean Std (Before obs transform)
    normalize_observation: bool = False
    # Running Mean Std for observation
    rms_observation: Optional[Any] = None
    # Transform function for observation
    transform_observation: Optional[Any] = None
    """ Action related """
    # Action repeat
    action_repeat: int = 1
    # Rescale action space to (-1, 1) (optional)
    rescale_action: Optional[bool] = None
    # Clip action to [-1, 1]
    clip_action: bool = False
    """ Reward related """
    # Normalize reward by a shared Running Mean Std (Before reward transform)
    normalize_reward: bool = False
    # If normalize reward is True, add gamma coefficient for exponential moving average (Default 0.99)
    normalize_reward_gamma: float = 0.99
    # Running Mean Std for return (normalize reward use return's rms, see https://github.com/openai/baselines/issues/538)
    rms_return: Optional[Any] = None
    # Transform function for reward
    transform_reward: Optional[Any] = None
    # Reward scale factor by transform: reward = reward * reward_scale
    reward_scale: float = 1.0

    """ Atari Wrappers """
    # NoopReset, MaxAndSkip, EpisodicLife, FireReset, ClipReward, ResizeObservation, GrayScaleObservation, FrameStack
    atari_wrappers: bool = False
    # If atari_wrappers is True, use max and skip frames
    max_and_skip: int = 1
    """ Logger (Update by env_maker) """
    path_logs: Optional[Path] = None

def get_env_name(cfg: EnvConfig) -> str:
    """ eg: 'Hoop-v4__gymnasium' """
    return f"{cfg.env_name}__{cfg.env_type}"

if __name__ == '__main__':
    import tyro
    args: EnvConfig = tyro.cli(EnvConfig)
    print(args)
    # cfg = EnvConfig(env_type=None, env_name=None)
    # cfg.obs_space = 123
    # cfg.act_space = 321
    # print(cfg)
    # print(cfg.obs_space, cfg.act_space)
