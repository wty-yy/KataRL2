from pathlib import Path
import gymnasium as gym
from typing import Literal, Optional, Annotated, Any
from dataclasses import dataclass

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
    # Action repeat
    action_repeat: int = 1
    # Rescale action space to (-1, 1) (optional)
    rescale_action: Optional[bool] = None
    # Reward scale factor
    reward_scale: float = 1.0
    """ Atari Wrappers """
    # Max and Skip
    max_and_skip: int = 4
    # NoopReset, MaxAndSkip, EpisodicLife, FireReset, ClipReward, ResizeObservation, GrayScaleObservation, FrameStack
    atari_wrappers: bool = False
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
