from pathlib import Path
import gymnasium as gym
from typing import Literal, Optional, Annotated, Any, Union
from dataclasses import dataclass, field
from katarl2.envs.common.running_mean_std import RunningMeanStd

@dataclass
class AtariWrapperConfig:
    # Maximum number of no-op steps for random initialization 
    noop_max: int = 30
    # Number of frames to stack 
    stack_num: int = 4
    # Use EpisodicLifeWrapper 
    episodic_life: bool = False
    # Execute FIRE action on reset 
    use_fire_reset: bool = True
    # Clip reward to {-1, 0, 1} 
    reward_clip: bool = True
    # Observation image height 
    img_height: int = 84
    # Observation image width 
    img_width: int = 84
    # Convert image to grayscale 
    gray_scale: bool = True
    # Number of repeated actions per step, and maximum last 2 frames
    max_and_skip: int = 4

@dataclass
class BaseEnvConfig:
    # Environment type
    env_type: str
    # Environment name
    env_name: str
    # Max episode environment steps
    max_episode_env_steps: int
    # Number of parallel environments (Train)
    num_envs: int = 1
    # Number of parallel environments (Evaluate)
    num_eval_envs: int = 4
    # Whether to capture video (checkout `PATH_LOGS/video` folder)
    capture_video: bool = False
    # Random seed for environment
    seed: int = 42

    """ Wrappers Parameters (Update by Algo) """
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
    # Action repeat wrapper
    action_repeat_wrapper: bool = False
    # Action repeat, env_step = interaction_step * action_repeat (action_repeat used in atari is max_and_skip)
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

    """ Logger (Update by env_maker) """
    path_logs: Optional[Path] = None

@dataclass
class AtariEnvConfig(BaseEnvConfig):
    # Atari Wrappers: NoopReset, MaxAndSkip, EpisodicLife, FireReset, ClipReward, ResizeObservation, GrayScaleObservation, FrameStack
    atari_wrappers: bool = True
    atari_wrapper_cfg: AtariWrapperConfig = field(default_factory=AtariWrapperConfig)

EnvConfig = Union[BaseEnvConfig, AtariEnvConfig]

def get_env_name(cfg: BaseEnvConfig) -> str:
    """ eg: 'Hoop-v4__gymnasium' """
    return f"{cfg.env_name}__{cfg.env_type}"

if __name__ == '__main__':
    import tyro
    args: BaseEnvConfig = tyro.cli(BaseEnvConfig)
    print(args)
    # cfg = EnvConfig(env_type=None, env_name=None)
    # cfg.obs_space = 123
    # cfg.act_space = 321
    # print(cfg)
    # print(cfg.obs_space, cfg.act_space)
