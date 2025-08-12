import gymnasium as gym
from typing import Literal, Optional, Annotated, Any
from dataclasses import dataclass

@dataclass
class EnvConfig:
    # Environment type
    env_type: Literal['gymnasium', 'dmc']
    # Environment name
    env_name: Literal[
        # gymnasium
        'Hopper-v4', 'Ant-v4', 'HalfCheetah-v4', 'HumanoidStandup-v4', 'Humanoid-v4',
        # dmc
        'humanoid-walk',
    ]
    # Number of parallel environments
    env_num: int = 1
    # Whether to capture video (checkout `PATH_LOGS/video` folder)
    capture_video: bool = False
    # Random seed for environment
    seed: int = 42
    """ Below params update by env """
    # Max episode steps (optional)
    max_episode_steps: Optional[int] = None
    # Action repeat (optional)
    action_repeat: Optional[int] = None

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
