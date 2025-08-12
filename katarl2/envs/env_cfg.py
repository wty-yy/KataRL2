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
    ]
    # Number of parallel environments (Train)
    env_num: int = 1
    # Number of parallel environments (Evaluate)
    eval_env_num: int = 1
    # Whether to capture video (checkout `PATH_LOGS/video` folder)
    capture_video: bool = False
    # Random seed for environment
    seed: int = 42
    """ Below params update by algo """
    # Max episode steps (optional)
    max_episode_steps: Optional[int] = None
    # Action repeat
    action_repeat: int = 1
    # Rescale action space to (-1, 1) (optional)
    rescale_action: Optional[bool] = None

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
