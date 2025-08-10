from typing import Literal
from dataclasses import dataclass

@dataclass
class EnvConfig:
    # Environment type
    env_type: Literal['gymnasium', 'dmc'] = 'gymnasium'
    # Environment name
    env_name: Literal[
        # gymnasium
        'Hopper-v4', 'Ant-v4', 'HalfCheetah-v4', 'HumanoidStandup-v4', 'Humanoid-v4',
        # dmc
        'humanoid-walk',
    ] = 'Hopper-v4'
    # Number of parallel environments
    env_num: int = 1
    # Whether to capture video (checkout `PATH_LOGS/video` folder)
    capture_video: bool = False
    # Random seed for environment
    seed: int = 42

if __name__ == '__main__':
    import tyro
    args: EnvConfig = tyro.cli(EnvConfig)
    print(args)
