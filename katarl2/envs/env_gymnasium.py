import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

from dataclasses import dataclass
import typing
from typing import Literal, Union
from katarl2.envs import BaseEnvConfig, AtariEnvConfig
from katarl2.common import path_manager
from gymnasium.core import ObsType
from typing import Any

@dataclass
class GymAtariEnvConfig(AtariEnvConfig):
    max_episode_env_steps: int = 108000
    action_repeat: int = 4  # max_and_skip
    atari_wrappers: bool = True
    env_type: Literal['gymnasium'] = 'gymnasium'
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
class GymMujocoEnvConfig(BaseEnvConfig):
    max_episode_env_steps: int = 1000
    env_type: Literal['gymnasium'] = 'gymnasium'
    env_name: Literal[
        'Ant-v4',
        'Ant-v5',
        'HalfCheetah-v4',
        'HalfCheetah-v5',
        'Hopper-v4',
        'Hopper-v5',
        'HumanoidStandup-v4',
        'HumanoidStandup-v5',
        'Humanoid-v4',
        'Humanoid-v5',
        'InvertedPendulum-v4',
        'InvertedPendulum-v5',
        'Pusher-v5',
        'Walker2d-v4',
        'Walker2d-v5',
    ] = 'Hopper-v4'

GymEnvConfig = Union[GymAtariEnvConfig, GymMujocoEnvConfig]

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

def make_gymnasium_env_from_cfg(cfg: GymEnvConfig):
    gym_make_kwargs = {}
    if cfg.env_name in typing.get_args(GymAtariEnvConfig.__annotations__['env_name']):
        cfg.env_name = 'ALE/' + cfg.env_name  # ALE
        # Reference https://ale.farama.org/environments/
        gym_make_kwargs.update({
            'frameskip': 1,                   # NoFrameSkip
            'repeat_action_probability': 0.25,  # Machado et al. 2017 (Revisitng ALE: Eval protocals) p. 12
        })

    if cfg.capture_video:
        env = gym.make(cfg.env_name, render_mode='rgb_array', **gym_make_kwargs)
    else:
        env = gym.make(cfg.env_name, **gym_make_kwargs)
    env = SeedWrapper(env, cfg.seed)
    env.action_space.seed(cfg.seed)
    return env
