import gymnasium as gym
from copy import deepcopy
from katarl2.envs.env_cfg import EnvConfig
from katarl2.envs.env_gymnasium import make_gymnaisum_env_fn

def make_env_fn(cfg: EnvConfig):
    """ Factory function to create an environment. """
    if cfg.env_type == 'gymnasium':
        return make_gymnaisum_env_fn(cfg)
    raise ValueError(f"Unsupported environment type: {cfg.env_type}.")

def make_envs(cfg: EnvConfig):
    envs_list = []
    for i in range(cfg.env_num):
        cfg_i = deepcopy(cfg)
        if cfg.capture_video and i != 0:
            cfg_i.capture_video = False
        cfg_i.seed += i
        envs_list.append(make_env_fn(cfg_i))
    envs = gym.vector.SyncVectorEnv(envs_list)
    return envs
    