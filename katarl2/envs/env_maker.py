import gymnasium as gym
from gymnasium.vector.vector_env import AutoresetMode
from copy import deepcopy
from katarl2.envs.common.env_cfg import EnvConfig
from katarl2.envs.env_gymnasium import make_gymnasium_env_from_cfg
from katarl2.envs.env_dmc import make_dmc_env_from_cfg
from katarl2.envs.env_envpool import make_envpool_envs_from_cfg
from gymnasium.wrappers import TimeLimit, RescaleAction, TransformReward, RecordEpisodeStatistics
from katarl2.envs.wrappers import RepeatAction

def make_envs(cfg: EnvConfig) -> tuple[gym.vector.SyncVectorEnv, gym.vector.SyncVectorEnv]:
    if cfg.env_type == 'gymnasium':
        env_fn = make_gymnasium_env_from_cfg
    elif cfg.env_type == 'dmc':
        env_fn = make_dmc_env_from_cfg
    elif cfg.env_type == 'envpool':
        envs_fn = make_envpool_envs_from_cfg
    else:
        raise ValueError(f"Unsupported environment type: {cfg.env_type}.")
    
    def env_wrapper_fn(cfg_i: EnvConfig):
        def thunk():
            env = env_fn(cfg_i)
            env = RecordEpisodeStatistics(env)  # 优先在RewardScale之前记录
            if cfg_i.max_episode_steps is not None:
                env = TimeLimit(env, cfg_i.max_episode_steps)
            if cfg_i.action_repeat is not None and cfg_i.action_repeat > 1:
                env = RepeatAction(env, cfg_i.action_repeat)
            if cfg_i.rescale_action is not None and cfg_i.rescale_action:
                env = RescaleAction(env, -1.0, 1.0)
            if cfg_i.reward_scale != 1.0:
                env = TransformReward(env, lambda r: r * cfg_i.reward_scale)
            return env
        return thunk

    def make_venv(num_envs, capture_video):
        envs_list = []
        for i in range(num_envs):
            cfg_i = deepcopy(cfg)
            cfg_i.capture_video = capture_video
            if capture_video and i != 0:
                cfg_i.capture_video = False
            cfg_i.seed += i
            envs_list.append(env_wrapper_fn(cfg_i))
        envs = gym.vector.SyncVectorEnv(envs_list, autoreset_mode=AutoresetMode.SAME_STEP)
        return envs
    
    if cfg.env_type == 'envpool':
        train_envs = envs_fn(cfg, train=True)
        eval_envs = envs_fn(cfg, train=False)
    else:
        train_envs = make_venv(cfg.num_envs, False)
        eval_envs = make_venv(cfg.num_eval_envs, cfg.capture_video)

    return train_envs, eval_envs
    