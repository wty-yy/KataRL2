import gymnasium as gym
from gymnasium.vector.vector_env import AutoresetMode
from copy import deepcopy
from katarl2.envs.common.env_cfg import EnvConfig
from katarl2.envs.env_gymnasium import make_gymnasium_env_from_cfg
from katarl2.envs.env_dmc import make_dmc_env_from_cfg
from gymnasium.wrappers import (
    TimeLimit, RescaleAction, TransformReward, RecordEpisodeStatistics,
    ClipAction, FlattenObservation,
    TransformObservation,
)
from katarl2.envs.common.running_mean_std import RunningMeanStd
from katarl2.envs.wrappers import (
    NormalizeObservation,
    NormalizeReward,
    RepeatAction, apply_atari_wrappers
)
from katarl2.common import path_manager

def make_envs(cfg: EnvConfig) -> tuple[gym.vector.SyncVectorEnv, gym.vector.SyncVectorEnv]:
    # Update logs path
    if path_manager._PATH_LOGS:
        cfg.path_logs = path_manager.PATH_LOGS

    # Update environment function
    if cfg.env_type == 'gymnasium':
        env_fn = make_gymnasium_env_from_cfg
    elif cfg.env_type == 'dmc':
        env_fn = make_dmc_env_from_cfg
    else:
        raise ValueError(f"Unsupported environment type: {cfg.env_type}.")

    # Environment wrapper function
    def env_wrapper_fn(cfg_i: EnvConfig, train: bool):
        def thunk():
            env = env_fn(cfg_i)
            if cfg_i.max_episode_env_steps is not None:
                env = TimeLimit(env, cfg_i.max_episode_env_steps)

            env = RecordEpisodeStatistics(env)  # TimeLimit之后, RewardScale之前记录

            """ Observation """
            if cfg_i.flatten_observation and isinstance(env.observation_space, gym.spaces.Dict):
                env = FlattenObservation(env)
            if cfg_i.normalize_observation:
                if cfg.rms_observation is None:
                    cfg.rms_observation = RunningMeanStd(shape=env.observation_space.shape)
                env = NormalizeObservation(env, rms=cfg.rms_observation, update_rms=train)
            if cfg_i.transform_observation is not None:
                env = TransformObservation(env, cfg_i.transform_observation, env.observation_space)

            """ Action """
            if cfg_i.clip_action:
                env = ClipAction(env)
            if cfg_i.action_repeat > 1:
                if cfg_i.action_repeat_wrapper:
                    env = RepeatAction(env, cfg_i.action_repeat)
                    if cfg_i.atari_wrappers:
                        print(f"[WARNING] Do you sure open `atari_wrappers` both `action_repeat={cfg_i.action_repeat}`?")
                if not cfg_i.action_repeat_wrapper and not cfg_i.atari_wrappers:
                    print(f"[WARNING] Useless `action_repeat={cfg_i.action_repeat}`, do you forget to toggle `action_repeat_wrapper` or `atari_wrappers`?")
            if cfg_i.rescale_action is not None and cfg_i.rescale_action:
                env = RescaleAction(env, -1.0, 1.0)

            """ Reward """
            if cfg_i.normalize_reward:
                if cfg.rms_return is None:
                    cfg.rms_return = RunningMeanStd(shape=())
                env = NormalizeReward(env, gamma=cfg_i.normalize_reward_gamma, rms=cfg.rms_return, update_rms=train)
            if cfg_i.transform_reward is not None:
                env = TransformReward(env, cfg_i.transform_reward)
            if cfg_i.reward_scale != 1.0:
                env = TransformReward(env, lambda r: r * cfg_i.reward_scale)
            """ Others """
            if cfg_i.atari_wrappers:
                env = apply_atari_wrappers(env, max_and_skip=cfg_i.action_repeat)
            return env
        return thunk

    def make_venv(num_envs, train: bool):
        envs_list = []
        for i in range(num_envs):
            cfg_i = deepcopy(cfg)
            cfg_i.capture_video = False if train or i != 0 else cfg.capture_video
            cfg_i.seed += i
            envs_list.append(env_wrapper_fn(cfg_i, train))
        envs = gym.vector.SyncVectorEnv(envs_list, autoreset_mode=AutoresetMode.SAME_STEP)
        return envs
    
    train_envs = make_venv(cfg.num_envs, train=True)
    eval_envs = make_venv(cfg.num_eval_envs, train=False)

    return train_envs, eval_envs
    