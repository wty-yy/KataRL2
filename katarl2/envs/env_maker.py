import gymnasium as gym
from gymnasium.vector.vector_env import AutoresetMode
from copy import deepcopy
from katarl2.envs.common.env_cfg import EnvConfig
from katarl2.envs.env_gymnasium import make_gymnasium_env_from_cfg
from katarl2.envs.env_dmc import make_dmc_env_from_cfg
from katarl2.envs.env_envpool import make_envpool_envs_from_cfg
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

    env_fn, envs_fn = None, None
    # Update environment function
    if cfg.env_type == 'gymnasium':
        env_fn = make_gymnasium_env_from_cfg
    elif cfg.env_type == 'dmc':
        env_fn = make_dmc_env_from_cfg
    elif cfg.env_type == 'envpool':
        envs_fn = make_envpool_envs_from_cfg
    else:
        raise ValueError(f"Unsupported environment type: {cfg.env_type}.")

    # Environment wrapper function
    def env_wrapper_fn(cfg_i: EnvConfig, train: bool):
        def thunk():
            env = env_fn(cfg_i)
            if cfg_i.max_episode_env_steps is not None:
                env = TimeLimit(env, cfg_i.max_episode_env_steps)

            env = RecordEpisodeStatistics(env)  # TimeLimit之后, RewardScale之前记录

            if cfg_i.capture_video:
                PATH_VIDEOS = cfg.path_logs / 'videos'
                env = gym.wrappers.RecordVideo(env, str(PATH_VIDEOS), episode_trigger=lambda x: x == 0, video_length=30*10*60, fps=60)

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
                    if getattr(cfg_i, 'atari_wrappers', False):
                        print(f"[WARNING] Do you sure open `atari_wrappers` both `action_repeat={cfg_i.action_repeat}`?")
                if not cfg_i.action_repeat_wrapper and not getattr(cfg_i, 'atari_wrappers', False):
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
            if getattr(cfg_i, 'atari_wrappers', False):
                atari_cfg = cfg_i.atari_wrapper_cfg
                env = apply_atari_wrappers(
                    env,
                    noop_max=atari_cfg.noop_max,
                    stack_num=atari_cfg.stack_num,
                    episodic_life=atari_cfg.episodic_life,
                    use_fire_reset=atari_cfg.use_fire_reset,
                    reward_clip=atari_cfg.reward_clip,
                    img_height=atari_cfg.img_height,
                    img_width=atari_cfg.img_width,
                    gray_scale=atari_cfg.gray_scale,
                    max_and_skip=atari_cfg.max_and_skip,
                )
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
    
    if envs_fn is not None:
        train_envs = envs_fn(cfg, train=True)
        eval_envs = envs_fn(cfg, train=False)
    else:
        train_envs = make_venv(cfg.num_envs, train=True)
        eval_envs = make_venv(cfg.num_eval_envs, train=False)

    return train_envs, eval_envs
    