"""
python ./demos/customed/demo1_simba_sac_train.py --help
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

# 如果不设置, 在服务器上就会炸CPU (某些环境上, Humanoid-v4, dog-walk)
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import tyro
import numpy as np
import gymnasium as gym
from typing import Literal
from dataclasses import dataclass
from katarl2.agents import SimbaSAC, SimbaSACConfig
from katarl2.envs import BaseEnvConfig
from katarl2.common.logger import LogConfig, get_tensorboard_writer
from katarl2.envs.env_maker import make_customed_envs
from katarl2.common import path_manager
from pprint import pprint
from demos.customed.demo1_target_tracking_env import TargetTrackingEnv

# 配置自定义环境的日志名称及最大步数, 前三个必须自定义
@dataclass
class CustomEnvConfig(BaseEnvConfig):
    env_type: str = 'customed'
    env_name: str = 'TargetTrackingEnv-v3'
    max_episode_env_steps: int = 1000
    num_envs: int = 1  # 训练环境数量, main中自定义实现
    num_eval_envs: int = 4  # 评估环境数量, main中自定义实现

    rescale_action: bool = True  # 将动作空间缩放到(-1, 1)

    """ 自定义环境的参数 """
    reward_mode: Literal['sparse', 'continuous'] = 'continuous'  # 奖励模式
    env_ndim: int = 2  # 环境状态维度
    n_targets: int = 1  # 目标数量

total_env_steps = 1000000
@dataclass
class CustomSimbaSACConfig(SimbaSACConfig):
    device: str = 'cuda:2'  # 设备
    total_env_steps: int = total_env_steps  # 总的环境交互步数
    
    policy_hidden_dim: int = 64
    q_hidden_dim: int = 128
    use_cdq: bool = True
    # policy_lr: float = 1e-3
    # q_lr: float = 1e-3
    log_per_interaction_step: int = total_env_steps // 100
    eval_per_interaction_step: int = total_env_steps // 20
    save_per_interaction_step: int = total_env_steps // 5
    num_eval_episodes: int = 8  # 评估回合数, 如果有eval_envs
    verbose: int = 1  # 日志打印等级, 0无, 1简略, 2详细

@dataclass
class Args:
    agent: CustomSimbaSACConfig
    env: CustomEnvConfig
    logger: LogConfig
    debug: bool = False

if __name__ == '__main__':
    """ Preprocess """
    args: Args = tyro.cli(Args, config=(tyro.conf.ConsolidateSubcommandArgs,))
    path_manager.build_path_logs(args.agent, args.env, args.debug)
    env_kwargs = {
        "reward_mode": args.env.reward_mode,
        "n_targets": args.env.n_targets,
        "env_ndim": args.env.env_ndim
    }
    make_env = lambda: TargetTrackingEnv(**env_kwargs)
    envs, eval_envs = make_customed_envs(make_env, args.env, create_train=True, create_eval=True)
    logger = get_tensorboard_writer(args.logger, args)

    """ Train """
    pprint(args)
    agent = SimbaSAC(cfg=args.agent, envs=envs, eval_envs=eval_envs, env_cfg=args.env, logger=logger)
    print("[INFO] Start Training")
    agent.learn()
    path_ckpt = agent.save()
    del agent
    envs.close()
    eval_envs.close()
    print("[INFO] Finish Training.")

    """ Eval """
    print("[INFO] Start Evaluation.")
    agent = SimbaSAC.load(path_ckpt, args.agent.device)
    args.env.num_eval_envs = 1
    args.env.capture_video = True
    agent.cfg.num_eval_episodes = 1
    make_env = lambda: TargetTrackingEnv(render_mode='rgb_array', **env_kwargs)
    _, eval_envs = make_customed_envs(make_env, args.env, create_train=False, create_eval=True)
    agent.eval_envs = eval_envs
    agent.eval()
    eval_envs.close()
