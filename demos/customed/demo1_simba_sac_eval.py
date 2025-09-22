"""
SimbaSAC (from simba)
单独启动训练 (子命令选择 {env:gym, env:dmc}):
python ./demos/customed/demo1_simba_sac.py --debug
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

# 如果不设置, 在服务器上就会炸CPU (某些环境上, Humanoid-v4, dog-walk)
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import tyro
import gymnasium as gym
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
    env_name: str = 'TargetTrackingEnv-v0'
    max_episode_env_steps: int = 1000
    num_eval_envs: int = 1  # 评估环境数量, main中自定义实现

    action_repeat: int = 2  # 动作重复次数
    action_repeat_wrapper: bool = True  # 使用动作重复包装器
    rescale_action: bool = True  # 将动作空间缩放到(-1, 1)

@dataclass
class CustomSimbaSACConfig(SimbaSACConfig):
    device: str = 'cuda:3'  # 设备

@dataclass
class Args:
    load_path: str
    agent: CustomSimbaSACConfig
    env: CustomEnvConfig
    logger: LogConfig
    debug: bool = False

if __name__ == '__main__':
    """ Preprocess """
    args: Args = tyro.cli(Args, config=(tyro.conf.ConsolidateSubcommandArgs,))
    path_manager.build_path_logs(args.agent, args.env, args.debug)
    make_env = lambda: TargetTrackingEnv(reward_mode='continuous', render_mode='human')
    logger = get_tensorboard_writer(args.logger, args)

    """ Eval """
    print("[INFO] Start Evaluation.")
    agent = SimbaSAC.load(args.load_path, args.agent.device)
    args.env.num_eval_envs = 1
    agent.cfg.num_eval_episodes = 10
    _, eval_envs = make_customed_envs(make_env, args.env, create_train=False, create_eval=True)
    agent.eval_envs = eval_envs
    agent.eval()
    eval_envs.close()
