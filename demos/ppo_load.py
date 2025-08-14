"""
PPO (from cleanrl)
启动脚本请用: bash ./benchmarks/ppo_run_experiments.py
查看可用参数: python ./demos/ppo.py --help
单独启动训练:
python ./demos/ppo_load.py --env-type envpool --env-name Breakout-v5
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import tyro
from dataclasses import dataclass
from katarl2.agents import PPO, PPOConfig
from katarl2.envs.common.env_cfg import EnvConfig
from katarl2.common.logger import LogConfig, get_tensorboard_writer
from katarl2.envs.env_maker import make_envs
from katarl2.common import path_manager
from katarl2.common.video_process import cvt_to_gif
from pprint import pprint

@dataclass
class PPOEnvConfig(EnvConfig):
    num_envs: int = 8
    num_eval_envs: int = 12

if __name__ == '__main__':
    """ Preprocess """
    env_cfg: PPOEnvConfig = tyro.cli(PPOEnvConfig)
    envs, eval_envs = make_envs(env_cfg)

    """ Eval """
    print("[INFO] Start Evaluation.")
    agent = PPO.load("/data/user/wutianyang/Coding/KataRL2/logs/ppo/basic_discrete_cnn+mlp/Breakout-v5__envpool/seed_0_0/20250814-011056/ckpts/sac-39060.pkl", 'cuda:3')
    agent.eval_envs = eval_envs
    agent.eval()
    print("[INFO] Finish evaluating.")
