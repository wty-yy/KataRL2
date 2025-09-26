"""
BasicSAC (from cleanrl)
启动脚本请用: ./benchmarks/sac_run_experiments.py
查看可用参数: python ./demos/sac.py --help
单独启动训练:
python ./demos/sac.py --env.env-type gymnasium --env.env-name Hopper-v4 --total-timesteps 10000
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
import tyro
from dataclasses import dataclass
from katarl2.agents import SAC, SACConfig
from katarl2.envs import BaseEnvConfig
from katarl2.common.logger import LogConfig, get_tensorboard_writer
from katarl2.envs.env_maker import make_envs
from katarl2.common import path_manager
from katarl2.common.video_process import cvt_to_gif
import numpy as np
from pprint import pprint

if __name__ == '__main__':
    print("[INFO] Start Evaluation.")
    agent = SAC.load("/data/user/wutianyang/Coding/KataRL2/logs/sac/basic_continuous_mlp/Hopper-v4__gymnasium/seed_6_6/20250811-171118/ckpts/sac-994999.pkl", 'cuda')
