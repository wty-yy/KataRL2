"""
PPO (from cleanrl)
启动脚本请用: bash ./benchmarks/ppo_run_experiments.py
查看可用参数: python ./demos/ppo_load.py --help
单独启动训练:
python ./demos/ppo_load.py --model-path /data1/user/wutianyang/Coding/GitHub/KataRL2/logs/ppo/simba_continuous_mlp/dog-walk__dmc/seed_0_0/20250830-173432/ckpts/sac-4880.pkl
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import tyro
from dataclasses import dataclass
from katarl2.agents import PPO
from katarl2.envs.env_maker import make_envs

@dataclass
class Args:
    model_path: str
    device: str = 'cuda'

if __name__ == '__main__':
    """ Parse Args """
    args: Args = tyro.cli(Args, config=(tyro.conf.ConsolidateSubcommandArgs,))

    """ Load Model """
    agent = PPO.load(args.model_path, args.device)
    print(agent.rms.get_statistics())
    env_cfg = agent.env_cfg
    env_cfg.num_eval_envs = 1  # 验证一个环境
    agent.cfg.num_eval_episodes = 1  # 验证一个回合

    """ Capture Video """
    env_cfg.capture_video = True
    from katarl2.common import path_manager
    path_manager._PATH_LOGS = Path("./logs/debug")  # 临时存储
    envs, eval_envs = make_envs(env_cfg)  # 使用agent.env_cfg可以读取之前存储的RMS信息
    agent.eval_envs = eval_envs  # 配置验证环境

    """ Eval """
    print("[INFO] Start Evaluation.")
    agent.eval(verbose=True)
    print("[INFO] Finish evaluating.")

    envs.close()
    eval_envs.close()
