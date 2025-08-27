"""
PPO (from cleanrl)
启动脚本请用: bash ./benchmarks/ppo_run_experiments.py
查看可用参数: python ./demos/ppo_load.py --help
单独启动训练:
python ./demos/ppo_load.py env:gym --env.env-name Breakout-v5 --model-path /data/user/wutianyang/Coding/KataRL2/logs/ppo/basic_discrete_cnn+mlp/Breakout-v5__envpool/seed_0_0/20250822-214056/ckpts/sac-39060.pkl
python ./demos/ppo_load.py env:envpool --env.env-name Breakout-v5 --model-path /data/user/wutianyang/Coding/KataRL2/logs/ppo/basic_discrete_cnn+mlp/Breakout-v5__envpool/seed_0_0/20250822-214056/ckpts/sac-39060.pkl
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import tyro
from typing import Union, Annotated
from dataclasses import dataclass
from katarl2.agents import PPO
from katarl2.envs import EnvpoolAtariConfig, GymAtariEnvConfig
from katarl2.envs.env_maker import make_envs

@dataclass
class Args:
    model_path: str
    env: Union[
        Annotated[EnvpoolAtariConfig, tyro.conf.subcommand('envpool')],
        Annotated[GymAtariEnvConfig, tyro.conf.subcommand('gym')],
    ]
    device: str = 'cuda'

if __name__ == '__main__':
    """ Parse Args """
    args: Args = tyro.cli(Args, config=(tyro.conf.ConsolidateSubcommandArgs,))
    args.env.num_eval_envs = 1

    """ Capture Video """
    args.env.capture_video = True
    from katarl2.common import path_manager
    path_manager._PATH_LOGS = Path("./logs/debug")  # 临时存储
    envs, eval_envs = make_envs(args.env)

    """ Load Model """
    agent = PPO.load(args.model_path, args.device)
    agent.cfg.num_eval_episodes = 1  # 仅验证1个回合
    agent.eval_envs = eval_envs

    """ Eval """
    print("[INFO] Start Evaluation.")
    agent.eval(verbose=True)
    print("[INFO] Finish evaluating.")

    envs.close()
    eval_envs.close()
