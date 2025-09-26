"""
PPO (from cleanrl)
启动脚本请用: ./benchmarks/ppo_run_experiments.py
查看可用参数: python ./demos/ppo_load.py --help
单独启动训练:
python ./demos/ppo_load.py --model-path /data1/user/wutianyang/Coding/GitHub/KataRL2/logs/ppo/simba_continuous_mlp/dog-walk__dmc/seed_0_0/20250830-173432/ckpts/sac-4880.pkl
python ./demos/ppo_load.py --model-path /data/user/wutianyang/Coding/KataRL2/logs/ppo/simba_discrete_cnn+mlp/Breakout-v5__gymnasium/seed_1_1/20250903-124100/ckpts/ppo_simba_discrete_cnn+mlp-39060.pkl
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import tyro
from pprint import pprint
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
    env_cfg = agent.env_cfg
    env_cfg.num_eval_envs = 1  # 验证一个环境
    agent.cfg.num_eval_episodes = 1  # 验证一个回合

    """ Capture Video """
    # env_cfg.capture_video = True
    env_cfg.capture_video = False
    from katarl2.common import path_manager
    path_manager._PATH_LOGS = Path("./logs/debug")  # 临时存储
    if env_cfg.env_type == 'envpool':
        env_cfg.env_type = 'gymnasium'  # Render in gymnasium suite
        env_cfg.atari_wrappers = True  # Render with atari wrappers
    envs, eval_envs = make_envs(env_cfg)  # 使用agent.env_cfg可以读取之前存储的RMS信息
    agent.eval_envs = eval_envs  # 配置验证环境

    """ Print Config """
    if agent.rms:
        print(agent.rms.get_statistics())
    pprint(agent.cfg)
    pprint(agent.env_cfg)

    """ Eval """
    agent.eval(verbose=True)

    envs.close()
    eval_envs.close()
