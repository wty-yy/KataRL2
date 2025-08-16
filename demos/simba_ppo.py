"""
SimbaPPO (from cleanrl+simba)
启动脚本请用: bash ./benchmarks/simba_ppo_run_experiments.py
查看可用参数: python ./demos/simba_ppo.py --help
单独启动训练:
python ./demos/simba_ppo.py --env.env-type envpool --env.env-name Breakout-v5 --agent.num-env-steps 100000 --agent.verbose 2 --debug
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import tyro
from dataclasses import dataclass
from katarl2.agents import SimbaPPO, SimbaPPOConfig
from katarl2.envs.common.env_cfg import EnvConfig
from katarl2.common.logger import LogConfig, get_tensorboard_writer
from katarl2.envs.env_maker import make_envs
from katarl2.common import path_manager
from katarl2.common.video_process import cvt_to_gif
from pprint import pprint

@dataclass
class SimbaPPOEnvConfig(EnvConfig):
    num_envs: int = 8
    num_eval_envs: int = 4

@dataclass
class Args:
    agent: SimbaPPOConfig
    env: SimbaPPOEnvConfig
    logger: LogConfig
    debug: bool = False

if __name__ == '__main__':
    """ Preprocess """
    args: Args = tyro.cli(Args)
    path_manager.build_path_logs(args.agent, args.env, args.debug)
    envs, eval_envs = make_envs(args.env)
    logger = get_tensorboard_writer(args.logger, args)

    """ Train """
    print("[INFO] Start Training, with args:")
    pprint(args)
    agent = SimbaPPO(cfg=args.agent, envs=envs, eval_envs=eval_envs, env_cfg=args.env, logger=logger)
    agent.learn()
    path_ckpt = agent.save()
    del agent
    envs.close()
    eval_envs.close()
    print("[INFO] Finish Training.")

    """ Eval """
    print("[INFO] Start Evaluation.")
    agent = SimbaPPO.load(path_ckpt, args.agent.device)
    args.env.num_envs = 1
    args.env.capture_video = True
    agent.cfg.num_eval_episodes = 1
    _, eval_envs = make_envs(args.env)
    agent.eval_envs = eval_envs
    agent.eval()

    """ Log Video """
    if args.logger.use_swanlab:
        path_videos = path_manager.PATH_LOGS / 'videos'
        import swanlab
        for path in path_videos.glob("*.mp4"):
            path = cvt_to_gif(path)
            swanlab.log({"videos": swanlab.Video(str(path))})
            path.unlink()
    print("[INFO] Finish evaluating.")
    eval_envs.close()
