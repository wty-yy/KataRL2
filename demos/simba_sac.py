"""
SimbaSAC (from simba)
启动脚本: bash ./benchmarks/simba_sac_run_experiments.py
查看可用参数: python ./demos/simba_sac.py --help
单独启动训练 (子命令选择 {env:gym, env:dmc}):
python ./demos/simba_sac.py env:gym --env.env-name Hopper-v4 --agent.num-env-steps 100000 --agent.verbose 2 --debug
python ./demos/simba_sac.py env:dmc --env.env-name walker-walk --agent.num-env-steps 100000 --agent.verbose 2 --agent.device cuda:1 --debug
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

# 如果不设置, 在服务器上就会炸CPU (某些环境上, Humanoid-v4, dog-walk)
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import tyro
from typing import Union, Annotated
from dataclasses import dataclass
from katarl2.agents import SimbaSAC, SimbaSACConfig
from katarl2.envs import DMCEnvConfig, GymMujocoEnvConfig
from katarl2.common.logger import LogConfig, get_tensorboard_writer
from katarl2.envs.env_maker import make_envs
from katarl2.common import path_manager
from katarl2.common.video_process import cvt_to_gif
from pprint import pprint

@dataclass
class SimbaDMCEnvConfig(DMCEnvConfig):
    action_repeat_wrapper: bool = True
    action_repeat: int = 2
    rescale_action: bool = True

@dataclass
class SimbaMujocoEnvConfig(GymMujocoEnvConfig):
    action_repeat_wrapper: bool = True
    action_repeat: int = 2
    rescale_action: bool = True

@dataclass
class Args:
    agent: SimbaSACConfig
    env: Union[
        Annotated[SimbaDMCEnvConfig, tyro.conf.subcommand("dmc")],
        Annotated[SimbaMujocoEnvConfig, tyro.conf.subcommand("gym")],
    ]
    logger: LogConfig
    debug: bool = False

if __name__ == '__main__':
    """ Preprocess """
    args: Args = tyro.cli(Args, config=(tyro.conf.ConsolidateSubcommandArgs,))
    if args.env.env_type == 'gymnasium' and args.env.reward_scale == 1.0:
        print("[INFO] Gymnasium mujoco-py envs reward are not normalized, set reward_scale to 0.1")
        args.env.reward_scale = 0.1
        print("[INFO] Gymnasium mujoco-py envs are episodic which have terminations, use_cdq is set to True")
        args.agent.use_cdq = True

    path_manager.build_path_logs(args.agent, args.env, args.debug)
    envs, eval_envs = make_envs(args.env)
    logger = get_tensorboard_writer(args.logger, args)

    """ Train """
    print("[INFO] Start Training, with args:")
    pprint(args)
    agent = SimbaSAC(cfg=args.agent, envs=envs, eval_envs=eval_envs, env_cfg=args.env, logger=logger)
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
