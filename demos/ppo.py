"""
PPO (from cleanrl)
启动脚本: bash ./benchmarks/ppo_run_experiments.py
查看可用参数: python ./demos/ppo.py --help
单独启动训练 (子命令选择 {agent:disc, agent:cont} {env:envpool-atari, env:gym-atari, env:gym-mujoco, env:dmc, env:gym-mujoco-simba, env:dmc-simba}):
python ./demos/ppo.py agent:disc env:envpool-atari --env.env-name Pong-v5 --agent.total-env-steps 10000 --agent.verbose 2 --debug
python ./demos/ppo.py agent:disc env:gym-atari --env.env-name Pong-v5 --agent.total-env-steps 10000 --agent.verbose 2 --debug
python ./demos/ppo.py agent:disc-simba env:envpool-atari-simba --env.env-name Pong-v5 --agent.total-env-steps 10000 --agent.verbose 2 --debug
python ./demos/ppo.py agent:disc-simba env:gym-atari-simba --env.env-name Pong-v5 --agent.total-env-steps 10000 --agent.verbose 2 --debug
python ./demos/ppo.py agent:cont env:gym-mujoco --env.env-name Hopper-v4 --agent.total-env-steps 10000 --agent.verbose 2 --debug
python ./demos/ppo.py agent:cont env:dmc --env.env-name walker-walk --agent.total-env-steps 10000 --agent.verbose 2 --debug
python ./demos/ppo.py agent:cont-simba env:gym-mujoco-simba --env.env-name Hopper-v4 --agent.total-env-steps 10000 --agent.verbose 2 --debug
python ./demos/ppo.py agent:cont-simba env:dmc-simba --env.env-name walker-walk --agent.total-env-steps 10000 --agent.verbose 2 --debug
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
from katarl2.agents import PPO, PPODiscreteConfig, PPOContinuousConfig, SimbaPPOContinuousConfig, SimbaPPODiscreteConfig
from katarl2.agents.ppo.ppo_env_cfg import (
    PPOEnvpoolAtariEnvConfig, PPOGymAtariEnvConfig,
    PPODMCEnvConfig, PPOGymMujocoEnvConfig,
    SimbaPPODMCEnvConfig, SimbaPPOGymMujocoEnvConfig
)
from katarl2.common.logger import LogConfig, get_tensorboard_writer
from katarl2.envs.env_maker import make_envs
from katarl2.common import path_manager
from katarl2.common.video_process import cvt_to_gif
from pprint import pprint

@dataclass
class Args:
    agent: Union[
        # Discrete action space
        Annotated[PPODiscreteConfig, tyro.conf.subcommand('disc')],
        # Continuous action space
        Annotated[PPOContinuousConfig, tyro.conf.subcommand('cont')],
        # Discrete action space with Simba tricks
        Annotated[SimbaPPODiscreteConfig, tyro.conf.subcommand('disc-simba')],
        # Continuous action space with Simba tricks
        Annotated[SimbaPPOContinuousConfig, tyro.conf.subcommand('cont-simba')],
    ]
    env: Union[
        # Env Basic Config
        Annotated[PPODMCEnvConfig, tyro.conf.subcommand('dmc')],
        Annotated[PPOGymMujocoEnvConfig, tyro.conf.subcommand('gym-mujoco')],
        Annotated[PPOEnvpoolAtariEnvConfig, tyro.conf.subcommand('envpool-atari')],
        Annotated[PPOGymAtariEnvConfig, tyro.conf.subcommand('gym-atari')],
        # Env Simba Config
        Annotated[SimbaPPODMCEnvConfig, tyro.conf.subcommand('dmc-simba')],
        Annotated[SimbaPPOGymMujocoEnvConfig, tyro.conf.subcommand('gym-mujoco-simba')],
        Annotated[PPOEnvpoolAtariEnvConfig, tyro.conf.subcommand('envpool-atari-simba')],  # same
        Annotated[PPOGymAtariEnvConfig, tyro.conf.subcommand('gym-atari-simba')],  # same
    ]
    logger: LogConfig
    debug: bool = False

if __name__ == '__main__':
    """ Preprocess """
    args: Args = tyro.cli(Args, config=(tyro.conf.ConsolidateSubcommandArgs,))
    path_manager.build_path_logs(args.agent, args.env, args.debug)
    envs, eval_envs = make_envs(args.env)
    logger = get_tensorboard_writer(args.logger, args)

    """ Train """
    print("[INFO] Start Training, with args:")
    pprint(args)
    agent = PPO(cfg=args.agent, envs=envs, eval_envs=eval_envs, env_cfg=args.env, logger=logger)
    agent.learn()
    path_ckpt = agent.save()
    del agent
    envs.close()
    eval_envs.close()
    print("[INFO] Finish Training.")

    """ Eval """
    print("[INFO] Start Evaluation.")
    agent = PPO.load(path_ckpt, args.agent.device)
    if args.env.env_type == 'envpool':
        args.env.env_type = 'gymnasium'  # Render in gymnasium suite
        args.env.atari_wrappers = True  # Render with atari wrappers
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
