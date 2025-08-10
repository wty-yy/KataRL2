import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
import tyro
from dataclasses import dataclass
from katarl2.agents import SAC, SACConfig
from katarl2.envs.env_cfg import EnvConfig
from katarl2.common.logger import LogConfig, get_tensorboard_writer
from katarl2.envs.env_maker import make_envs
from katarl2.common import path_manager
import numpy as np

@dataclass
class Args:
    agent: SACConfig
    env: EnvConfig
    logger: LogConfig
    total_timesteps: int = int(1e6)

if __name__ == '__main__':
    args: Args = tyro.cli(Args)
    path_manager.build_path_logs(
        agent_name='sac',
        algo_name=SAC.get_algo_name(args.agent),
        env_cfg=args.env,
        seeds=(args.agent.seed, args.env.seed)
    )
    envs = make_envs(args.env)
    logger = get_tensorboard_writer(args.logger, args)
    agent = SAC(args.agent, envs, logger)
    agent.learn(total_timesteps=args.total_timesteps)
    # print(type(envs.single_observation_space))
    # print(envs.single_observation_space.low)
    # model.save('sac_humanoid_walk')

    # for i in range(100):
    #     action = envs.action_space.sample()
    #     obs, rew, terminal, truncated = envs.step(action)
    #     print(f"{i=}")
    #     print(obs.shape, rew, terminal, truncated)
    #     print(obs.round(2))
