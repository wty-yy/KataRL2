"""
SimbaSAC (from simba)
启动脚本请用: bash ./benchmarks/simba_sac_run_experiments.py
查看可用参数: python ./demos/simba_sac.py --help
单独启动训练:
python ./demos/simba_sac.py --env.env-type gymnasium --env.env-name Hopper-v4 --total-timesteps 100000 --agent.verbose 2 
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
import tyro
from dataclasses import dataclass
from katarl2.agents import SimbaSAC, SimbaSACConfig
from katarl2.envs.env_cfg import EnvConfig
from katarl2.common.logger import LogConfig, get_tensorboard_writer
from katarl2.envs.env_maker import make_envs
from katarl2.common import path_manager
from katarl2.common.video_process import cvt_to_gif
from pprint import pprint

@dataclass
class Args:
    agent: SimbaSACConfig
    env: EnvConfig
    logger: LogConfig
    total_timesteps: int = int(1e6)
    debug: bool = False

if __name__ == '__main__':
    args: Args = tyro.cli(Args)
    path_manager.build_path_logs(args.agent, args.env, args.debug)
    envs = make_envs(args.env)
    logger = get_tensorboard_writer(args.logger, args)

    # 将环境的必要参数保存到agent配置中, 便于模型加载时使用
    args.agent.num_envs = args.env.env_num
    args.agent.obs_space = envs.single_observation_space
    args.agent.act_space = envs.single_action_space

    """ Train """
    print("[INFO] Start Training, with args:")
    pprint(args)
    agent = SimbaSAC(cfg=args.agent, envs=envs, env_cfg=args.env, logger=logger)
    agent.learn(total_timesteps=args.total_timesteps)
    path_ckpt = agent.save()
    del agent
    print("[INFO] Finish Training.")

    """ Eval """
    print("[INFO] Start Evaluation.")
    agent = SimbaSAC.load(path_ckpt, args.agent.device)
    args.env.env_num = 1
    args.env.capture_video = True
    envs = make_envs(args.env)

    obs = envs.reset()[0]
    for i in range(int(1e5)):
        action = agent.predict(obs)
        obs, rewards, terminations, truncations, infos = envs.step(action)
        if terminations.any() or truncations.any():
            envs.step(envs.action_space.sample())  # 再执行一步以保存视频
            break
    
    """ Log Video """
    if args.logger.use_swanlab:
        path_videos = path_manager.PATH_LOGS / 'videos'
        import swanlab
        for path in path_videos.glob("*.mp4"):
            path = cvt_to_gif(path)
            swanlab.log({"videos": swanlab.Video(str(path))})
            path.unlink()
    print("[INFO] Finish evaluating.")
