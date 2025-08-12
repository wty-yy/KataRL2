"""
SimbaSAC (from simba)
启动脚本请用: bash ./benchmarks/simba_sac_run_experiments.py
查看可用参数: python ./demos/simba_sac.py --help
单独启动训练:
python ./demos/simba_sac_load.py --env.env-type gymnasium --env.env-name Humanoid-v4
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

    """ Eval """
    print("[INFO] Start Evaluation.")
    agent = SimbaSAC.load("/data/user/wutianyang/Coding/KataRL2/logs/sac/simba_continuous_mlp/Humanoid-v4__gymnasium/seed_2_2/20250811-234639/ckpts/sac-994999.pkl", args.agent.device)
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
