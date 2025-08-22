import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import time
import tyro
import numpy as np
from katarl2.envs.env_gymnasium import GymAtariEnvConfig, GymMujocoEnvConfig
from katarl2.envs.env_maker import make_envs
from katarl2.common.path_manager import path_manager
from tqdm import tqdm

if __name__ == '__main__':
    # cfg = tyro.cli(GymMujocoEnvConfig)
    cfg = tyro.cli(GymAtariEnvConfig)
    path_manager._PATH_LOGS = Path("./logs/debug")
    # cfg.env_name = 'Ant-v4'
    cfg.capture_video = True
    cfg.num_envs = 2
    cfg.seed = 0
    _, envs = make_envs(cfg)
    envs.reset()
    # act = np.full(envs.action_space.shape, 0.5)
    act = envs.action_space.sample()
    start_time = time.time()
    total_reward = 0
    bar = tqdm(range(100))
    for i in bar:
        obs, rewards, terminations, truncations, infos = envs.step(act)
        total_reward += rewards.mean()

        real_next_obs = obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_obs"][idx]

        # if terminations.any() or truncations.any():
        #     envs.step(act)
        #     break
        # if 'lives' in infos:
        #     bar.set_description(f"{infos['lives']=}")
        # elif 'final_info' in infos:
        #     bar.set_description(f"{infos['final_info']['lives']=}")
        # if 'final_info' in infos:
        #     print(f"{infos['final_info']}")
        #     print(terminations, truncations)
        # Atari中用RecordEpisodeStatistics会自动记录5条命全部结束时的奖励, 和各种论文中的标准一致
        if 'final_info' in infos and infos['final_info']['lives'] == 0:
            print(f"{infos['final_info']['episode']}")
            print(terminations, truncations)
    print((time.time() - start_time) / i)  # 0.0005393951269898108
    print(total_reward)
    envs.close()
