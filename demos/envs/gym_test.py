import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import time
import tyro
import numpy as np
from katarl2.envs.env_gymnasium import GymEnvConfig
from katarl2.envs.env_maker import make_envs

if __name__ == '__main__':
    cfg = tyro.cli(GymEnvConfig)
    cfg.env_name = 'Ant-v4'
    cfg.num_envs = 3
    cfg.seed = 0
    envs, _ = make_envs(cfg)
    envs.reset()
    act = np.full(envs.action_space.shape, 0.5)
    start_time = time.time()
    total_reward = 0
    for i in range(10000):
        obs, rewards, terminations, truncations, infos = envs.step(act)
        total_reward += rewards.mean()

        real_next_obs = obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_obs"][idx]

        # if terminations.any() or truncations.any():
        #     break
    print((time.time() - start_time) / i)  # 0.0005393951269898108
    print(total_reward)
    envs.close()
