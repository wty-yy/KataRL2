import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import time
import tyro
import numpy as np
from katarl2.envs.env_dmc import DMCEnvConfig
from katarl2.envs.env_maker import make_envs
from katarl2.common.path_manager import path_manager

if __name__ == '__main__':
    cfg = tyro.cli(DMCEnvConfig)
    path_manager._PATH_LOGS = Path("./logs/debug")
    cfg.env_num = 'walker-walk'
    cfg.env_num = 1
    # cfg.capture_video = True
    cfg.seed = 0
    envs, _ = make_envs(cfg)
    print(type(envs.single_observation_space), type(envs.single_action_space))
    print(envs.single_action_space.low, envs.single_action_space.high)
    envs.reset()
    act = np.full(envs.action_space.shape, 0.0)
    start_time = time.time()
    for i in range(10000):
        obs, rewards, terminations, truncations, infos = envs.step(act)

        real_next_obs = obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                tmp = infos["final_obs"][idx]
                print(tmp.shape)
                real_next_obs[idx] = infos["final_obs"][idx]

        if terminations.any() or truncations.any():
            print(terminations, truncations)
            break
    print((time.time() - start_time) / i)  # 0.00028695897086046496
    envs.close()
