import tyro
import numpy as np
from katarl2.envs.env_maker import make_envs
from katarl2.envs.env_envpool import EnvpoolConfig
import envpool
import cv2

if __name__ == '__main__':
    cfg = tyro.cli(EnvpoolConfig)
    cfg.num_envs = 4
    cfg.seed = 4
    np.random.seed(cfg.seed)
    envs, _ = make_envs(cfg)
    obs, infos = envs.reset()
    print(obs.shape)
    print(infos)
    total_reward = 0
    # writer = cv2.VideoWriter(str("./logs/videos/test.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps=30, frameSize=(obs.shape[-1], obs.shape[-2]))
    # for i in range(200):
    #     action = np.random.randint(0, 4, size=cfg.num_envs)
    #     envs.step(action)

    for i in range(1000):
        action = np.random.randint(0, 4, size=cfg.num_envs)
        # action = np.array([0] * 4)
        obs, rewards, terminations, truncations, infos = envs.step(action)
        # if (obs[0] == obs[1]).all():
        #     print(f"{i=}, obs[0] == obs[1]")
        # img = np.mean(obs[0], axis=0).astype(np.uint8)
        # img = np.stack([img, img, img], axis=-1)
        # writer.write(img)
        # if (obs[0] == obs[1]).all():
        #     print(f"{i=}, obs[0] == obs[1]")
        total_reward += rewards.sum()
        # print(rewards)
        if 'final_info' in infos:
            print(f"{i=}")
            print(infos['final_info'])
    # writer.release()

    print(total_reward)  # 33
    envs.close()
