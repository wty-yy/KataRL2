import time
import subprocess
import typing
from katarl2.envs.env_gymnasium import GymAtariEnvConfig
from typing import Literal
from pathlib import Path
from pprint import pprint
from katarl2.common.utils import cvt_string_time
PATH_NOHUP_OUT_DIR = Path(__file__).parents[1] / "logs/runs"
PATH_NOHUP_OUT_DIR.mkdir(parents=True, exist_ok=True)

sleep_time = 0

ppo_type2suffix = {  # suffix for action_type and env_subcommand
    'basic': '',
    'simba': '-simba'
}

# 配置任务
ppo_type: Literal['basic', 'simba'] = 'basic'
atari_suit: Literal['gym', 'envpool'] = 'gym'
tasks = [
    # (action_type, env_subcommand, env_name, cuda_list, seed_list)
    # Discrete Gymnasium
    ("agent:disc", f"env:{atari_suit}-atari", 'Assault-v5',       [1, 1, 1], [0, 1, 2]),
    ("agent:disc", f"env:{atari_suit}-atari", 'Asterix-v5',       [1, 1, 1], [0, 1, 2]),
    ("agent:disc", f"env:{atari_suit}-atari", 'BeamRider-v5',     [3, 3, 3], [0, 1, 2]),
    ("agent:disc", f"env:{atari_suit}-atari", 'Boxing-v5',        [3, 3, 3], [0, 1, 2]),
    ("agent:disc", f"env:{atari_suit}-atari", 'Breakout-v5',      [5, 5, 5], [0, 1, 2]),
    ("agent:disc", f"env:{atari_suit}-atari", 'Enduro-v5',        [5, 5, 5], [0, 1, 2]),
    ("agent:disc", f"env:{atari_suit}-atari", 'Frostbite-v5',     [5, 5, 5], [0, 1, 2]),
    ("agent:disc", f"env:{atari_suit}-atari", 'Hero-v5',          [5, 5, 5], [0, 1, 2]),
    ("agent:disc", f"env:{atari_suit}-atari", 'MsPacman-v5',      [6, 6, 6], [0, 1, 2]),
    ("agent:disc", f"env:{atari_suit}-atari", 'Phoenix-v5',       [6, 6, 6], [0, 1, 2]),
    ("agent:disc", f"env:{atari_suit}-atari", 'Pong-v5',          [6, 6, 6], [0, 1, 2]),
    ("agent:disc", f"env:{atari_suit}-atari", 'Qbert-v5',         [6, 6, 6], [0, 1, 2]),
    ("agent:disc", f"env:{atari_suit}-atari", 'Seaquest-v5',      [7, 7, 7], [0, 1, 2]),
    ("agent:disc", f"env:{atari_suit}-atari", 'SpaceInvaders-v5', [7, 7, 7], [0, 1, 2]),
    ("agent:disc", f"env:{atari_suit}-atari", 'UpNDown-v5',       [0, 1, 2], [0, 1, 2]),
    ("agent:disc", f"env:{atari_suit}-atari", 'WizardOfWor-v5',   [7, 7, 7], [0, 1, 2]),
    # Continuous
    ("agent:cont", "env:gym-mujoco", "Ant-v4",              [6, 6, 6], [0, 1, 2]),
    ("agent:cont", "env:gym-mujoco", "HalfCheetah-v4",      [0, 0, 0], [0, 1, 2]),
    ("agent:cont", "env:gym-mujoco", "Hopper-v4",           [1, 1, 1], [0, 1, 2]),
    ("agent:cont", "env:gym-mujoco", "HumanoidStandup-v4",  [7, 7, 7], [0, 1, 2]),
    ("agent:cont", "env:gym-mujoco", "Humanoid-v4",         [2, 2, 2], [0, 1, 2]),
    ("agent:cont", "env:gym-mujoco", "InvertedPendulum-v4", [3, 3, 3], [0, 1, 2]),
    ("agent:cont", "env:gym-mujoco", "Pusher-v5",           [4, 4, 4], [0, 1, 2]),
    ("agent:cont", "env:gym-mujoco", "Walker2d-v4",         [5, 5, 5], [0, 1, 2]),
    # DMC Easy
    ("agent:cont", "env:dmc", "walker-walk",                [2, 2, 2], [0, 1, 2]),
    ("agent:cont", "env:dmc", "walker-run",                 [0, 0, 0], [0, 1, 2]),
    # DMC HARD
    ("agent:cont", "env:dmc", "humanoid-walk",              [3, 3, 3], [0, 1, 2]),
    ("agent:cont", "env:dmc", "dog-walk",                   [6, 6, 6], [0, 1, 2]),
    ("agent:cont", "env:dmc", "humanoid-run",               [1, 1, 1], [0, 1, 2]),
    ("agent:cont", "env:dmc", "dog-run",                    [2, 2, 2], [0, 1, 2]),
    ("agent:cont", "env:dmc", "dog-trot",                   [3, 3, 3], [0, 1, 2]),
    ("agent:cont", "env:dmc", "humanoid-stand",             [4, 4, 4], [0, 1, 2]),
    ("agent:cont", "env:dmc", "dog-stand",                  [6, 6, 6], [0, 1, 2]),
]

# 额外参数（可选）
extra_args = [
    # "--logger.use-swanlab",
    # "--logger.use-wandb",
    "--agent.verbose 1",
    # "--debug",
    # "--agent.layer-norm-network",
    # "--agent.instance-norm-network",
    # "--agent.norm-before-activate-network",
    # "--agent.weight-decay 0.01",
    # "--agent.optimizer adamw",
]

# 基础命令
base_cmd = [
    "python", "-u", "demos/ppo.py",
]

print(f"Start tasks: {base_cmd+extra_args=}")
pprint(tasks)
if sleep_time > 0:
    print("Sleep for time: ", cvt_string_time(sleep_time))
    time.sleep(sleep_time)

def run_command(cmd):
    print(f"启动命令: {' '.join(cmd)}")
    subprocess.Popen(cmd)  # 非阻塞启动

if __name__ == "__main__":
    total = 0
    ppo_type_suffix = ppo_type2suffix[ppo_type]
    for action_type, env_subcommand, env_name, cuda_list, seed_list in tasks:
        for cuda_id, seed in zip(cuda_list, seed_list):
            cmd = base_cmd + [
                action_type + ppo_type_suffix,
                env_subcommand + ppo_type_suffix,
                "--env.env-name", env_name,
                "--env.seed", str(seed),
                "--agent.seed", str(seed),
                "--agent.device", f"cuda:{cuda_id}",
            ] + extra_args

            # 用 nohup 启动并输出到对应 log 文件
            log_file = str(PATH_NOHUP_OUT_DIR / f"{ppo_type}_ppo_{env_subcommand}_{env_name}_seed{seed}_{time.strftime('%Y%m%d_%H%M%S')}.log")
            full_cmd = ["nohup"] + cmd + [">", log_file, "2>&1", "&", "echo", "$!"]
            pid = subprocess.check_output(" ".join(full_cmd), shell=True).decode().strip()
            # os.system(" ".join(full_cmd))
            print(f"[Start PID={pid}]: '{' '.join(full_cmd)}'")
            total += 1
            time.sleep(1)  # 避免同时启动多个任务时系统过载
    print(f"Successfully start {total} tasks.")
