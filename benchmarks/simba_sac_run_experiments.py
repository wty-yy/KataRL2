import time
import subprocess
from pprint import pprint
from katarl2.common.utils import cvt_string_time
from pathlib import Path
PATH_NOHUP_OUT_DIR = Path(__file__).parents[1] / "logs/runs"
PATH_NOHUP_OUT_DIR.mkdir(parents=True, exist_ok=True)

sleep_time = 0

# 配置任务
tasks = [
    # (env_subcommand, env_name, cuda_list, seed_list)
    # Mujoco
    ("env:gym", "Ant-v4",                   [1, 1, 1], [0, 1, 2]),
    ("env:gym", "HalfCheetah-v4",           [2, 2, 2], [0, 1, 2]),
    ("env:gym", "Hopper-v4",                [0, 0, 0], [0, 1, 2]),
    ("env:gym", "HumanoidStandup-v4",       [3, 3, 3], [0, 1, 2]),
    ("env:gym", "Humanoid-v4",              [4, 4, 4], [0, 1, 2]),
    ("env:gym", "InvertedPendulum-v4",      [1, 1, 1], [0, 1, 2]),
    ("env:gym", "Pusher-v5",                [2, 2, 2], [0, 1, 2]),
    ("env:gym", "Walker2d-v4",              [2, 2, 2], [0, 1, 2]),
    # DMC EASY
    ("env:dmc", "acrobot-swingup",          [0, 0, 0], [0, 1, 2]),
    ("env:dmc", "cartpole-balance",         [0, 0, 0], [0, 1, 2]),
    ("env:dmc", "cartpole-balance_sparse",  [0, 0, 0], [0, 1, 2]),
    ("env:dmc", "cartpole-swingup",         [0, 0, 0], [0, 1, 2]),
    ("env:dmc", "cartpole-swingup_sparse",  [1, 1, 1], [0, 1, 2]),
    ("env:dmc", "cheetah-run",              [1, 1, 1], [0, 1, 2]),
    ("env:dmc", "finger-spin",              [1, 1, 1], [0, 1, 2]),
    ("env:dmc", "finger-turn_easy",         [1, 1, 1], [0, 1, 2]),
    ("env:dmc", "finger-turn_hard",         [2, 2, 2], [0, 1, 2]),
    ("env:dmc", "fish-swim",                [2, 2, 2], [0, 1, 2]),
    ("env:dmc", "hopper-hop",               [2, 2, 2], [0, 1, 2]),
    ("env:dmc", "hopper-stand",             [2, 2, 2], [0, 1, 2]),
    ("env:dmc", "pendulum-swingup",         [3, 3, 3], [0, 1, 2]),
    ("env:dmc", "quadruped-walk",           [3, 3, 3], [0, 1, 2]),
    ("env:dmc", "quadruped-run",            [3, 3, 3], [0, 1, 2]),
    ("env:dmc", "reacher-easy",             [3, 3, 3], [0, 1, 2]),
    ("env:dmc", "reacher-hard",             [0, 1, 2], [0, 1, 2]),
    ("env:dmc", "walker-stand",             [3, 3, 2], [0, 1, 2]),
    ("env:dmc", "walker-walk",              [2, 2, 2], [0, 1, 2]),
    ("env:dmc", "walker-run",               [4, 4, 4], [0, 1, 2]),
    # DMC HARD
    ("env:dmc", "humanoid-walk",            [3, 3, 3], [0, 1, 2]),
    ("env:dmc", "dog-walk",                 [4, 4, 4], [0, 1, 2]),
    ("env:dmc", "humanoid-run",             [2, 2, 2], [0, 1, 2]),
    ("env:dmc", "dog-run",                  [2, 2, 2], [0, 1, 2]),
    ("env:dmc", "dog-trot",                 [2, 2, 2], [0, 1, 2]),
    ("env:dmc", "humanoid-stand",           [2, 2, 2], [0, 1, 2]),
    ("env:dmc", "dog-stand",                [3, 3, 3], [0, 1, 2]),
]

# 额外参数（可选）
extra_args = [
    # "--logger.use-swanlab",
    # "--logger.use-wandb",
    "--agent.verbose 1",
    # "--debug",
]

# 基础命令
base_cmd = [
    "python", "-u", "demos/simba_sac.py",
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
    for env_subcommand, env_name, cuda_list, seed_list in tasks:
        for cuda_id, seed in zip(cuda_list, seed_list):
            cmd = base_cmd + [
                env_subcommand,
                "--env.env-name", env_name,
                "--env.seed", str(seed),
                "--agent.seed", str(seed),
                "--agent.device", f"cuda:{cuda_id}",
            ] + extra_args

            # 用 nohup 启动并输出到对应 log 文件
            log_file = str(PATH_NOHUP_OUT_DIR / f"simba_sac_{env_subcommand}_{env_name}_seed{seed}_{time.strftime('%Y%m%d_%H%M%S')}.log")
            full_cmd = ["nohup"] + cmd + [">", log_file, "2>&1", "&", "echo", "$!"]
            pid = subprocess.check_output(" ".join(full_cmd), shell=True).decode().strip()
            # os.system(" ".join(full_cmd))
            print(f"[Start PID={pid}]: '{' '.join(full_cmd)}'")
            total += 1
            time.sleep(1)  # 避免同时启动多个任务时系统过载
    print(f"Successfully start {total} tasks.")
