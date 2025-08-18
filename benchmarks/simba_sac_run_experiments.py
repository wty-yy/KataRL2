import time
import subprocess
from katarl2.common.utils import cvt_string_time
from pathlib import Path
PATH_NOHUP_OUT_DIR = Path(__file__).parents[1] / "logs/runs"
PATH_NOHUP_OUT_DIR.mkdir(parents=True, exist_ok=True)

# sleep_time = 3 * 3600
# print("Sleep for time: ", cvt_string_time(sleep_time))
# time.sleep(sleep_time)

# 配置任务
tasks = [
    # (env_type, env_name, cuda_list, seed_list)
    ("gymnasium", "Hopper-v4",          [0, 0, 0], [0, 1, 2]),
    ("gymnasium", "Ant-v4",             [0, 0, 1], [0, 1, 2]),
    ("gymnasium", "HalfCheetah-v4",     [1, 1, 1], [0, 1, 2]),
    ("gymnasium", "HumanoidStandup-v4", [1, 1, 1], [0, 1, 2]),
    ("gymnasium", "Humanoid-v4",        [1, 1, 1], [0, 1, 2]),
    # # DMC EASY
    ("dmc", "walker-walk",              [2, 2, 2], [0, 1, 2]),
    ("dmc", "walker-run",              [4, 4, 4], [0, 1, 2]),
    # DMC HARD
    ("dmc", "humanoid-walk",            [3, 3, 3], [0, 1, 2]),
    ("dmc", "dog-walk",                 [4, 4, 4], [0, 1, 2]),
    ("dmc", "humanoid-run",             [2, 2, 2], [0, 1, 2]),
    ("dmc", "dog-run",                  [2, 2, 2], [0, 1, 2]),
    ("dmc", "dog-trot",                 [2, 2, 2], [0, 1, 2]),
    ("dmc", "humanoid-stand",           [2, 2, 2], [0, 1, 2]),
    ("dmc", "dog-stand",                [3, 3, 3], [0, 1, 2]),
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

def run_command(cmd):
    print(f"启动命令: {' '.join(cmd)}")
    subprocess.Popen(cmd)  # 非阻塞启动

if __name__ == "__main__":
    total = 0
    for env_type, env_name, cuda_list, seed_list in tasks:
        for cuda_id, seed in zip(cuda_list, seed_list):
            cmd = base_cmd + [
                "--env.env-type", env_type,
                "--env.env-name", env_name,
                "--env.seed", str(seed),
                "--agent.seed", str(seed),
                "--agent.device", f"cuda:{cuda_id}",
            ] + extra_args

            # 用 nohup 启动并输出到对应 log 文件
            log_file = str(PATH_NOHUP_OUT_DIR / f"log_simba_sac_{env_name}_seed{seed}_{time.strftime('%Y%m%d_%H%M%S')}.out")
            full_cmd = ["nohup"] + cmd + [">", log_file, "2>&1", "&", "echo", "$!"]
            pid = subprocess.check_output(" ".join(full_cmd), shell=True).decode().strip()
            # os.system(" ".join(full_cmd))
            print(f"[Start PID={pid}]: '{' '.join(full_cmd)}'")
            total += 1
            # time.sleep(1)  # 确保每个任务间有间隔, 避免过快启动导致资源竞争
    print(f"Successfully start {total} tasks.")
