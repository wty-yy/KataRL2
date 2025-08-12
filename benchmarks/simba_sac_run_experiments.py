import time
import subprocess
from pathlib import Path
PATH_NOHUP_OUT_DIR = Path(__file__).parents[1] / "logs/runs"
PATH_NOHUP_OUT_DIR.mkdir(parents=True, exist_ok=True)

# 配置任务
tasks = [
    # (env_name, cuda_list, seed_list)
    # ("Hopper-v4",          [3, 3, 3], [0, 1, 2]),
    # ("Ant-v4",             [4, 4, 4], [0, 1, 2]),
    # ("HalfCheetah-v4",     [2, 2, 2], [0, 1, 2]),
    # ("HumanoidStandup-v4", [3, 3, 3], [0, 1, 2]),
    # ("Humanoid-v4",        [6, 6, 7], [0, 1, 2]),
    ("dmc", "walker-walk",            [2, 2, 3], [0, 1, 2]),
    ("dmc", "humanoid-walk",          [3, 4, 7], [0, 1, 2]),
]

# 额外参数（可选）
extra_args = [
    "--logger.use-swanlab",
    # "--logger.use-wandb",
    "--agent.verbose 1",
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
    print(f"Successfully start {total} tasks.")
