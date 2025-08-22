import time
import subprocess
from pathlib import Path
from pprint import pprint
from katarl2.common.utils import cvt_string_time
PATH_NOHUP_OUT_DIR = Path(__file__).parents[1] / "logs/runs"
PATH_NOHUP_OUT_DIR.mkdir(parents=True, exist_ok=True)

sleep_time = 0

# 配置任务
tasks = [
    # (env_subcommand, env_name, cuda_list, seed_list)
    # Envpool (no render)
    ("env:envpool", 'Assault-v5',          [2, 2, 2], [0, 1, 2]),
    ("env:envpool", 'Asterix-v5',          [2, 2, 2], [0, 1, 2]),
    ("env:envpool", 'Boxing-v5',           [2, 2, 2], [0, 1, 2]),
    ("env:envpool", 'Breakout-v5',         [3, 3, 3], [0, 1, 2]),
    ("env:envpool", 'Phoenix-v5',          [3, 3, 3], [0, 1, 2]),
    ("env:envpool", 'Pong-v5',             [3, 3, 3], [0, 1, 2]),
    ("env:envpool", 'Qbert-v5',            [4, 4, 4], [0, 1, 2]),
    ("env:envpool", 'Seaquest-v5',         [4, 4, 4], [0, 1, 2]),
    ("env:envpool", 'UpNDown-v5',          [4, 4, 4], [0, 1, 2]),
    ("env:envpool", 'WizardOfWor-v5',      [6, 6, 6], [0, 1, 2]),
    # Gymnasium (can render)
    ("env:gym", 'Assault-v5',          [1, 1, 1], [0, 1, 2]),
    ("env:gym", 'Asterix-v5',          [1, 1, 1], [0, 1, 2]),
    ("env:gym", 'Boxing-v5',           [1, 1, 1], [0, 1, 2]),
    ("env:gym", 'Breakout-v5',         [1, 1, 1], [0, 1, 2]),
    ("env:gym", 'Phoenix-v5',          [1, 1, 1], [0, 1, 2]),
    ("env:gym", 'Pong-v5',             [1, 1, 1], [0, 1, 2]),
    ("env:gym", 'Qbert-v5',            [1, 1, 1], [0, 1, 2]),
    ("env:gym", 'Seaquest-v5',         [1, 1, 1], [0, 1, 2]),
    ("env:gym", 'UpNDown-v5',          [1, 1, 1], [0, 1, 2]),
    ("env:gym", 'WizardOfWor-v5',      [1, 1, 1], [0, 1, 2]),
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
            log_file = str(PATH_NOHUP_OUT_DIR / f"ppo_{env_subcommand}_{env_name}_seed{seed}_{time.strftime('%Y%m%d_%H%M%S')}.log")
            full_cmd = ["nohup"] + cmd + [">", log_file, "2>&1", "&", "echo", "$!"]
            pid = subprocess.check_output(" ".join(full_cmd), shell=True).decode().strip()
            # os.system(" ".join(full_cmd))
            print(f"[Start PID={pid}]: '{' '.join(full_cmd)}'")
            total += 1
    print(f"Successfully start {total} tasks.")
