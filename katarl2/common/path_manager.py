import time
from pathlib import Path
from katarl2.envs import get_env_name
from katarl2.envs.env_cfg import EnvConfig

class PathManager:
    PATH_ROOT = Path(__file__).parents[2]
    PATH_LOGS_DIR = PATH_ROOT / "logs"

    _PATH_LOGS = None
    _RUN_NAME = None
    def build_path_logs(
            self,
            agent_name: str, algo_name: str,
            env_cfg: EnvConfig, seeds: tuple[int, int]
        ) -> Path:
        """ 创建 logs/agent_name/algo_name/env_name/seed_<algo>_<env>/timestamp/ 日志目录
        Args:
            agent_name: str, 智能体名称, eg: 'sac', 'ppo'
            algo_name: str, 算法名称, eg: 'basic', 'simba', 'simba_v2'
            env_cfg: EnvConfig, 环境配置, 用于生成env_name, eg: 'Hoop-v4__gymnasium'
            seeds: tuple[int, int], 随机种子, 分别为智能体和第一个环境的随机种子
        Returns:
            path_logs: Path, 日志目录路径
        """
        if self._PATH_LOGS is None:
            env_name = get_env_name(env_cfg)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self._PATH_LOGS = (
                self.PATH_LOGS_DIR /
                f"{agent_name}/{algo_name}/{env_name}/seed_{seeds[0]}_{seeds[1]}/{timestamp}"
            )
            self._PATH_LOGS.mkdir(parents=True, exist_ok=True)
        self._RUN_NAME = f"{agent_name}|{algo_name}|{env_name}|seed_{seeds[0]}_{seeds[1]}|{timestamp}"
        return self.PATH_LOGS
    
    @property
    def PATH_LOGS(self) -> Path:
        """ 返回当前日志目录路径 """
        if self._PATH_LOGS is None:
            raise ValueError("Logs path has not been built yet. Call build_path_logs() first.")
        return self._PATH_LOGS

    @property
    def RUN_NAME(self) -> Path:
        """ 返回当前实验名 (用于wandb名称) """
        if self._RUN_NAME is None:
            raise ValueError("Run name has not been built yet. Call build_path_logs() first.")
        return self._RUN_NAME

path_manager = PathManager()
