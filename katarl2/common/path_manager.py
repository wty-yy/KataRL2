import time
from pathlib import Path
from katarl2.envs import get_env_name
from katarl2.envs.common.env_cfg import EnvConfig
from katarl2.agents.common.agent_cfg import AgentConfig, get_full_policy_name

class PathManager:
    PATH_ROOT = Path(__file__).parents[2]
    PATH_LOGS_DIR = PATH_ROOT / "logs"
    PATH_DEBUG_LOGS_DIR = PATH_ROOT / "logs/debug"

    _PATH_LOGS = None
    _RUN_NAME = None
    def build_path_logs(
            self,
            agent_cfg: AgentConfig,
            env_cfg: EnvConfig,
            debug: bool = False
        ) -> Path:
        """ 创建 logs/agent_name/algo_name/env_name/seed_<algo>_<env>/timestamp/ 日志目录
        Args:
            agent_cfg: AgentConfig, 算法配置
            env_cfg: EnvConfig, 环境配置, 用于生成env_name, eg: 'Hoop-v4__gymnasium'
            debug: bool, 如果是debug模型, 则将日志保存在 logs/debug/ 文件夹下
        Returns:
            path_logs: Path, 日志目录路径
        """
        if self._PATH_LOGS is None:
            algo_name = agent_cfg.algo_name.lower()
            full_policy_name = get_full_policy_name(agent_cfg)
            env_name = get_env_name(env_cfg)
            seeds = [agent_cfg.seed, env_cfg.seed]
            timestamp = time.strftime("%Y%m%d-%H%M%S")

            self._PATH_LOGS = (
                (self.PATH_LOGS_DIR if not debug else self.PATH_DEBUG_LOGS_DIR) /
                f"{algo_name}/{full_policy_name}/{env_name}/seed_{seeds[0]}_{seeds[1]}/{timestamp}"
            )
            self._PATH_LOGS.mkdir(parents=True, exist_ok=True)
        self._RUN_NAME = f"{algo_name}|{full_policy_name}|{env_name}|seed_{seeds[0]}_{seeds[1]}|{timestamp}"
        if debug:
            self._RUN_NAME = f"debug|{self._RUN_NAME}"
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
