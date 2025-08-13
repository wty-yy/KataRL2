import torch
import numpy as np
import gymnasium as gym
from pathlib import Path
from typing import Optional
from katarl2.envs.common.env_cfg import EnvConfig
from torch.utils.tensorboard import SummaryWriter
from katarl2.agents.common.agent_cfg import AgentConfig

class Agent:
    def __init__(
            self, *,
            cfg: AgentConfig,
            envs: Optional[gym.vector.SyncVectorEnv] = None,
            eval_envs: Optional[gym.vector.SyncVectorEnv] = None,
            env_cfg: Optional[EnvConfig] = None,
            logger: Optional[SummaryWriter] = None,
    ):
        """
        对于load模型无需训练, 不需要给出envs, eval_envs, env_cfg, logger
        即模型加载所需的全部配置均位于cfg中

        Args:
            cfg: AgentConfig, 算法所需的全部配置
            envs: Optional[gym.vector.SyncVectorEnv], 训练所需的交互环境
            eval_envs: Optional[gym.vector.SyncVectorEnv], 评估所需的环境
            env_cfg: Optional[EnvConfig], 环境配置文件
            logger: Optional[SummaryWriter], 训练记录的日志信息
        """
        self.cfg = cfg
        self.envs = envs
        self.eval_envs = eval_envs
        self.env_cfg = env_cfg
        self.logger = logger
        self.device = cfg.device if torch.cuda.is_available() and 'cuda' in cfg.device else 'cpu'
    
    def predict(self, obs: np.ndarray):
        raise NotImplementedError("The predict method must be implemented by the agent subclass.")
    
    def learn(self):
        """ 训练, 和envs环境交互步进次数为cfg.num_env_steps """
        raise NotImplementedError("The learn method must be implemented by the agent subclass.")
    
    def eval(self):
        """ 评估, 和eval_envs环境交互步进次数为cfg.num_eval_episodes """
        raise NotImplementedError("The eval method must be implemented by the agent subclass.")

    def save(self, path: str | Path):
        """ 保存, 使用torch.save保存所有转为cpu的模型权重和其他重要参数 """
        raise NotImplementedError("The save method must be implemented by the agent subclass.")

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device):
        """ 加载, 使用torch.load加载模型权重和其他重要参数 """
        raise NotImplementedError("The load method must be implemented by the agent subclass.")
