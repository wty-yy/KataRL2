import torch
import numpy as np
import gymnasium as gym
from pathlib import Path
from typing import Optional
from katarl2.envs.common.env_cfg import EnvConfig
from torch.utils.tensorboard import SummaryWriter
from katarl2.agents.common.base_agent_cfg import BaseAgentConfig

class BaseAgent:
    def __init__(
            self, *,
            cfg: BaseAgentConfig,
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

        if envs is not None:
            # 将环境的必要参数保存到agent配置中, 便于模型加载时使用
            cfg.num_envs = envs.num_envs
            cfg.obs_space = envs.single_observation_space
            cfg.act_space = envs.single_action_space
    
    def predict(self, obs: np.ndarray):
        raise NotImplementedError("The predict method must be implemented by the agent subclass.")
    
    def learn(self):
        """ 训练, 和envs环境交互步进次数为cfg.num_env_steps """
        raise NotImplementedError("The learn method must be implemented by the agent subclass.")
    
    def eval(self, env_step: Optional[int] = None):
        """ 评估, 和eval_envs环境交互步进次数为cfg.num_eval_episodes, 在train中记录需传入当前env_step
        仅需实现对应智能体的predict方法, 环境会自动包装RecordEpisodeStatistics记录总奖励, 从而实现以下评估代码
        """
        episodic_returns = []
        episodic_lens = []

        obs, _ = self.eval_envs.reset()
        while len(episodic_returns) < self.cfg.num_eval_episodes:
            action = self.predict(obs)
            obs, rewards, terminations, truncations, infos = self.eval_envs.step(action)
            
            if "final_info" in infos:
                final_info = infos['final_info']
                mask = final_info['_episode']
                episodic_returns.extend(final_info['episode']['r'][mask].tolist())
                episodic_lens.extend(final_info['episode']['l'][mask].tolist())

        if self.logger is not None and env_step is not None:
            self.logger.add_scalar("charts/episodic_return", np.mean(episodic_returns), env_step)
            self.logger.add_scalar("charts/episodic_length", np.mean(episodic_lens), env_step)

    def save(self, path: str | Path = 'default'):
        """ 保存, 使用torch.save保存所有转为cpu的模型权重和其他重要参数 """
        raise NotImplementedError("The save method must be implemented by the agent subclass.")

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device):
        """ 加载, 使用torch.load加载模型权重和其他重要参数 """
        raise NotImplementedError("The load method must be implemented by the agent subclass.")
