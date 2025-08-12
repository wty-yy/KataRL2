""" 基础的Agent配置类, 所有Agent配置文件都至少需要这些参数 """
from typing import Literal
from dataclasses import dataclass
from typing import Any

@dataclass
class AgentConfig:
    # Algorithm name
    algo_name: str
    # Policy name
    policy_name: str
    # Action type
    action_type: Literal['continuous', 'discrete']
    # Network name
    network_name: Literal['MLP', 'CNN+MLP']
    # Random seed
    seed: int = 42
    # Output train/eval details, level 0,1,2, message from low to high
    verbose: int = 0
    # Pytorch model device, cpu, cuda, cuda:0, cuda:1, ...
    device: str = 'cuda'
    
    """ Environment (setup after envs created) """
    # Don't setup these parameters in CLI
    num_envs: Any = None
    obs_space: Any = None
    act_space: Any = None

    """ hyper-parameters (each algorithm has diff params, here are some examples) """
    # the replay memory buffer size
    # buffer_size: int = int(1e6)
    # the discount factor gamma
    # gamma: float = 0.9
    # the batch size of sample from the replay memory buffer
    # batch_size: int = 256
    # entropy regularization coefficient
    # ent_coef: float = 1e-2
    # more params ...

def get_full_policy_name(cfg: AgentConfig) -> str:
    return f"{cfg.policy_name.lower()}_{cfg.action_type.lower()}_{cfg.network_name.lower()}"
