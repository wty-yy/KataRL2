""" 基础的Agent配置类, 所有Agent配置文件都至少需要这些参数 """
from typing import Literal
from dataclasses import dataclass
from typing import Any

@dataclass
class BaseAgentConfig:
    # Algorithm name
    algo_name: str  # eg: PPO, SAC, ...
    # Policy name
    policy_name: str  # eg: Basic, Simba, SimbaV2, ...
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
    # Don't setup these params in CLI
    num_envs: Any = None
    obs_space: Any = None
    act_space: Any = None

    """ Logger """
    # Log every n interaction steps
    log_per_interaction_step: int = 2000

    """ Training / Evaluating """
    # Total environment steps
    num_env_steps: int = int(1e6)
    # Total agent interaction steps (Don't setup this param in CLI)
    num_interaction_steps: Any = None
    # Evaluation in learn() function
    eval_per_interaction_step: int = 10000
    # Number of evaluation episodes
    num_eval_episodes: int = 10

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

def get_full_policy_name(cfg: BaseAgentConfig) -> str:
    name = f"{cfg.policy_name.lower()}_{cfg.action_type.lower()}_{cfg.network_name.lower()}"
    if cfg.algo_name.lower() == 'sac' and cfg.policy_name.lower() == 'simba':
        if cfg.use_cdq:
            name += '_cdq'
    if cfg.algo_name.lower() == 'ppo' and cfg.policy_name.lower() == 'simba':
        if cfg.origin_agent:
            name += '_OrgNet'
    if cfg.algo_name.lower() == 'ppo' and cfg.policy_name.lower() == 'basic':
        if cfg.layer_norm_network:
            name += '_LN'
        if cfg.instance_norm_network:
            name += '_IN'
        if cfg.norm_before_activate_network:
            name += '_NBA'
        if cfg.optimizer != 'adam':
            name += f'_{cfg.optimizer}'
        if cfg.weight_decay != 0.0:
            name += '_WD'
    return name
