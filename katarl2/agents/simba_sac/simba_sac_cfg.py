from typing import Literal
from dataclasses import dataclass
from typing import Any, Union
from katarl2.agents.common.base_agent_cfg import BaseAgentConfig

@dataclass
class SimbaSACConfig(BaseAgentConfig):
    # Algorithm name
    algo_name: Literal['SAC'] = 'SAC'
    # Policy name for SimbaSAC
    policy_name: Literal['Simba'] = 'Simba'
    # Action type for SimbaSAC
    action_type: Literal['continuous'] = 'continuous'
    # Network name for SimbaSAC
    network_name: Literal['MLP'] = 'MLP'
    # Random seed for SimbaSAC
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

    """ Training """
    # Total environment steps
    num_env_steps: int = int(1e6)

    """ hyper-parameters """
    # the replay memory buffer size
    buffer_size: int = int(1e6)
    # the discount factor gamma (auto is set with a heuristic from TD-MPCv2)
    gamma: Union[float, Literal['auto']] = 'auto'
    # target smoothing coefficient
    tau: float = 0.005
    # the batch size of sample from the replay memory buffer
    batch_size: int = 256
    # timestep to start learning
    learning_starts: int = int(5e3)
    # Number of updates per interaction step
    updates_per_interaction_step: int = 2

    # the number of policy residual blocks
    policy_num_blocks: int = 1
    # the hidden dimension of policy residual block
    policy_hidden_dim: int = 128
    # the learning rate of the policy network optimizer
    policy_lr: float = 1e-4
    # the weight decay of the policy network optimizer
    policy_weight_decay: float = 1e-2

    # the number of Q network residual blocks
    q_num_blocks: int = 2
    # the hidden dimension of Q network residual block
    q_hidden_dim: int = 512
    # the learning rate of Q network optimizer
    q_lr: float = 1e-4
    # the weight decay of Q network optimizer
    q_weight_decay: float = 1e-2
    # whether use clipping double Q-learning (When env has termination use it)
    use_cdq: bool = False

    # entropy regularization coefficient
    ent_coef: float = 0.2
    # automatic tuning of the entropy coefficient
    autotune: bool = True
    # target entropy = -target_entropy_coef * action_dim
    temp_target_entropy_coef: float = 0.5
    # initial value of temperature parameter log_ent_coef
    temp_initial_value: float = 0.01

