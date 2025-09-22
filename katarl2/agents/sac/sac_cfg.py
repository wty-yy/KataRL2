from typing import Literal
from dataclasses import dataclass
from typing import Any
from katarl2.agents.common.base_agent_cfg import BaseAgentConfig

@dataclass
class SACConfig(BaseAgentConfig):
    # Algorithm name
    algo_name: Literal['SAC'] = 'SAC'
    # Policy name for SAC
    policy_name: Literal['Basic'] = 'Basic'
    # Action type for SAC
    action_type: Literal['continuous'] = 'continuous'
    # Network name for SAC
    network_name: Literal['MLP'] = 'MLP'
    # Random seed for SAC
    seed: int = 42
    # Output train/eval details, level 0,1,2, message from low to high
    verbose: int = 0
    # Pytorch model device, cpu, cuda, cuda:0, cuda:1, ...
    device: str = 'cuda'

    """ Training """
    # Total environment steps
    total_env_steps: int = int(1e6)

    """ hyper-parameters """
    # the replay memory buffer size
    buffer_size: int = int(1e6)
    # the discount factor gamma
    gamma: float = 0.99
    # target smoothing coefficient
    tau: float = 0.005
    # the batch size of sample from the replay memory buffer
    batch_size: int = 256
    # timestep to start learning
    learning_starts: int = int(5e3)
    # the learning rate of the policy network optimizer
    policy_lr: float = 3e-4
    # the learning rate of Q network optimizer
    q_lr: float = 1e-3
    # the frequency of training policy
    policy_frequency: int = 2
    # the frequency of updates for the target networks
    target_network_frequency: int = 1
    # entropy regularization coefficient
    ent_coef: float = 0.2
    # automatic tuning of the entropy coefficient
    autotune: bool = True

if __name__ == '__main__':
    import tyro
    args: SACConfig = tyro.cli(SACConfig)
    print(args)
