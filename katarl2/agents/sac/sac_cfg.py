from typing import Literal
from dataclasses import dataclass

@dataclass
class SACConfig:
    # Algorithm name
    algo_name: Literal['SAC'] = 'SAC'
    # Policy name for SAC
    policy_name: Literal['Basic', 'Simba', 'SimbaV2'] = 'Basic'
    # Action type for SAC
    action_type: Literal['continuous', 'discrete'] = 'continuous'
    # Network name for SAC
    network_name: Literal['MLP', 'CNN+MLP'] = 'MLP'
    # Random seed for SAC
    seed: int = 42
    # Output train/eval details
    verbose: bool = False
    # Pytorch model device, cpu, cuda, cuda:0, cuda:1, ...
    device: str = 'cuda'

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
