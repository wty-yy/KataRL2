from typing import Any, Literal, Optional
from dataclasses import dataclass
from katarl2.agents.common.base_agent_cfg import BaseAgentConfig

@dataclass
class PPOConfig(BaseAgentConfig):
    # Algorithm name
    algo_name: Literal['PPO'] = 'PPO'
    # Policy name
    policy_name: Literal['Basic'] = 'Basic'
    # Action type
    action_type: Literal['discrete'] = 'discrete'
    # Network name
    network_name: Literal['CNN+MLP'] = 'CNN+MLP'
    # Random seed
    seed: int = 42
    # Output train/eval details, level 0,1,2, message from low to high
    verbose: int = 0
    # Pytorch model device, cpu, cuda, cuda:0, cuda:1, ...
    device: str = 'cuda'

    """ Training / Evaluating """
    # Total environment steps
    num_env_steps: int = int(1e7)

    """ hyper-parameters (each algorithm has diff params, here are some examples) """
    # the learning rate of the optimizer
    learning_rate: float = 2.5e-4
    # the number of steps to run in each environment per policy rollout
    num_steps: int = 128
    # Toggle learning rate annealing for policy and value networks
    anneal_lr: bool = True
    # the discount factor gamma
    gamma: float = 0.99
    # the lambda for the general advantage estimation
    gae_lambda: float = 0.95
    # the number of mini-batches
    num_minibatches: int = 4
    # the K epochs to update the policy
    update_epochs: int = 4
    # Toggles advantages normalization
    norm_adv: bool = True
    # the surrogate clipping coefficient
    clip_coef: float = 0.1
    # Toggles whether or not to use a clipped loss for the value function, as per the paper.
    clip_vloss: bool = True
    # coefficient of the entropy
    ent_coef: float = 0.01
    # coefficient of the value function
    vf_coef: float = 0.5
    # the maximum norm for the gradient clipping
    max_grad_norm: float = 0.5
    # the target KL divergence threshold
    target_kl: Optional[float] = None

    """ these params will be filled in runtime """
    # the batch size (computed in runtime)
    batch_size: int = 0
    # the mini-batch size (computed in runtime)
    minibatch_size: int = 0
    # the number of iterations (computed in runtime)
    num_iterations: int = 0

    """ DIY """
    layer_norm_network: bool = False
    instance_norm_network: bool = False
    norm_before_activate_network: bool = False
    weight_decay: float = 0.0
    optimizer: Literal['adam', 'adamw'] = 'adam'

