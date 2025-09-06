from typing import Literal, Optional, Union, Any
from dataclasses import dataclass
from katarl2.agents.common.base_agent_cfg import BaseAgentConfig

@dataclass
class PPODiscreteConfig(BaseAgentConfig):
    # Algorithm name
    algo_name: Literal['PPO'] = 'PPO'
    # Policy name
    policy_name: Literal['Basic', 'Simba'] = 'Basic'
    # Action type
    action_type: Literal['discrete', 'continuous'] = 'discrete'
    # Network name
    network_name: Literal['CNN+MLP', 'MLP'] = 'CNN+MLP'
    # Random seed
    seed: int = 42
    # Output train/eval details, level 0,1,2, message from low to high
    verbose: int = 0
    # Pytorch model device, cpu, cuda, cuda:0, cuda:1, ...
    device: str = 'cuda'

    """ Training / Evaluating """
    # Total environment steps
    total_env_steps: int = int(4e7)

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


@dataclass
class PPOContinuousConfig(PPODiscreteConfig):
    # Action type
    action_type: Literal['discrete', 'continuous'] = 'continuous'
    # Network name
    network_name: Literal['CNN+MLP', 'MLP'] = 'MLP'
    # Total environment steps
    total_env_steps: int = int(1e6)
    # the learning rate of the optimizer
    learning_rate: float = 3e-4
    # the number of steps to run in each environment per policy rollout
    num_steps: int = 2048
    # the number of mini-batches
    num_minibatches: int = 32
    # the K epochs to update the policy
    update_epochs: int = 10
    # the surrogate clipping coefficient
    clip_coef: float = 0.2
    # coefficient of the entropy
    ent_coef: float = 0.0

@dataclass
class SimbaPPOContinuousConfig(PPOContinuousConfig):
    # Policy name
    policy_name: Literal['Basic', 'Simba'] = 'Simba'
    # the number of critic residual blocks
    critic_num_blocks: int = 2
    # the hidden dimension of critic residual block
    critic_hidden_dim: int = 512
    # the number of actor residual blocks
    actor_num_blocks: int = 1
    # the hidden dimension of actor residual block
    actor_hidden_dim: int = 128

    # the running mean std for observation
    obs_rms: Optional[Any] = None

    optimizer: Literal['adam', 'adamw'] = 'adamw'
    weight_decay: float = 1e-2

@dataclass
class SimbaPPODiscreteConfig(PPODiscreteConfig):
    # Policy name
    policy_name: Literal['Basic', 'Simba'] = 'Simba'
    # the number of critic residual blocks
    critic_num_blocks: int = 2
    # the hidden dimension of critic residual block
    critic_hidden_dim: int = 512
    # the number of actor residual blocks
    actor_num_blocks: int = 1
    # the hidden dimension of actor residual block
    actor_hidden_dim: int = 128

    # the running mean std for observation
    obs_rms: Optional[Any] = None

    optimizer: Literal['adam', 'adamw'] = 'adamw'
    weight_decay: float = 1e-2

PPOConfig = Union[PPODiscreteConfig, PPOContinuousConfig, SimbaPPOContinuousConfig, SimbaPPODiscreteConfig]
