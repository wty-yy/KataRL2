from typing import Any, Literal
from dataclasses import dataclass
from katarl2.agents.common.base_agent_cfg import BaseAgentConfig

@dataclass
class SimbaPPOConfig(BaseAgentConfig):
    # Algorithm name
    algo_name: Literal['PPO'] = 'PPO'
    # Policy name
    policy_name: Literal['Simba'] = 'Simba'
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
    num_env_steps: int = int(4e7)

    """ hyper-parameters (each algorithm has diff params, here are some examples) """
    # the learning rate of the agent
    learning_rate: float = 2.5e-4
    # the weight decay of the agent
    weight_decay: float = 1e-2

    # the number of actor residual blocks
    actor_num_blocks: int = 1
    # the hidden dimension of actor residual block
    actor_hidden_dim: int = 128

    # the number of critic network residual blocks
    critic_num_blocks: int = 2
    # the hidden dimension of critic network residual block
    critic_hidden_dim: int = 512

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
    target_kl: float = None

    """ these params will be filled in runtime """
    # the batch size (computed in runtime)
    batch_size: int = 0
    # the mini-batch size (computed in runtime)
    minibatch_size: int = 0
    # the number of iterations (computed in runtime)
    num_iterations: int = 0

    """ DIY """
    origin_agent: bool = False
