from dataclasses import dataclass, field
from typing import Any, Union, Literal
from katarl2.agents.common.base_agent_cfg import BaseAgentConfig
from katarl2.agents.simbav2_sac.models.networks import ModelConfig


@dataclass
class ActorModelConfig(ModelConfig):
    num_blocks: int = 1
    hidden_dim: int = 128
    c_shift: float = 3.0

@dataclass
class CriticModelConfig(ModelConfig):
    num_blocks: int = 2
    hidden_dim: int = 512
    c_shift: float = 3.0

    # Whether to use the second critic network for CDQ (if environment has termination)
    use_cdq: bool = False
    # Number of atoms for the value distribution
    num_bins: int = 101
    # Minimum value of the value distribution
    min_v: float = -5.0
    # Maximum value of the value distribution
    max_v: float = 5.0

@dataclass
class SimbaV2ModelConfig:
    actor: ActorModelConfig = field(default_factory=ActorModelConfig)
    critic: CriticModelConfig = field(default_factory=CriticModelConfig)

@dataclass
class SimbaV2SACConfig(BaseAgentConfig):
    # Algorithm name
    algo_name: Literal['SAC'] = 'SAC'
    # Policy name for SimbaSAC
    policy_name: Literal['Simba'] = 'SimbaV2'
    # Action type for SimbaSAC
    action_type: Literal['continuous'] = 'continuous'
    # Network name for SimbaSAC
    network_name: Literal['MLP'] = 'MLP'

    """ Training """
    # Total environment steps
    total_env_steps: int = int(1e6)

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
    # Number of updates per interaction step ('auto' is set to env.action_repeat)
    updates_per_interaction_step: str | int = 'auto'
    # Rescale the RSNorm reward
    reward_normalized_max: float = 5.0
    """ learning rate schedule for actor, critic, temperature """
    # initial learning rate
    learning_rate_init: float = 1e-4
    # end learning rate
    learning_rate_end: float = 5e-5
    # fraction of entire training period over which the learning rate is annealed
    learning_rate_fraction: float = 1.0

    """ Model """
    model: SimbaV2ModelConfig = field(default_factory=SimbaV2ModelConfig)

    # entropy regularization coefficient
    ent_coef: float = 0.2
    # automatic tuning of the entropy coefficient
    autotune: bool = True
    # target entropy = -target_entropy_coef * action_dim
    temp_target_entropy_coef: float = 0.5
    # initial value of temperature parameter log_ent_coef
    temp_initial_value: float = 0.01

    """ DIY """
    q_loss: Literal['cross_entropy', 'mse', 'twohot', 'hl_gauss'] = 'cross_entropy'
