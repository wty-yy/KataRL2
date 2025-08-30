""" From gymnasium.wrappers.NormalizeObservation """
import numpy as np

from typing import Optional 
import gymnasium as gym
from gymnasium.core import ActType, ObsType, WrapperObsType
from katarl2.envs.common.running_mean_std import RunningMeanStd


class NormalizeObservation(
    gym.ObservationWrapper[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """ Normalizes observations to be centered at the mean with unit variance. """

    def __init__(
            self,
            env: gym.Env[ObsType, ActType],
            epsilon: float = 1e-8,
            rms: Optional[RunningMeanStd] = None,
            update_rms: bool = True,
        ):
        """This wrapper will normalize observations such that each observation is centered with unit variance.

        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
            rms: If provided, this running mean and standard deviation will be used to normalize the observations.
            update_rms: If True, the running mean and standard deviation will be updated with each observation.
        """
        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
        gym.ObservationWrapper.__init__(self, env)

        assert env.observation_space.shape is not None
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

        if rms is None:
            self.obs_rms = RunningMeanStd(
                shape=self.observation_space.shape, epsilon=epsilon
            )
        else:
            self.obs_rms = rms
        self.update_rms = update_rms

    def observation(self, observation: ObsType) -> WrapperObsType:
        """Normalises the observation using the running mean and variance of the observations."""
        if self.update_rms:
            self.obs_rms.update(observation)
        return self.obs_rms.normalize(observation)
