from typing import Any, SupportsFloat, Optional

import numpy as np

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from katarl2.envs.common.running_mean_std import RunningMeanStd

class NormalizeReward(
    gym.Wrapper[ObsType, ActType, ObsType, ActType], gym.utils.RecordConstructorArgs
):
    """ Normalizes immediate rewards such that their exponential moving average has an approximately fixed variance. """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        rms: Optional[RunningMeanStd] = None,
        update_rms: bool = True,
    ):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has an approximately fixed variance.

        Args:
            env (env): The environment to apply the wrapper
            epsilon (float): A stability parameter
            gamma (float): The discount factor that is used in the exponential moving average.
        """
        gym.utils.RecordConstructorArgs.__init__(self, gamma=gamma, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)

        if rms is None:
            self.return_rms = RunningMeanStd(shape=())
        else:
            self.return_rms = rms
        self.discounted_reward = np.array([0.0])
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_rms = update_rms

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment, normalizing the reward returned."""
        obs, reward, terminated, truncated, info = super().step(action)

        # Using the `discounted_reward` rather than `reward` makes no sense but for backward compatibility, it is being kept
        self.discounted_reward = self.discounted_reward * self.gamma * (
            1 - terminated
        ) + float(reward)
        if self.update_rms:
            self.return_rms.update(self.discounted_reward)

        # We don't (reward - self.return_rms.mean) see https://github.com/openai/baselines/issues/538
        normalized_reward = reward / np.sqrt(self.return_rms.var + self.epsilon)
        return obs, normalized_reward, terminated, truncated, info
