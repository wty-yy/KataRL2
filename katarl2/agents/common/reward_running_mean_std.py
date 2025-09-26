import numpy as np
from katarl2.agents.common.running_mean_std import RunningMeanStd

class RewardRunningMeanStd:
    """ Scale the reward with running std of return (discounted sum of rewards) """
    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-9, max_return: float = 10.0):
        """
        Args:
            gamma (float): Discount factor.
            epsilon (float): A small value to avoid division by zero.
            max_return (float): Maximum value for the return to avoid large variance.
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_return = max_return
        self.returns = 0.0
        self.current_max_returns = 0.0
        self.rms = RunningMeanStd(shape=(), epsilon=epsilon)
    
    def update(self, rewards: np.ndarray, dones: np.ndarray) -> None:
        """
        Update the running mean and std of return.

        Args:
            rewards (np.ndarray): Rewards at the current step.
            dones (np.ndarray): Done flags at the current step.
        """
        self.returns = self.returns * self.gamma * (1 - dones) + rewards
        self.rms.update(self.returns)
        self.current_max_returns = max(self.current_max_returns, max(abs(self.returns)))
    
    def normalize(self, rewards: np.ndarray) -> np.ndarray:
        """
        Normalize the rewards.

        Args:
            rewards (np.ndarray): Rewards to be normalized.

        Returns:
            np.ndarray: Normalized rewards.
        """
        std = np.sqrt(self.rms.var + self.epsilon)
        min_denominator = self.current_max_returns / self.max_return
        denominator = max(std, min_denominator)
        rewards = rewards / denominator
        return rewards
    
    def reset(self) -> None:
        """
        Reset the running statistics.
        """
        self.returns = 0.0
        self.current_max_returns = 0.0
        self.rms.reset()
    
    def get_data(self) -> dict[str, np.ndarray]:
        """
        Get data for saving.

        Returns:
            dict[str, np.ndarray]: A dictionary containing the mean, variance, and count.
        """
        data = {
            'rms': self.rms.get_statistics(),
            'current_max_returns': self.current_max_returns,
            'gamma': self.gamma,
            'max_return': self.max_return
        }
        return data
    
    def load_data(self, stats: dict[str, np.ndarray]) -> None:
        """
        Load data from a saved state.

        Args:
            stats (dict[str, np.ndarray]): A dictionary containing the mean, variance, and count.
        """
        self.rms.load_statistics(stats['rms'])
        self.current_max_returns = stats['current_max_returns']
        self.gamma = stats['gamma']
        self.max_return = stats['max_return']
