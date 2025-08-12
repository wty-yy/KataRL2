""" From stable_baselines3_common_running_mean_std.py """
import numpy as np

class RunningMeanStd:
    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4):
        """
        Calculates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon
        self.epsilon = epsilon

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean.copy()
        new_object.var = self.var.copy()
        new_object.count = float(self.count)
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, arr: np.ndarray) -> np.ndarray:
        expand_dim = False
        if arr.ndim == 1:
            expand_dim = True
            arr = arr[None, :]
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
        if expand_dim:
            arr = arr[0]
        return self.normalize(arr)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def get_statistics(self) -> dict[str, np.ndarray]:
        """
        Get the current mean, variance, and count as a dictionary.

        :return: A dictionary containing the mean, variance, and count.
        """
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count
        }

    def reset(self) -> None:
        """
        Reset the running statistics to their initial state.
        """
        self.mean = np.zeros_like(self.mean)
        self.var = np.ones_like(self.var)
        self.count = self.epsilon

    def load_statistics(self, stats: dict[str, np.ndarray]) -> None:
        """
        Load the running statistics from a dictionary.

        :param stats: A dictionary containing the mean, variance, and count.
        """
        self.mean = stats.get("mean", self.mean)
        self.var = stats.get("var", self.var)
        self.count = stats.get("count", self.count)
    
    def normalize(self, obs) -> np.ndarray:
        """
        Normalize the observation using the running statistics.
        """
        return (obs - self.mean) / np.sqrt(self.var + self.epsilon)
