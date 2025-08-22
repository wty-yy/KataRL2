import os
import torch
import random
import numpy as np

def calc_gamma(max_episode_steps, action_repeat):
    # gamma value is set with a heuristic from TD-MPCv2
    eff_episode_len = max_episode_steps / action_repeat
    return max(min((eff_episode_len/5-1)/(eff_episode_len/5), 0.995), 0.95)

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def enable_deterministic_run():
    """ Set cudnn operator to be deterministic """
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

if __name__ == "__main__":
    print(calc_gamma(600, 1))
