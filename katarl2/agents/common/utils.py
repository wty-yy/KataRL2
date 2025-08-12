import math

def calc_gamma(max_episode_steps, action_repeat):
    # gamma value is set with a heuristic from TD-MPCv2
    eff_episode_len = max_episode_steps / action_repeat
    return max(min((eff_episode_len/5-1)/(eff_episode_len/5), 0.995), 0.95)

if __name__ == "__main__":
    print(calc_gamma(600, 1))
