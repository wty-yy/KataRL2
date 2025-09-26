import os

os.environ["LAZY_LEGACY_OP"] = "0"
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["TORCH_LOGS"] = "+recompiles"

import warnings

warnings.filterwarnings("ignore")

import torch
import math
from networks import SimbaV2Actor, SimbaV2Critic, Temperature, Ensemble, ModelConfig, update_model_config, l2normalize_network
from utils import soft_ce
from functools import partial

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def test_predict(obs, action):
    print("==== Test Single Actor ====")
    mean, logstd = actor(obs)
    print("Actor mean:", mean.shape)
    print("Actor logstd:", logstd.shape)

    print("\n==== Test Single Critic ====")
    value, log_prob = critic(obs, action)
    print("Critic value:", value.shape)
    print("Critic log prob:", log_prob.shape)

    print("\n==== Test Temperature ====")
    print("Temperature:", temperature().shape)

    print("\n==== Test Critic Ensemble ====")

    ensemble_value, ensemble_log_prob = critic_ensemble(obs, action)
    print("Ensemble Critic value:", ensemble_value.shape)
    print("Ensemble Critic log prob:", ensemble_log_prob.shape)

LOG_STD_MAX = 2
LOG_STD_MIN = -10

def test_train(obs_sample, action_sample, target_q):
    actor.train()
    critic_ensemble.train()
    temperature.train()

    # Actor loss
    mean, log_std = actor(obs_sample)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (torch.tanh(log_std) + 1)
    act_dist = torch.distributions.Normal(loc=mean, scale=log_std.exp())
    action = act_dist.sample()
    log_prob = act_dist.log_prob(action)

    ## 随便选一个logprob作为loss，测试一下
    actor_loss = (log_prob * temperature().detach()).mean()
    actor_loss.backward()
    actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 20)
    actor_opt.step()
    actor_opt.zero_grad(set_to_none=True)

    # Critic loss
    all_values, all_log_prob = critic_ensemble(obs_sample, action_sample)
    critic_loss = 0
    for i in range(ensemble_size):
        critic_loss += soft_ce(all_log_prob[i], target_q, cfg, input_mode='log_prob').mean()
    critic_loss /= ensemble_size

    ## **注意**
    ## 此处必须在backward()后执行clip_grad_norm_()，告诉torch.compile这里的网络参数地址是不会变的
    ## 否则torch.compile会报torch._dynamo.decorators.mark_static_address警告，而这个警告意味着更低的compile效率
    critic_loss.backward()
    critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic_ensemble.parameters(), 20)
    critic_opt.step()
    critic_opt.zero_grad(set_to_none=True)

    # Temperature loss
    entropy = -log_prob
    temp_loss = temperature() * (entropy.detach() - target_entropy).mean()
    temp_loss.backward()
    temp_grad_norm = torch.nn.utils.clip_grad_norm_(temperature.parameters(), 20)
    temp_opt.step()
    temp_opt.zero_grad(set_to_none=True)

    actor.eval()
    critic_ensemble.eval()
    temperature.eval()

    return actor_loss, critic_loss, temp_loss


if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    compile = True

    # hyperparameters
    lr = 3e-4
    obs_dim = 10
    action_dim = 3
    num_bins = 51
    ensemble_size = 5
    batch_size = 8
    target_entropy = -0.5 * action_dim
    model_cfg = ModelConfig(
        num_blocks=2,
        hidden_dim=64,
        c_shift=3.0,
    )
    update_model_config(model_cfg)

    # log_std_min = -2
    # log_std_max = 10

    class DummyCfg:
        vmin = -10.0
        vmax = 10.0
        num_bins = num_bins
        bin_size = (vmax - vmin) / (num_bins - 1)

    cfg = DummyCfg()

    actor = SimbaV2Actor(
        in_dim=obs_dim,
        action_dim=action_dim,
        cfg=model_cfg,
    ).to(device)

    make_critic = partial(
        SimbaV2Critic,
        in_dim=obs_dim + action_dim,
        num_bins=num_bins,
        min_v=cfg.vmin,
        max_v=cfg.vmax,
        cfg=model_cfg,
    )
    critic = make_critic().to(device)
    critic_ensemble = Ensemble([make_critic() for _ in range(ensemble_size)]).to(device)
    temperature = Temperature(initial_value=0.05).to(device)

    l2normalize_network(actor)
    print("Actor network normalized.")
    l2normalize_network(critic)
    print("Critic network normalized.")
    l2normalize_network(critic_ensemble)
    print("Critic Ensemble network normalized.")

    # Dummy inputs
    obs = torch.randn(batch_size, obs_dim).to(device)
    action = torch.randn(batch_size, action_dim).to(device)

    # test predict
    if compile:
        test_predict = torch.compile(test_predict, mode="reduce-overhead")
    torch.compiler.cudagraph_mark_step_begin()
    test_predict(obs, action)

    # test train
    actor_opt = torch.optim.Adam(actor.parameters(), lr=lr, capturable=True)
    critic_opt = torch.optim.Adam(critic_ensemble.parameters(), lr=lr, capturable=True)
    temp_opt = torch.optim.Adam(temperature.parameters(), lr=lr, capturable=True)
    if compile:
        test_train = torch.compile(test_train, mode="reduce-overhead")
    # 目标 Q 值：随机生成假目标（实际中来自目标网络 + bootstrapped reward）
    target_q = torch.randn(batch_size, 1).to(device) * 5.0  # 随机分布，但在合理范围内
    print("\n==== Test Training Step ====")
    torch.compiler.cudagraph_mark_step_begin()
    actor_loss, critic_loss, temp_loss = test_train(obs, action, target_q)
    print(
        "Training step completed. Actor Loss:{}, Critic Loss:{}, Temperature Loss:{}".format(
            actor_loss.item(), critic_loss.item(), temp_loss.item()
        )
    )
    # 对网络权重做l2normalize
    l2normalize_network(actor)
    l2normalize_network(critic_ensemble)
