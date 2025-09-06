""" From cleanrl/ppo_atari.py """
import gymnasium as gym
from typing import Optional
from katarl2.envs.common.env_cfg import EnvConfig
from torch.utils.tensorboard import SummaryWriter
from katarl2.agents.ppo.ppo_cfg import PPOConfig
from typing import Literal

import torch
from torch import nn
from torch import optim

import time
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from katarl2.agents.common.utils import set_seed_everywhere, enable_deterministic_run
from katarl2.agents.common.base_agent import BaseAgent
from katarl2.agents.ppo.models import (
    Agent,
    Agent_IN, Agent_IN_before_norm,
    Agent_LN, Agent_LN_before_norm,
    Agent_MLP_continuous,
    Agent_MLP_continuous_Simba,
    Agent_CNN_Simba,
)
from katarl2.common import path_manager
from katarl2.common.utils import cvt_string_time
from katarl2.agents.common.running_mean_std import RunningMeanStd

class PPO(BaseAgent):
    def __init__(
            self, *,
            cfg: PPOConfig,
            envs: Optional[gym.vector.SyncVectorEnv] = None,
            eval_envs: Optional[gym.vector.SyncVectorEnv] = None,
            env_cfg: Optional[EnvConfig] = None,
            logger: Optional[SummaryWriter] = None,
    ):
        super().__init__(cfg=cfg, envs=envs, eval_envs=eval_envs, env_cfg=env_cfg, logger=logger)

        if env_cfg is not None:
            cfg.batch_size = int(env_cfg.num_envs * cfg.num_steps)
            cfg.minibatch_size = int(cfg.batch_size // cfg.num_minibatches)
            cfg.num_iterations = cfg.total_env_steps // env_cfg.action_repeat // cfg.batch_size

        self.obs_space: gym.Space = cfg.obs_space
        self.act_space: gym.Space = cfg.act_space
        if cfg.action_type == 'discrete':
            assert isinstance(self.act_space, gym.spaces.Discrete), f"[ERROR] Only discrete action space is supported, but current action space={type(self.act_space)}"
        elif cfg.action_type == 'continuous':
            assert isinstance(self.act_space, gym.spaces.Box), f"[ERROR] Only continuous action space is supported, but current action space={type(self.act_space)}"

        """ Seed """
        set_seed_everywhere(cfg.seed)
        if cfg.deterministic:
            enable_deterministic_run()

        """ Model """
        if cfg.policy_name == 'Basic':
            if cfg.action_type == 'discrete':
                if cfg.layer_norm_network and not cfg.norm_before_activate_network:
                    print("[INFO] Use default Agent_LN")
                    self.agent = Agent_LN(self.act_space).to(self.device)
                elif cfg.layer_norm_network and cfg.norm_before_activate_network:
                    print("[INFO] Use default Agent_LN_before_norm")
                    self.agent = Agent_LN_before_norm(self.act_space).to(self.device)
                elif cfg.instance_norm_network and not cfg.norm_before_activate_network:
                    print("[INFO] Use default Agent_IN")
                    self.agent = Agent_IN(self.act_space).to(self.device)
                elif cfg.instance_norm_network and cfg.norm_before_activate_network:
                    print("[INFO] Use default Agent_IN_before_norm")
                    self.agent = Agent_IN_before_norm(self.act_space).to(self.device)
                else:
                    print("[INFO] Use default agent")
                    self.agent = Agent(self.act_space).to(self.device)
            elif cfg.action_type == 'continuous':
                self.agent = Agent_MLP_continuous(self.obs_space, self.act_space).to(self.device)
        elif cfg.policy_name == 'Simba':
            if cfg.action_type == 'discrete':
                self.agent = Agent_CNN_Simba(
                    self.act_space,
                    cfg.actor_hidden_dim, cfg.actor_num_blocks,
                    cfg.critic_hidden_dim, cfg.critic_num_blocks
                ).to(self.device)
            elif cfg.action_type == 'continuous':
                self.agent = Agent_MLP_continuous_Simba(
                    self.obs_space, self.act_space,
                    cfg.critic_num_blocks, cfg.critic_hidden_dim,
                    cfg.actor_num_blocks, cfg.actor_hidden_dim
                ).to(self.device)
        
        """ Create/Inherit Running Statistics Normalization """
        if cfg.policy_name == 'Simba':
            if cfg.obs_rms is None:
                cfg.obs_rms = RunningMeanStd(shape=self.obs_space.shape)
            self.rms = cfg.obs_rms
        else:
            self.rms = None

        if cfg.optimizer == 'adam':
            print(f"Use Adam, weight_decay={cfg.weight_decay}")
            self.optimizer = optim.Adam(self.agent.parameters(), lr=cfg.learning_rate, eps=1e-5, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == 'adamw':
            print(f"Use AdamW, weight_decay={cfg.weight_decay}")
            self.optimizer = optim.AdamW(self.agent.parameters(), lr=cfg.learning_rate, eps=1e-5, weight_decay=cfg.weight_decay)

        """ Buffer """
        self.obs = torch.zeros((cfg.num_steps, cfg.num_envs) + self.obs_space.shape).to(self.device)
        self.actions = torch.zeros((cfg.num_steps, cfg.num_envs) + self.act_space.shape).to(self.device)
        self.logprobs = torch.zeros((cfg.num_steps, cfg.num_envs)).to(self.device)
        self.rewards = torch.zeros((cfg.num_steps, cfg.num_envs)).to(self.device)
        self.dones = torch.zeros((cfg.num_steps, cfg.num_envs)).to(self.device)
        self.values = torch.zeros((cfg.num_steps, cfg.num_envs)).to(self.device)
    
    def try_rms(self, obs, update=False, normalize=False):
        if self.rms is not None:
            if update: self.rms.update(obs)
            if normalize: obs = self.rms.normalize(obs)
        return obs

    def predict(self, obs: np.ndarray):
        obs = self.try_rms(obs, normalize=True)
        action = self.agent.get_action_and_value(torch.Tensor(obs).to(self.device), train=False)[0]
        return action.detach().cpu().numpy()
    
    def learn(self):
        cfg: PPOConfig = self.cfg
        envs, logger = self.envs, self.logger
        self.interaction_step = 0
        fixed_start_time = start_time = time.time()
        last_log_interaction_step = -1
        last_eval_interaction_step = -1
        last_save_interaction_step = -1
        last_verbose_time = 0

        """ Training """
        next_obs, _ = envs.reset()
        self.try_rms(next_obs, update=True)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(cfg.num_envs).to(self.device)

        bar = range(1, cfg.num_iterations + 1)
        if cfg.verbose == 2:
            bar = tqdm(bar)
        for iteration in bar:
            # Annealing the rate if instructed to do so.
            if cfg.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
                lrnow = frac * cfg.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, cfg.num_steps):
                cfg.num_env_steps += cfg.num_envs * self.env_cfg.action_repeat
                self.interaction_step += 1

                self.obs[step] = next_obs
                self.dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(self.try_rms(next_obs, normalize=True))
                    self.values[step] = value.flatten()
                self.actions[step] = action
                self.logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                self.try_rms(next_obs, update=True)
                next_done = np.logical_or(terminations, truncations)
                self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(next_done).to(self.device)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(self.try_rms(next_obs, normalize=True)).reshape(1, -1)
                advantages = torch.zeros_like(self.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(cfg.num_steps)):
                    if t == cfg.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        nextvalues = self.values[t + 1]
                    delta = self.rewards[t] + cfg.gamma * nextvalues * nextnonterminal - self.values[t]
                    advantages[t] = lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + self.values

            # flatten the batch
            b_obs = self.obs.reshape((-1,) + self.obs_space.shape)
            b_obs = self.try_rms(b_obs, normalize=True)
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + self.act_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            # Convert actions to appropriate type
            if cfg.action_type == 'discrete':
                b_actions = b_actions.long()

            # Optimizing the policy and value network
            b_inds = np.arange(cfg.batch_size)
            clipfracs = []
            for epoch in range(cfg.update_epochs):
                cfg.num_train_steps += 1
                np.random.shuffle(b_inds)
                for start in range(0, cfg.batch_size, cfg.minibatch_size):
                    end = start + cfg.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if cfg.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if cfg.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -cfg.clip_coef,
                            cfg.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), cfg.max_grad_norm)
                    self.optimizer.step()

                if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            """ Logging """
            if self.interaction_step - last_log_interaction_step >= cfg.log_per_interaction_step:
                last_log_interaction_step = self.interaction_step
                time_used = time.time() - start_time
                SPS = int(cfg.num_env_steps / time_used)
                logs = {
                    "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/old_approx_kl": old_approx_kl.item(),
                    "losses/approx_kl": approx_kl.item(),
                    "losses/clipfrac": np.mean(clipfracs),
                    "losses/explained_variance": explained_var,
                    "charts/SPS": SPS,
                    "charts/time_sec": time_used,
                }
                for name, value in logs.items():
                    self.logger.add_scalar(name, value, cfg.num_env_steps)
                if cfg.verbose == 1 and time.time() - last_verbose_time > 10:
                    last_verbose_time = time.time()
                    time_left = (cfg.num_iterations - iteration) * cfg.num_steps * cfg.num_envs * self.env_cfg.action_repeat / SPS
                    print(f"[INFO] {iteration}/{cfg.num_iterations} iters, SPS: {SPS}, [{cvt_string_time(time_used)}<{cvt_string_time(time_left)}], total time used: {cvt_string_time(time.time() - fixed_start_time)}")
            
            """ Evaluating """
            if (
                last_eval_interaction_step == -1 or
                self.interaction_step - last_eval_interaction_step >= cfg.eval_per_interaction_step or
                iteration == cfg.num_iterations
            ):
                last_eval_interaction_step = self.interaction_step
                eval_time = self.eval()
                start_time += eval_time
            
            """ Save Model """
            if cfg.save_per_interaction_step > 0 and (
                self.interaction_step - last_save_interaction_step >= cfg.save_per_interaction_step or
                iteration == cfg.num_iterations
            ):
                last_save_interaction_step = self.interaction_step
                self.save('lastest' if cfg.overwrite_model else 'default')

    def save(self, path: Literal['default', 'lastest'] | Path = 'default'):
        to_cpu = lambda data: {k: v.to('cpu') for k, v in data.items()}
        data = {
            'config': self.cfg,
            'env_config': self.env_cfg,
            'agent': to_cpu(self.agent.state_dict()),
        }
        path_ckpt = self._save(data, path)
        return path_ckpt

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device):
        data = torch.load(str(path), map_location=device, weights_only=False)
        data['config'].device = device
        self = cls(cfg=data['config'], env_cfg=data['env_config'])
        self.agent.load_state_dict(data['agent'])
        print(f"[INFO] Load PPO model from {path} successfully.")
        return self


