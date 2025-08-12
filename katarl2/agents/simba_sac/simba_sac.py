""" From https://github.com/SonyResearch/simba """
import torch
from torch import optim
import torch.nn.functional as F

import time
import random
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from katarl2.agents.simba_sac.simba_sac_cfg import SimbaSACConfig
from katarl2.agents.simba_sac.model.mlp_continuous import Actor, SoftQNetwork
from katarl2.common import path_manager
from katarl2.agents.common.buffers import ReplayBuffer
from typing import Optional
from katarl2.common.utils import cvt_string_time
from katarl2.agents.common.running_mean_std import RunningMeanStd
from katarl2.envs.env_cfg import EnvConfig
from katarl2.agents.common.utils import calc_gamma

class SimbaSAC:
    def __init__(
            self, *,
            cfg: SimbaSACConfig,
            envs: Optional[gym.vector.SyncVectorEnv] = None,
            env_cfg: Optional[EnvConfig] = None,
            logger: Optional[SummaryWriter] = None,
        ):
        """
        Args:
            cfg: SimbaSACConfig, SAC算法配置文件
            envs: Optional[gym.vector.SyncVectorEnv], 训练所需的交互环境
            env_cfg: Optional[EnvConfig], 环境配置文件
            logger: Optional[SummaryWriter], 训练记录的日志信息
        """
        self.cfg = cfg
        self.envs = envs
        self.env_cfg = env_cfg
        self.logger = logger
        self.device = cfg.device if torch.cuda.is_available() and 'cuda' in cfg.device else 'cpu'
        self.obs_space: gym.Space = cfg.obs_space
        self.act_space: gym.Space = cfg.act_space
        assert isinstance(self.act_space, gym.spaces.Box), f"[ERROR] Only continuous action space is supported, but current action space={type(self.act_space)}"

        """ Seed """
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.backends.cudnn.deterministic = True

        """ Model """
        self.actor = Actor(self.obs_space, self.act_space, cfg.policy_num_blocks, cfg.policy_hidden_dim).to(self.device)
        create_q_net = lambda: SoftQNetwork(self.obs_space, self.act_space, cfg.q_num_blocks, cfg.q_hidden_dim).to(self.device)
        self.qf1 = create_q_net()
        self.qf2 = create_q_net()
        self.qf1_target = create_q_net()
        self.qf2_target = create_q_net()
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        """ Optimizer """
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=cfg.q_lr, weight_decay=cfg.q_weight_decay)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=cfg.policy_lr, weight_decay=cfg.policy_weight_decay)

        """ Auto entropy tuning """
        if cfg.autotune:
            self.target_entropy = -torch.prod(torch.Tensor(self.act_space.shape).to(self.device)).item()
            self.log_ent_coef = torch.zeros(1, requires_grad=True, device=self.device)
            self.ent_coef = self.log_ent_coef.exp().item()
            self.ent_optimizer = optim.Adam([self.log_ent_coef], lr=cfg.q_lr)
        else:
            self.ent_coef = cfg.ent_coef
        
        """ Buffer """
        self.obs_space.dtype = np.float32
        self.rb = ReplayBuffer(
            cfg.buffer_size,
            self.obs_space,
            self.act_space,
            self.device,
            cfg.num_envs,
            handle_timeout_termination=False
        )

        """ Checkpoints """
        if envs is not None:
            PATH_LOGS = path_manager.PATH_LOGS
            self.PATH_CKPTS = PATH_LOGS / "ckpts"
            self.PATH_CKPTS.mkdir(exist_ok=True, parents=True)

        """ Running statistics normalization """
        self.rms = RunningMeanStd(self.obs_space.shape)

        self.train_steps = 0
        self.global_step = 0
    
    def predict(self, obs):
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor.get_action(torch.Tensor(obs).to(self.device))[0]
        return actions.detach().cpu().numpy()

    def learn(self, total_timesteps):
        cfg, envs, logger = self.cfg, self.envs, self.logger
        self.actor.train()
        self.qf1.train()
        self.qf2.train()
        self.qf1_target.train()
        self.qf2_target.train()

        if cfg.gamma == 'auto':
            # gamma value is set with a heuristic from TD-MPCv2
            cfg.gamma = calc_gamma(self.env_cfg.max_episode_steps, self.env_cfg.action_repeat)

        # start the game
        start_time = time.time()
        last_verbose_time = time.time()
        obs, _ = envs.reset()
        obs = self.rms.update(obs)

        if cfg.verbose == 2:
            bar = tqdm(range(total_timesteps))
        else:
            bar = range(total_timesteps)
        for self.global_step in bar:
            # put action logic here
            if self.global_step < cfg.learning_starts:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                actions, _, _ = self.actor.get_action(torch.Tensor(obs).to(self.device))
                actions = actions.detach().cpu().numpy()

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            next_obs = self.rms.update(next_obs)

            if "final_info" in infos:
                final_info = infos['final_info']
                mask = final_info['_episode']
                if cfg.verbose == 2:
                    bar.set_description(f"episodic_return={final_info['episode']['r'][mask].mean().round(2)}")
                logger.add_scalar("charts/episodic_return", final_info["episode"]["r"][mask].mean(), self.global_step)
                logger.add_scalar("charts/episodic_length", final_info["episode"]["l"][mask].mean(), self.global_step)

            # save data to reply buffer; handle `final_obs`
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = self.rms.update(infos["final_obs"][idx])
            self.rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

            # CRUCIAL step easy to overlook
            obs = next_obs

            # training.
            if self.global_step > cfg.learning_starts:
                self.train_steps += 1
                data = self.rb.sample(cfg.batch_size)
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = self.actor.get_action(data.next_observations)
                    qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
                    qf2_next_target = self.qf2_target(data.next_observations, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.ent_coef * next_state_log_pi
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * cfg.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
                qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # optimize the model
                self.q_optimizer.zero_grad()
                qf_loss.backward()
                self.q_optimizer.step()

                if self.global_step % cfg.policy_frequency == 0:  # TD 3 Delayed update support
                    for _ in range(
                        cfg.policy_frequency
                    ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                        pi, log_pi, _ = self.actor.get_action(data.observations)
                        qf1_pi = self.qf1(data.observations, pi)
                        qf2_pi = self.qf2(data.observations, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((self.ent_coef * log_pi) - min_qf_pi).mean()

                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor_optimizer.step()

                        if cfg.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = self.actor.get_action(data.observations)
                            ent_coef_loss = (-self.log_ent_coef.exp() * (log_pi + self.target_entropy)).mean()

                            self.ent_optimizer.zero_grad()
                            ent_coef_loss.backward()
                            self.ent_optimizer.step()
                            self.ent_coef = self.log_ent_coef.exp().item()

                # update the target networks
                if self.global_step % cfg.target_network_frequency == 0:
                    for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                        target_param.data.copy_(cfg.tau * param.data + (1 - cfg.tau) * target_param.data)
                    for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                        target_param.data.copy_(cfg.tau * param.data + (1 - cfg.tau) * target_param.data)

                if self.global_step % 100 == 0:
                    logger.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), self.global_step)
                    logger.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), self.global_step)
                    logger.add_scalar("losses/qf1_loss", qf1_loss.item(), self.global_step)
                    logger.add_scalar("losses/qf2_loss", qf2_loss.item(), self.global_step)
                    logger.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, self.global_step)
                    logger.add_scalar("losses/actor_loss", actor_loss.item(), self.global_step)
                    logger.add_scalar("losses/ent_coef", self.ent_coef, self.global_step)
                    time_used = time.time() - start_time
                    SPS = int(self.global_step / time_used)
                    logger.add_scalar("charts/SPS", SPS, self.global_step)
                    logger.add_scalar("charts/time_sec", time.time() - start_time, self.global_step)
                    if cfg.autotune:
                        logger.add_scalar("losses/ent_coef_loss", ent_coef_loss.item(), self.global_step)
                    if cfg.verbose == 1 and time.time() - last_verbose_time > 10:
                        last_verbose_time = time.time()
                        time_left = (total_timesteps - self.global_step) / SPS
                        print(f"[INFO] {self.global_step}/{total_timesteps} steps, SPS: {SPS}, [{cvt_string_time(time_used)}<{cvt_string_time(time_left)}]")


    def save(self, path='default') -> Path:
        to_cpu = lambda data: {k: v.to('cpu') for k, v in data.items()}
        data = {
            'config': self.cfg,
            'model': {
                'actor': to_cpu(self.actor.state_dict()),
                'qf1': to_cpu(self.qf1.state_dict()),
                'qf2': to_cpu(self.qf2.state_dict())
            },
            'rms': {**self.rms.get_statistics()}
        }
        if path == 'default':
            path_ckpt = self.PATH_CKPTS / f"sac-{self.train_steps}.pkl"
        else:
            path_ckpt = path
        torch.save(data, str(path_ckpt))
        print(f"[INFO] Save model-{self.train_steps} to {path_ckpt}.")
        return path_ckpt
    
    @classmethod
    def load(cls, path, device='cpu'):
        data = torch.load(str(path), map_location=device, weights_only=False)
        self = cls(cfg=data['config'])
        self.actor.load_state_dict(data['model']['actor'])
        self.qf1.load_state_dict(data['model']['qf1'])
        self.qf2.load_state_dict(data['model']['qf2'])
        self.rms.load_statistics(data['rms'])
        print(f"[INFO] Load SimbaSAC model from {path} successfully.")
        return self
