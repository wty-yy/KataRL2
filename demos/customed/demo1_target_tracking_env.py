from typing import Literal
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo

class TargetTrackingEnv(gym.Env):
    """
    一个2D目标追踪环境。
    智能体需要访问空间中的n个目标点。

    - 状态空间 (Observation):
        - 智能体位置 (2)
        - 智能体速度 (2)
        - 最近的未访问目标的相对位置 (2)  (partial_obs=True)
        # - 全部目标的相对位置 (2*n_targets)  (partial_obs=False, 从近到远排序)
        - 全部目标相对位置及掩码 (2*n_targets + n_targets)  (partial_obs=False)
    - 动作空间 (Action):
        - 施加在智能体上的二维力 (2)，范围 [-1, 1]
    - 奖励 (Reward):
        sparse reward:
            - +10: 第一次到达一个目标点
            - -0.05: 每一步的时间惩罚
        continuous reward (额外奖励):
            - (上次到最近目标的距离 - 当前距离) * 缩放系数
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
            self,
            n_targets=1,
            env_ndim=2,
            partial_obs=False,
            reward_mode: Literal['sparse', 'continuous']='sparse',
            render_mode=None,
            width=640, height=480,  # 前两维维度大小
            remain_dim_size=512,  # 其余维度维度大小 (如果env_ndim > 2)
        ):
        super().__init__()

        self.width = width
        self.height = height
        self.n_targets = n_targets
        self.partial_obs = partial_obs
        self.env_ndim = env_ndim
        self.reward_mode = reward_mode
        self.remain_dim_size = remain_dim_size
        
        # 物理参数
        self.agent_mass = 1.0
        self.force_multiplier = 2.0  # 动作力度乘数
        self.dt = 0.5                # 时间步长
        self.max_vel = 5.0           # 最大速度
        self.target_radius = 15      # 目标被视为“到达”的半径
        self._max_episode_steps = 1000 # 最大回合步数

        # 定义动作空间: 2D力向量，每个分量在[-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.env_ndim,), dtype=np.float32
        )

        # 定义观测空间
        if self.partial_obs:
            # agent_pos (ndim) + agent_vel (ndim) + nearest target relative position (ndim)
            obs_dim = 3 * self.env_ndim
        else:
            # agent_pos (ndim) + agent_vel (ndim) + all targets relative positions (n_targets * ndim)
            # obs_dim = (2 + self.n_targets) * self.env_ndim
            # agent_pos (ndim) + agent_vel (ndim) + all targets relative positions (n_targets * ndim) + visited mask (n_targets)
            obs_dim = 2 * self.env_ndim + self.n_targets * (self.env_ndim + 1) # +1 for visited mask
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # 奖励配置
        self.reward_target = 10.0  # 到达目标奖励
        self.reward_time_penalty = -0.01  # 时间惩罚
        self.reward_distance_scale = 10.0 / max(self.width, self.height) if reward_mode == 'continuous' else 0.0  # 距离奖励缩放

        # Pygame 可视化相关
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        # 如果是 'human' 模式，初始化pygame display
        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Target Tracking Environment")
            self.clock = pygame.time.Clock()
        # 如果是 'rgb_array' 模式，只创建一个surface用于绘图
        elif self.render_mode == "rgb_array":
            self.screen = pygame.Surface((self.width, self.height))


    def _get_obs(self):
        """将环境状态整合成一个扁平的观测向量"""
        high = np.array([self.width, self.height] + [self.remain_dim_size] * (self.env_ndim - 2))
        obs = [
            self.agent_pos.flatten() / high, # 归一化位置
            self.agent_vel.flatten() / self.max_vel, # 归一化速度
        ]
        if self.partial_obs:
            _, min_target = self._get_min_distance_target()
            delta_min_target = (min_target - self.agent_pos) if min_target is not None else np.array([0.0, 0.0])
            obs.append((delta_min_target / high).flatten())
        # else:
        #     # 计算所有目标的相对位置，并按距离排序
        #     delta_positions = self.target_positions - self.agent_pos
        #     delta_positions[self.targets_visited == 0] = 0  # 已访问目标设为原点
        #     distances = np.linalg.norm(delta_positions, axis=1)
        #     distances[self.targets_visited == 0] = 1e9  # 已访问目标设为无穷远
        #     sorted_indices = np.argsort(distances)
        #     sorted_delta_positions = delta_positions[sorted_indices]
        #     obs.append((sorted_delta_positions / high).flatten())
        else:
            delta_positions = self.target_positions - self.agent_pos
            obs.append((delta_positions / high).flatten())
            obs.append(self.targets_unvisited.flatten())  # 访问掩码
        return np.concatenate(obs).astype(np.float32)

    def _get_info(self):
        """返回辅助信息"""
        return {"targets_remaining": int(np.sum(self.targets_unvisited))}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 随机初始化智能体位置和速度
        low = np.zeros(self.env_ndim)
        high = np.array([self.width, self.height] + [self.remain_dim_size] * (self.env_ndim - 2))
        self.agent_pos = self.np_random.uniform(
            low=low, high=high, size=(self.env_ndim,)
        )
        self.agent_vel = np.zeros(self.env_ndim, dtype=np.float32)
        
        # 随机初始化目标位置
        self.target_positions = self.np_random.uniform(
            low=low + self.target_radius,
            high=high - self.target_radius,
            size=(self.n_targets, self.env_ndim)
        )
        
        # 重置已访问目标掩码 (1表示未访问)
        self.targets_unvisited = np.ones(self.n_targets, dtype=np.float32)

        # 重置步数计数器
        self._current_step = 0

        # 重置上次最近距离
        self._last_min_distance = None
        
        return self._get_obs(), self._get_info()
    
    def _get_min_distance_target(self):
        min_distance = np.inf
        min_target = None
        unvisited_targets = self.target_positions[self.targets_unvisited == 1]
        if unvisited_targets.shape[0] > 0:
            distances = np.linalg.norm(unvisited_targets - self.agent_pos, axis=1)
            min_idx = np.argmin(distances)
            min_distance = distances[min_idx]
            min_target = unvisited_targets[min_idx]
        
        if np.isinf(min_distance):
            min_distance = 0.0
        return min_distance, min_target


    def step(self, action):
        # 1. 更新物理状态
        force = np.clip(action, -1.0, 1.0) * self.force_multiplier
        acceleration = force / self.agent_mass
        
        self.agent_vel += acceleration * self.dt
        # 限制最大速度
        vel_norm = np.linalg.norm(self.agent_vel)
        if vel_norm > self.max_vel:
            self.agent_vel = (self.agent_vel / vel_norm) * self.max_vel
            
        self.agent_pos += self.agent_vel * self.dt

        # 边界检测
        self.agent_pos[0] = np.clip(self.agent_pos[0], 0, self.width)
        self.agent_pos[1] = np.clip(self.agent_pos[1], 0, self.height)
        # 如果撞墙，反向速度分量
        if self.agent_pos[0] == 0 or self.agent_pos[0] == self.width:
            self.agent_vel[0] *= -0.5
        if self.agent_pos[1] == 0 or self.agent_pos[1] == self.height:
            self.agent_vel[1] *= -0.5
            
        # 2. 计算奖励
        reward = self.reward_time_penalty  # 时间惩罚

        min_distance, _ = self._get_min_distance_target()
        if self._last_min_distance is not None:
            reward += (self._last_min_distance - min_distance) * self.reward_distance_scale
        self._last_min_distance = min_distance
        
        # 检查是否到达目标
        for i in range(self.n_targets):
            if self.targets_unvisited[i] == 1:
                distance = np.linalg.norm(self.agent_pos - self.target_positions[i])
                if distance < self.target_radius:
                    num_visited = self.n_targets - np.sum(self.targets_unvisited)
                    reward += self.reward_target * (num_visited + 1)  # 到达奖励
                    self.targets_unvisited[i] = 0 # 标记为已访问
                    self._last_min_distance = None # 到达后重置距离计算

        # 3. 检查结束条件
        self._current_step += 1
        terminated = np.all(self.targets_unvisited == 0)
        truncated = self._current_step >= self._max_episode_steps

        # 4. 获取观测和信息
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="human")'
            )
            return
        
        self._render_frame()

        if self.render_mode == "rgb_array":
            # Pygame的坐标系是(width, height)，我们需要返回(height, width, 3)
            return np.transpose(
                pygame.surfarray.array3d(self.screen), axes=(1, 0, 2)
            )

    def _render_frame(self):
        if self.screen is None:
            # Handle case where render() is called before reset() in a non-human mode
            if self.render_mode != "human":
                 self.screen = pygame.Surface((self.width, self.height))
            else: # Should not happen if __init__ is correct
                return

        # 绘制背景
        self.screen.fill((20, 20, 20)) # 深灰色背景

        # 绘制目标
        for i in range(self.n_targets):
            color = (50, 205, 50) if self.targets_unvisited[i] == 1 else (100, 100, 100) # 绿色/灰色
            pygame.draw.circle(
                self.screen,
                color,
                self.target_positions[i, :2].astype(int),  # 只绘制前两个维度
                self.target_radius,
            )

        # 绘制智能体
        pygame.draw.circle(
            self.screen,
            (65, 105, 225), # 皇家蓝
            self.agent_pos[:2].astype(int),
            10, # 智能体半径
        )
        
        if self.render_mode == "human":
            # 处理pygame事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()

            # 更新显示
            pygame.display.flip()
            # 控制帧率
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            if self.render_mode == "human":
                pygame.display.quit()
            pygame.quit()
            self.screen = None


if __name__ == '__main__':
    # --- 测试环境 ---
    # --- 1. 测试 'human' 渲染模式 ---
    print("--- 正在测试 'human' 渲染模式 ---")
    env_human = TargetTrackingEnv(
        reward_mode='continuous',
        render_mode="human",
        n_targets=1,
        env_ndim=2
    )
    
    from gymnasium.utils.env_checker import check_env
    try:
        check_env(env_human.unwrapped)
        print("环境通过了Gymnasium API检查。")
    except Exception as e:
        print(f"环境检查失败: {e}")

    print(f"{env_human.observation_space=}, {env_human.action_space=}")

    obs, info = env_human.reset()
    for _ in range(200):
        action = env_human.action_space.sample() 
        obs, reward, terminated, truncated, info = env_human.step(action)
        env_human.render() # 在循环中显式调用render
        if terminated or truncated:
            print("一个回合结束，重置环境。")
            obs, info = env_human.reset()
    env_human.close()
    print("'human' 模式测试完成。\n")

    # --- 2. 测试 'rgb_array' 渲染模式 ---
    print("--- 正在测试 'rgb_array' 渲染模式 ---")
    env_rgb = TargetTrackingEnv(reward_mode='continuous', render_mode="rgb_array", n_targets=5)
    env_rgb = RecordVideo(env_rgb, video_folder="./logs/debug/videos", episode_trigger=lambda x: True)
    obs, info = env_rgb.reset()

    frame = env_rgb.render()
    print(f"获取到的第一帧 (Frame) 的类型: {type(frame)}")
    print(f"第一帧的形状 (Shape): {frame.shape}")
    # 检查返回的数组是否正确
    assert isinstance(frame, np.ndarray), "返回的不是一个NumPy数组"
    
    # 获取第一帧
    for _ in range(200):
        action = env_rgb.action_space.sample() 
        obs, reward, terminated, truncated, info = env_rgb.step(action)
        if terminated or truncated:
            obs, info = env_rgb.reset()
            break
    
    env_rgb.close()
    print("'rgb_array' 模式测试完成。")
