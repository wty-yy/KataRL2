# Notes
## Simba
论文中在SAC上加了两点
1. obs全部做rms (Running Mean Std)
2. 网络使用Residual和LayerNorm
Simba官方的SAC和cleanrl的SAC区别：
1. 对于存在`termination` (例如机器人摔倒在地的重置, mujoco-py需要) 的环境上使用cdq (Clip Double Q-Network), 而cleanrl默认用cdq
2. 优化器使用的是`AdamW`且有`weight_decay=1e-2`, cleanrl用的是`Adam`没有`weight_decay`
3. 环境上使用`repeat_action=2`, 每个`interaction_step`更新两次模型`updates_per_interaction_step=2`; 而cleanrl间隔更新模型, 每隔两步更新一次, TD3的trick?
4. 环境上使用`rescale_action`将动作scale到`(-1,1)`
5. **重要** 使用`temp_target_entropy_coef=0.5`和`temp_initial_value=0.01`对目标熵系数和初始熵系数进行控制, cleanrl默认分别为`1.0, 1.0`
6. Q value估计比普通sac准确很多, 对于很大的return (mujoco-py), 需要对环境做reward_scale (mujoco-py取为0.1)
7. **重要** 官方的Simba的RMS的更新频率就是每个obs一次, buffer中是环境返回的obs, 每次更新模型和采样时候用obs更新RMS, 对于从buffer中采样出来的obs用实时的RMS进行更新(很重要)
