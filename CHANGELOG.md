# CHANGELOG
## v0.1 (20250811 - 20250813)
1. 支持SAC算法(参考cleanrl), 并完成在5个mujoco-py环境上5个随机种子的测试
2. 已完成katarl2的SAC和cleanrl的实验测试, 确定曲线一致
3. 完成sac_run_matrix.py实验启动脚本
4. 支持SAC模型保存与加载, swanlab上传, 最终的模型保存与评估, GIF效果图上传
5. 加入SimbaSAC和官方效果基本一致, 需要注意的内容写在[NOTES](./NOTES.md)中, 并完成mujoco-py的3个环境, dmc的3个环境上测试3个种子
6. 加入对tb日志读取, 绘制曲线图, [plot_tb_graphs](./demos/common/plot_tb_graphs.py)
7. 加入`gymnasium[mujoco-py]`, `DMC`环境

## v0.2 (20250813 - )
1. 加入gymnaisum Atari环境
2. 加入更高效的envpool的Atari环境
3. 加入PPO算法, 确定和cleanrl曲线一致
TODO:
- [ ] 在PPO中测试10个Atari环境的三个种子
- [ ] 在PPO中加入Simba测试是否有性能提升

P.S. 由于gymnasium>1.0默认的会将VecEnv的autoreset在下一步中进行处理, 导致错误的将最后一帧状态和初始化的状态加入buff, 导致模型效果很差, 通过修改`gym.vector.SyncVectorEnv(envs_list, autoreset_mode=AutoresetMode.SAME_STEP)`修复
