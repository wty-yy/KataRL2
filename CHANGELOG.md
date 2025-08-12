# CHANGELOG
## v0.1 (20250811)
1. 支持SAC算法(参考cleanrl), 并完成在5个mujoco-py环境上5个随机种子的测试
2. 已完成katarl2的SAC和cleanrl的实验测试, 确定曲线一致
3. 完成sac_run_matrix.py实验启动脚本
4. 支持SAC模型保存与加载, swanlab上传, 最终的模型保存与评估, GIF效果图上传
5. 加入SimbaSAC, 并完成在3个mujoco-py环境上3个种子测试
6. 加入对tb日志读取, 绘制曲线图, [plot_tb_graphs](./demos/common/plot_tb_graphs.py)
7. 加入`gymnasium[mujoco-py]`, `DMC`环境

TODO:
- [ ] 复现的SimbaSAC无法稳定, 需要在DMC上与官方simba进行比对曲线
- [ ] 对sac和simbaSAC的learn中加入eval功能
- [ ] 加入PPO算法

P.S. 由于gymnasium>1.0默认的会将VecEnv的autoreset在下一步中进行处理, 导致错误的将最后一帧状态和初始化的状态加入buff, 导致模型效果很差, 通过修改`gym.vector.SyncVectorEnv(envs_list, autoreset_mode=AutoresetMode.SAME_STEP)`修复
