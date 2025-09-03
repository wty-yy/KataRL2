# CHANGELOG
## v0.1 (20250811 - 20250813)
1. 支持SAC算法(参考cleanrl), 并完成在5个mujoco-py环境上5个随机种子的测试
2. 已完成katarl2的SAC和cleanrl的实验测试, 确定曲线一致
3. 完成sac_run_matrix.py实验启动脚本
4. 支持SAC模型保存与加载, swanlab上传, 最终的模型保存与评估, GIF效果图上传
5. 加入SimbaSAC和官方效果基本一致, 需要注意的内容写在[NOTES](./NOTES.md)中, 并完成mujoco-py的3个环境, dmc的3个环境上测试3个种子
6. 加入对tb日志读取, 绘制曲线图, [plot_tb_graphs](./demos/common/plot_tb_graphs.py)
7. 加入`gymnasium[mujoco-py]`, `DMC`环境

P.S.
1. 由于gymnasium>1.0默认的会将VecEnv的autoreset在下一步中进行处理, 导致错误的将最后一帧状态和初始化的状态加入buff, 导致模型效果很差, 通过修改`gym.vector.SyncVectorEnv(envs_list, autoreset_mode=AutoresetMode.SAME_STEP)`修复
2. 在环境交互速度很快且obs非常大的情况下, torch可能默认启用所有线程, 导致服务器的cpu占满, 需要通过设置`os.environ["OMP_NUM_THREADS"] = "2"; os.environ["MKL_NUM_THREADS"] = "2"`避免该问题

## v0.2 (20250813 - 20250817)
1. 加入gymnaisum Atari环境
2. 加入更高效的envpool的Atari环境
3. 加入PPO算法, 确定和cleanrl曲线一致
4. 在PPO中测试10个Atari环境的三个种子
5. 在PPO中加入Simba, 基本打平手, 4个环境上前期有提升, 后期和原版类似

完成对这些模型进行8环境3种子测试 (192个测试)：
1. basic ppo
2. basic ppo IN
3. basic ppo LN
4. basic ppo IN-NBA
5. simba ppo LN-NBA
6. simba ppo
7. simba ppo orgNet
测试结果在图[ppo_basic_vs_simba_vs_diy.png](./assets/ppo_basic_vs_simba_vs_diy.png)中

P.S.
1. 由于游戏貌似可能能无限玩下去, Breakout和Phoenix发现, 通过加入TimeLimit限制为108k, 也就是玩家玩30min (参考MuZero, DreamerV3), 由于存在4frame skip因此agent的交互次数是25000
2. 验证时, 如果固定episode次数验证且开多环境并行验证, 则总episode次数应该是eval_episode*num_eval_envs (如果总episode次数为eval_episdoe, 则环境并行方差很大)

## v0.3 (20250822 - 20250903)
1. 修改Atari相关算法的action_repeat, 记录的步数为环境真实步进, 由于Atari默认叠4帧, 因此num_env_steps也从原来的1e7 -> 4e7
2. 修复DMC默认渲染导致速度较慢的问题
3. tyro中对大环境(gym,dmc,envpool)加入sub_command命令, 只有输入python *.py env:gym --help才会显示全部的配置参数, 因为每个环境的参数各不相同, 所以需要用子命令进一步划分明确, 也让环境配置更分离, 更容易配置
4. 修复Gymnaisum的Atari中默认v5有FrameSkip导致训练极差的问题, 现在统一Atari的环境标准为FrameSkip=1, sticky=0.25
5. 设置RecordVideo的最大长度为10min, 如果存储episode整个需要60min过慢, 而且后面可能都是重复动作
6. 加入连续控制版的PPO和SimbaPPO, 在MujocoPy和DMControl上测试
7. 优化plot_tb_graphs的绘图效果, 参考DreamerV3的绘图效果, 新增不同suit相同环境名的比对(gym和envpool下训练图像比对)
8. 发现envpool的TimeLimit代码Bug因此重新测试所有实验并绘制图像

P.S.
1. cleanrl的连续控制版本的PPO默认使用了, obs和reward的RMS来动态归一化, 这导致了train和eval环境上的RMS参数不一致, 因此不再使用gym的默认包装器, 而使用我们自己的统一的RMS包装器, 对向量化环境进行RMS (环境级别的)
2. NormalizeReward有一个小细节, reward的RMS计算指标是return的RMS, 并且归一化并不减去均值, 而只是除以标准差, 参考: [Gamma in VecNormalize for rms updates.](https://github.com/openai/baselines/issues/538)
3. 虽然lambda函数无法用pkl打包, 但可以用functools.partial或者返回包装的thunk函数替代lambda函数


## v0.4 (20250903 -)
1. 由于无法保证envpool和gymnasium的实验一致性, 删除了envpoolq全部环境, 最后存在版本为v0.3 9004d097a461996f19365fc3287acbb50c614624

TODO:
- [ ] 加入DreamerV3
