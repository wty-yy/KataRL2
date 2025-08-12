# KataRL2

> [KataRL](https://github.com/wty-yy/katarl)第一版使用的是JAX，但结构布局错乱因此弃用

算法全部使用PyTorch实现，参考
- [cleanrl](https://github.com/vwxyzjn/cleanrl)保持简洁性
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3.git)保持易用性
- [tyro](https://github.com/brentyi/tyro)利用Python的field对变量配置进行管理, 并支持美观的CLI解析
- [simba](https://github.com/SonyResearch/simba)复现论文中的SimbaSAC算法（原仓库为JAX版本）

## 特点
1. 简单的代码结构, 没有复杂的继承关系, 易于添加自定义算法
2. 支持模型的训练曲线保存, 模型保存与加载
3. 使用`tyro`方便地进行参数管理
4. 支持`swanlab`, `wandb`训练曲线上传
5. 环境使用`gymnasium`标准, 易于添加自定义环境
6. 相比`cleanrl`, `stable baselines3`更加高效, 减少冗余信息输出并减少嵌套代码
7. 用python脚本方便地启动`benchmark`测试
8. 保持日志文件路径规范, 自动读取tensorboard日志整理绘制曲线图

## 支持算法
| 算法 | `Box` | `Discrete` | `MultiDiscrete` | 网络结构 |
| - | - | - | - | - |
| SAC | ✔️ | ❌ | ❌ | MLP |
| SimbaSAC | ✔️ | ❌ | ❌ | MLP |

## 运行环境安装
1. 最简安装
```bash
# conda创建环境并安装
conda env create -f requirements/conda_requirements.yaml
# pip安装
pip install -r requirements/requirements.txt
```

2. 完全安装 (包含全部交互环境)
```bash
# conda创建环境并安装
conda env create -f requirements/conda_requirements_full.yaml
# pip安装
pip install -r requirements/requirements_full.txt
```

### 可选安装
上传tensorboard日志信息
- `swanlab`: `pip install swanlab`
- `wandb`: `pip install wandb` (中国大陆无法访问)

智能体交互环境 (完全安装中包含)
- `gymnasium`:
    - `mujoco-py`: `pip install "gymnasium[mujoco]"`
    - 视频录制: `pip install "gymnasium[other]"` (包含opencv-python, matplotlib, tqdm, imageio, pandas)

## 可能的报错
1. `mujoco.FatalError: an OpenGL platform library has not been loaded into this process, this most likely means that a valid OpenGL context has not been created before mjr_makeContext was called`
    参考[CSDN - 关于在vscode运行mujoco报错...](https://blog.csdn.net/weixin_43807119/article/details/141814122), 设置环境变量`conda env config vars set MUJOCO_GL=egl`, 重启`conda`环境即可
