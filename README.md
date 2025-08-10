# KataRL2

> [KataRL](https://github.com/wty-yy/katarl)第一版使用的是JAX，但结构布局错乱因此弃用

本版本使用PyTorch，参考
- [cleanrl](https://github.com/vwxyzjn/cleanrl)保持简洁性
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3.git)保持易用性
- [tyro](https://github.com/brentyi/tyro)利用Python的field对变量配置进行管理, 并支持美观的CLI解析

复现论文[simba](https://github.com/SonyResearch/simba)（原仓库为JAX版本）

## 运行环境安装
```bash
# conda创建环境并安装
conda env create -f conda_requirements.yaml
# pip安装
pip install -r requirements.txt
```

### 可选安装
上传tensorboard日志信息
- `wandb`: `pip install wandb`

智能体交互环境
- `gymnasium`:
    - `mujoco-py`: `pip install "gymnasium[mujoco]"`
    - 视频录制: `pip install "gymnasium[other]"` (包含opencv-python, matplotlib, tqdm, imageio, pandas)
