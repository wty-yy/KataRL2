# 测试torch.compile的加速效果

compile通用方案
```python
class Algo:
    def __init__(self):
        if cfg.compile:
            mode = "reduce-overhead"
            self.update_step = torch.compile(self.update_step, mode=mode)
    
    def update_step(self, data):
        # loss计算和反向传播, 以及优化器的step
        ...
    
    def update(self, minibatches):
        for epoch in range(cfg.num_epochs):
            for batch in minibatches:
                torch.compiler.cudagraph_mark_step_begin()
                self.update_step(batch)
```

这个写法更适合训练循环里的热点函数，不是所有`torch.compile`场景都必须机械套用。
核心思路是：
1. 把会被高频重复调用、图比较稳定、循环次数较少的部分单独拿出来编译
2. 外层Python循环尽量保留在eager模式
3. 每次调用这类compiled热点函数前，使用`torch.compiler.cudagraph_mark_step_begin()`来避免cudagraph输出跨step复用

测试服务器配置为 RTX 4090 + AMD EPYC 7763 64-Core Processor

## PPO

有三个函数可以编译加速：
1. `update_step`：PPO的更新函数, 包含了loss的计算和反向传播, 以及优化器的step
2. `get_action_and_value`：PPO的动作选择和价值评估函数, 包含了actor和critic的前向传播
3. `gae`：PPO的广义优势估计函数, 包含了优势函数的计算

总体来说，离散和连续都有速度提升，但离散动作提升更明显。
主要原因不是`mode="reduce-overhead"`本身有区别，而是离散Atari默认并行环境数更多，且单轮iteration更短，因此首轮compile成本更容易摊薄。

### 离散动作
在离散动作PPO中，`get_action_and_value`通常也值得编译。
主要原因是：
1. 并行环境数更多，单次policy调用的batch更大
2. `num_steps=128`较小，单轮iteration更短，首轮compile成本更容易摊薄
3. 离散策略分布路径相对更稳定，因此`get_action_and_value`收益更容易体现

测试指令如下
<details>
<summary>展开测试指令</summary>

```bash
CUDA_VISIBLE_DEVICES=0 python ./demos/ppo.py agent:disc env:envpool-atari --env.env-name Pong-v5 --agent.total-env-steps 1000000 --agent.verbose 2 --debug \
  --agent.compile \
  --agent.eval-per-interaction-step 100000000 \
  --agent.num-eval-episodes 1 \
  --env.num-eval-envs 1 \
  --agent.save-per-interaction-step 0
```

</details>

测试结果如下
| gae | get_action_and_value | update_step | 训练用时 | 提升比例 |
| --- | --- | --- | --- | --- |
| ❌ | ❌ | ❌ | 1m30s | +0.0% |
| ❌ | ❌ | ✅ | 1m20s | +12.5% |
| ❌ | ✅ | ✅ | 49s | +83.7% |
| ✅ | ❌ | ✅ | 1m15s | +20.0% |
| ✅ | ✅ | ✅ | 49s | +83.7% |

### 连续动作
在连续动作PPO中，更推荐优先只编译`update_step`。

原因如下：
1. 并行环境数较少，单次`get_action_and_value`调用的batch较小，首轮compile收益不稳定
2. 默认`num_steps=2048`较大，单轮iteration本身很重，首轮compile/捕获开销体感会非常明显
3. `gae`在连续控制里通常是长时间维递推，并且每个iteration只调用一次，compile成本不容易摊平

这里需要特别说明：
`get_action_and_value`和`gae`在连续动作中不是“死锁”或“真的卡住”，而是首轮compile成本较高，体感上像卡住，因此benchmark里默认不打开这两部分的编译。

测试指令如下
<details>
<summary>展开测试指令</summary>

```bash
CUDA_VISIBLE_DEVICES=0 python ./demos/ppo.py agent:cont env:dmc \
  --env.env-name walker-walk \
  --agent.total-env-steps 50000 \
  --agent.verbose 2 \
  --debug \
  --agent.compile \
  --agent.eval-per-interaction-step 100000000 \
  --agent.num-eval-episodes 1 \
  --env.num-eval-envs 1 \
  --agent.save-per-interaction-step 0
```

</details>

测试结果如下
| gae | get_action_and_value | update_step | 训练用时 | 提升比例 |
| --- | --- | --- | --- | --- |
| ❌ | ❌ | ❌ | 1m35s | +0.0% |
| ❌ | ❌ | ✅ | 1m11s | +33.8% |

结论：
1. `update_step`是最值得优先编译的部分，因为它调用频率高，且包含loss、反向传播和optimizer step
2. 离散动作中，`get_action_and_value`通常也值得编译
3. 连续动作中，`get_action_and_value`和`gae`未必值得编译，尤其是在`num_envs`较少、`num_steps`较大的配置下

## SPO
SPO是PPO的一个变体，主要区别在于它使用了更深的ResNet编码器，损失函数，动态调整lr，并行环境8envs，num_steps=256


<details>
<summary>展开网络参数量</summary>

```bash
# CNN
[INFO] Parameters: total=1,687,719, trainable=1,687,719

# ResNet-18
[INFO] Parameters: total=11,436,295, trainable=11,436,295
```

</details>

### 离散动作
测试指令如下
<details>
<summary>展开测试指令</summary>

```bash
CUDA_VISIBLE_DEVICES=0 python ./demos/ppo.py agent:disc-spo env:envpool-atari-spo --env.env-name Pong-v5 --agent.total-env-steps 1000000 --agent.verbose 2 --debug \
  --agent.compile \
  --agent.eval-per-interaction-step 100000000 \
  --agent.num-eval-episodes 1 \
  --env.num-eval-envs 1 \
  --agent.save-per-interaction-step 0
```

</details>

测试结果如下
| gae | get_action_and_value | update_step | 训练用时 | 提升比例 |
| --- | --- | --- | --- | --- |
| ❌ | ❌ | ❌ | 3m16s | +0.0% |
| ✅ | ✅ | ✅ | 2m09s | +51.9% |

### 连续动作
测试指令如下
<details>
<summary>展开测试指令</summary>

```bash
CUDA_VISIBLE_DEVICES=0 python ./demos/ppo.py agent:cont-spo env:gym-mujoco-spo \
  --env.env-name Hopper-v4 \
  --agent.total-env-steps 200000 \
  --agent.verbose 2 \
  --debug \
  --agent.compile \
  --agent.eval-per-interaction-step 100000000 \
  --agent.num-eval-episodes 1 \
  --env.num-eval-envs 1 \
  --agent.save-per-interaction-step 0
```

</details>

测试结果如下
| gae | get_action_and_value | update_step | 训练用时 | 提升比例 |
| --- | --- | --- | --- | --- |
| ❌ | ❌ | ❌ | 1m39s | +0.0% |
| ✅ | ✅ | ✅ | 1m12s | +37.5% |
