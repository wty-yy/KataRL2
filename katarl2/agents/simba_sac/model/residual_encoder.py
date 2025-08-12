import torch
import torch.nn as nn
import torch.nn.functional as F

def init_he_linear(linear: nn.Linear):
    """ He normal (Kaiming normal) 初始化 + 零偏置，用于 ReLU """
    nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)

def init_orthogonal_linear(linear: nn.Linear, gain: float = 1.0):
    """ 正交初始化 + 零偏置 """
    nn.init.orthogonal_(linear.weight, gain=gain)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
        init_he_linear(self.fc1)
        init_he_linear(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.ln(x)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        return res + x

class SACResidualEncoder(nn.Module):
    """
    PyTorch 版的 residual encoder:
      x -> Linear(orthogonal, gain=1) -> [ResidualBlock]*N -> LayerNorm
    说明：
      - 输入 x 形状应为 (B, *, in_dim)，最后一维是特征维度
      - 若输入是图像，请先展平或用卷积得到特征后再喂入
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_blocks: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        init_orthogonal_linear(self.proj, gain=1.0)

        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.ln_out = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_out(x)
        return x
