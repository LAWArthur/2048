import torch
import torch.nn.functional as F
from torch import nn

class Model2048Config:
    size = 20
    n_blocks = 12
    h_size = 128
    block_size = 256

class Block(nn.Module):
    def __init__(self, config: Model2048Config):
        super().__init__()
        self.ln = nn.LayerNorm(config.h_size)
        self.c_fc = nn.Linear(config.h_size, config.block_size)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.block_size, config.h_size)

    def forward(self, x):
        x = x + self.c_proj(self.gelu(self.c_fc(self.ln(x))))
        return x

class Model2048(nn.Module):
    def __init__(self, config: Model2048Config) -> None:
        super().__init__()
        self.config = config
        self.f_proj = nn.Linear(config.size, config.h_size)
        self.h_layers = nn.ModuleList(Block(config) for _ in range(config.n_blocks))
        self.l_proj = nn.Linear(config.h_size, 1)
    
    def forward(self, x):
        x = self.f_proj(x)
        for block in self.h_layers:
            x = block(x)
        y = self.l_proj(x).view(-1)
        return y