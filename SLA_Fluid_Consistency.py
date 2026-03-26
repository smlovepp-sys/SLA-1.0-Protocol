"""
【SLA 1.1 逻辑分支声明】
本模块专用作用于：解决“雨夜 (Rainy Nights)”模版下的画面闪烁。
通过散度约束，确保流体特征在生成过程中具有物理连续性。
"""

import torch
import torch.nn as nn

class SLAFluidConsistency(nn.Module):
    def __init__(self, weight=30.0):
        super().__init__()
        self.w = weight

    def forward(self, flow_fields):
        """
        作用：强制执行质量守恒（散度为零）。
        """
        # 计算特征流场的散度 (Divergence)
        div_x = flow_fields[:, 0, :, 1:] - flow_fields[:, 0, :, :-1]
        div_y = flow_fields[:, 1, 1:, :] - flow_fields[:, 1, :-1, :]
        divergence = torch.abs(div_x[:, :-1, :] + div_y[:, :, :-1])
        
        return torch.mean(divergence) * self.w
