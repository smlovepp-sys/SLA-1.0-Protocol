"""
【SLA 1.1 逻辑分支声明】
本模块专用作用于：消除 AI 视频中物体的“非自然抖动”与“果冻形变”。
通过锁定物理质心（CoM），确保每帧位移符合动力学公式。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SLARigidDynamics(nn.Module):
    def __init__(self, gravity=-9.8, weight=25.0):
        super().__init__()
        self.g = gravity
        self.w = weight

    def forward(self, props, latents, dt=0.04):
        """
        作用：执行抛物线拟合与惯性修正。
        """
        # 物理公理：Delta_Pos = V0*dt + 0.5*g*t^2
        g_vec = torch.tensor([0, self.g, 0], device=latents.device)
        expected_pos = props['pos_t'] + props['velocity_t'] * dt + 0.5 * g_vec * (dt**2)
        
        # 轨迹偏离损失
        return F.mse_loss(props['pos_t_next'], expected_pos) * self.w
