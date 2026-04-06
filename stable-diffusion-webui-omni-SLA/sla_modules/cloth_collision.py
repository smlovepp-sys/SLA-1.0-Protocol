"""
cloth_collision.py - V13.0 Zenith-Adaptive (P1-Internal)
🧬 [SLA-Cloth] 布料碰撞算子：动态压感与能量保护版本。
修复说明：
1. 步数敏感度耦合：引入 ratio 变量，自动调节布料褶皱的碰撞频率。
2. 能量溢出保护：对 edge_weight 执行物理钳位，防止高压下的画面撕裂。
3. 空间张力校准：针对宽屏分辨率（如 1344x768）优化 Laplacian 算子的结构抓取力。
"""
import torch
import torch.nn.functional as F

class SLAClothCollisionManager:
    def __init__(self, device="cuda", dtype=torch.float16):
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def calculate_collision_offset(self, x, physics_payload):
        """
        [物理重构]：基于图像比例与步数比率实时计算布料位移。
        """
        if physics_payload is None:
            return torch.zeros_like(x)

        # --- 1. 信号拦截与维度测量 ---
        # 优先级：P3 实时 Ratio > 默认 1.5
        ratio = physics_payload.get("current_ratio", 1.5)
        # 实时物理压强 (active_p)
        p = physics_payload.get("active_p", 1.414)
        
        B, C, H, W = x.shape
        aspect_ratio = W / H
        
        # 计算各向异性补偿：确保宽屏下横向纤维抗性保持恒定
        rel_scale_x = 1.0 / aspect_ratio if aspect_ratio > 1.0 else 1.0
        rel_scale_y = aspect_ratio if aspect_ratio < 1.0 else 1.0
        
        # 归一化张力因子：用于修正卷积核在不同分辨率下的能量分布
        norm_factor = (rel_scale_x + rel_scale_y) * 0.5
        
        # --- 2. 材质敏感度与 Ratio 耦合 ---
        current_id = physics_payload.get("current_id", "F0")
        # 识别薄材质 (如 F8/F9 丝绸、薄纱)
        is_thin = current_id.startswith("F8") or current_id.startswith("F9")
        
        # 核心逻辑：Ratio 越大（步数少），敏感度越高，产生更深硬的褶皱
        # 随着步数增加（Ratio 趋向 1.0），敏感度下降，布料更顺滑
        sensitivity = (1.5 if is_thin else 1.0) * (ratio / 1.5)
        
        # --- 3. 动态核校准与能量钳位 ---
        # 计算中心权重：这里加入了 ratio 驱动和安全钳位
        # 限制 edge_weight 最大值，防止 Latent 空间数值爆炸（解决图像白点/黑点）
        raw_edge_weight = 4.0 * p * sensitivity * norm_factor
        edge_weight = torch.clamp(torch.tensor(raw_edge_weight), min=2.0, max=8.0).to(self.device).to(self.dtype)
        
        # 构建改进型 Laplacian 物理卷积核
        # 该核用于提取 Latent 层中的结构梯度，并将其转化为物理位移
        kernel = torch.tensor([[-1, -1, -1], 
                               [-1, edge_weight, -1], 
                               [-1, -1, -1]], 
                              device=self.device, dtype=self.dtype).view(1, 1, 3, 3)
        
        # --- 4. 物理拓印：提取结构特征 ---
        # 仅对第一个通道（通常携带亮度/结构信息）进行卷积以节省显存
        edge_map = F.conv2d(x[:, :1, :, :], kernel, padding=1)
        
        # --- 5. 动态空间避让 ---
        # 逻辑：在图像边缘处衰减物理场，防止布料由于溢出屏幕而产生的拉伸畸变
        # 基于宽度的 5% 作为安全缓冲区
        margin = int(W * 0.05)
        mask = torch.ones((1, 1, H, W), device=self.device, dtype=self.dtype)
        if margin > 0:
            mask[:, :, :, :margin] = torch.linspace(0, 1, margin, device=self.device, dtype=self.dtype)
            mask[:, :, :, -margin:] = torch.linspace(1, 0, margin, device=self.device, dtype=self.dtype)
            
        # --- 6. 生成最终位移场 ---
        # 0.001 为微观物理常数，确保形变在采样可控范围内
        offset = edge_map.repeat(1, C, 1, 1) * 0.001 * mask
        
        return offset