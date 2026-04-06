"""
rigid_dynamics.py - V13.0 Zenith-Adaptive (P1-Internal)
🧬 [SLA-Rigid] 刚体动力学算子：全分辨率自适应与 Ratio 深度耦合版。
修复说明：
1. 空间维度感知：自动计算 Aspect Ratio，抵消非正方形分辨率下的几何畸变。
2. 步数比率对齐：深度集成 P3 下发的 ratio，实现物理场随步数动态收敛。
3. 自动重心修正：解决 B 位（肌肉/生理）在宽屏下收缩导致的视觉偏离问题。
"""
import torch
import torch.nn.functional as F

class RigidDynamicsManager:
    def __init__(self, device="cuda", dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        # 缓存机制：增加分辨率与比率指纹，防止逻辑残留
        self.last_id = None
        self.last_p = None
        self.last_res = None
        self.last_ratio = None
        self.cached_grid = None

    @torch.no_grad()
    def process_rigid_delta(self, x, raw_id, payload):
        """
        核心方法：计算具备空间感知的刚体位移增量。
        """
        B_size, C, H, W = x.shape
        current_res = (H, W)
        
        # --- 1. 信号探测 (Zenith 协议) ---
        # 优先级：P3 实时 Ratio > 默认黄金比例 1.5
        ratio = payload.get("current_ratio", 1.5)
        # 实时物理压强
        p = payload.get("active_p", 1.414)
        
        # --- 2. 空间维度自适应计算 ---
        aspect_ratio = W / H
        # 计算各向同性补偿系数：确保在 1344x768 这种宽屏下，挤压压力在 X/Y 轴分布均匀
        # 这里的逻辑是抵消 Latent 空间的拉伸感
        comp_x = 1.0 / aspect_ratio if aspect_ratio > 1.0 else 1.0
        comp_y = aspect_ratio if aspect_ratio < 1.0 else 1.0

        # --- 3. 缓存命中检查 ---
        if (self.last_id == raw_id and 
            self.last_p == p and 
            self.last_res == current_res and
            self.last_ratio == ratio):
            return self.cached_grid

        # --- 4. 仿射变换矩阵构建 ---
        # 基础单位矩阵 [1, 0, 0] [0, 1, 0]
        theta = torch.eye(2, 3, device=self.device, dtype=self.dtype).unsqueeze(0).repeat(B_size, 1, 1)

        # --- 5. 核心物理场逻辑 (1.4P 协议) ---
        if raw_id.startswith("R"):
            # 【R 位：骨骼/根节点】
            # 作用：轻微强化结构稳定性，受 ratio 影响较小，保持地基稳固
            rotation_intensity = 0.001 * p * (1.0 / ratio)
            theta[:, 0, 1] = rotation_intensity  # 剪切力注入
            theta[:, 1, 0] = -rotation_intensity

        elif raw_id.startswith("B"): 
            # 【B 位：生理肌肉挤压 - 1.4P 核心】
            # 逻辑：Ratio 越大（步数少），挤压越狠。
            # 使用 base_scale 结合 comp 系数实现非对称分辨率下的完美圆周挤压
            base_scale = 0.0025 * p * (ratio / 1.5)
            
            scale_x = 1.0 - (base_scale * comp_x)
            scale_y = 1.0 - (base_scale * comp_y)
            
            theta[:, 0, 0] *= scale_x  
            theta[:, 1, 1] *= scale_y  
            
            # --- 自动重心修正 (Core Anchor) ---
            # 修正点：原 0.08 硬编码改为随 H/W 比例动态偏移
            # 确保收缩时，人物的中心点（如脸部或躯干中心）不会因分辨率拉伸而产生视觉位移
            anchor_offset = (1.0 - scale_y) * 0.08 * (H / W if H > W else 1.0)
            theta[:, 1, 2] -= anchor_offset 

        # --- 6. 执行网格采样与增量提取 ---
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        # 基础对等网格（用于计算 Delta）
        id_grid = F.affine_grid(
            torch.eye(2, 3, device=self.device, dtype=self.dtype).unsqueeze(0).repeat(B_size, 1, 1), 
            x.size(), align_corners=True
        )
        
        # 计算物理场位移矢量 (Delta Flow)
        delta_flow = (grid - id_grid).permute(0, 3, 1, 2) 

        # 更新缓存
        self.last_id = raw_id
        self.last_p = p
        self.last_res = current_res
        self.last_ratio = ratio
        self.cached_grid = delta_flow

        return delta_flow

    def apply_rigid_stabilizer(self, x, delta):
        """
        物理平滑器：执行亚像素采样，将计算出的位移应用到 Latent
        """
        B, C, H, W = x.shape
        # 构建采样网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=self.device, dtype=self.dtype),
            torch.linspace(-1, 1, W, device=self.device, dtype=self.dtype),
            indexing='ij'
        )
        base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
        
        # 叠加物理场位移
        sampling_grid = base_grid + delta.permute(0, 2, 3, 1)
        
        # 使用双线性插值进行物理拓印
        return F.grid_sample(x, sampling_grid, mode='bilinear', padding_mode='reflection', align_corners=True)