"""
fluid_consistency.py - V13.0 Zenith-Adaptive (P1-Internal)
🧬 [SLA-Fluid] 流体与动量算子：动态收敛与时间步长同步版。
修复说明：
1. 动量收敛协议：引入 ratio 变量，步数越多（Ratio越小），动量自动锁定，消除发丝重影。
2. 缓存指纹升级：将分辨率与 ratio 绑定，确保步数切换时物理场实时更新。
3. 空间张力优化：针对宽屏比例调整能量掩码，防止边缘流体溢出。
"""
import torch
import torch.nn.functional as F

class FluidConsistencyManager:
    def __init__(self, device="cuda", dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        self.prev_delta = None
        self.base_momentum = 0.45  # 基础动量保持率
        # 性能优化：网格缓存字典
        self.grid_cache = {}

    def _generate_energy_mask(self, x, ratio):
        """
        [物理重构]：基于步数比率生成动态能量掩码。
        作用：在高步数阶段（Ratio趋近1.0）自动收缩物理场半径，保护细节。
        """
        B, C, H, W = x.shape
        # 计算 Latent 平均能量分布
        energy = torch.mean(torch.abs(x), dim=1, keepdim=True)
        # 归一化能量
        norm_energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-6)
        
        # 引入 Ratio 阈值：Ratio 越小，掩码越严格
        threshold = 0.3 * (1.0 / ratio)
        mask = torch.where(norm_energy > threshold, 1.0, 0.0).to(self.dtype)
        
        # 高斯平滑掩码边缘
        return F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1)

    @torch.no_grad()
    def apply_fluid_flow(self, x, physics_payload):
        """
        核心方法：执行具备步数感知的流体模拟。
        """
        if physics_payload is None:
            return x

        B, C, H, W = x.shape
        
        # --- 1. 信号探测 (Zenith 协议) ---
        # 优先级：P3 实时 Ratio > 默认 1.5
        ratio = physics_payload.get("current_ratio", 1.5)
        p = physics_payload.get("active_p", 1.414)
        seed = physics_payload.get("current_seed", 0)

        # --- 2. 动态衰减计算 (消除重影的核心) ---
        # 逻辑：当 Ratio = 1.5 (20步) 时，decay = 1.0 (全强度)
        # 当 Ratio = 1.1 (50步) 时，decay = 0.2 (大幅锁定)
        decay = torch.clamp(torch.tensor((ratio - 1.0) * 2.0), min=0.0, max=1.0).item()
        
        if decay <= 0.05: # 如果步数极高，直接跳过流体计算以保护画质
            return x

        # --- 3. 构建物理噪声场 (动量源) ---
        torch.manual_seed(seed)
        noise_field = torch.randn((B, 2, H, W), device=self.device, dtype=self.dtype) * 0.02 * p * decay

        # --- 4. 动量守恒计算 ---
        if self.prev_delta is not None and self.prev_delta.shape == noise_field.shape:
            # 引入比率系数调整动量保持率
            momentum_rate = self.base_momentum * (ratio / 1.5)
            combined_flow = self.prev_delta * momentum_rate + noise_field * (1.0 - momentum_rate)
        else:
            combined_flow = noise_field
        
        self.prev_delta = combined_flow.detach()

        # --- 5. 掩码保护与空间映射 ---
        energy_mask = self._generate_energy_mask(x, ratio)
        final_flow = combined_flow * energy_mask * decay

        return self._apply_final_sample(x, final_flow, ratio)

    def _apply_final_sample(self, x, flow, ratio):
        """
        执行亚像素重采样 (优化版：具备步数感知的缓存机制)
        """
        B, C, H, W = x.shape
        
        # --- 核心修复：指纹包含 Ratio ---
        # 这样当你调整步数时，网格会重新生成，确保物理场与时间步长同步
        res_key = f"{H}x{W}x{round(ratio, 2)}"
        
        if res_key not in self.grid_cache:
            # 自动清理旧缓存防止显存溢出
            if len(self.grid_cache) > 5:
                self.grid_cache.clear()
                
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1, 1, H, device=self.device, dtype=self.dtype),
                torch.linspace(-1, 1, W, device=self.device, dtype=self.dtype),
                indexing='ij'
            )
            self.grid_cache[res_key] = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
        
        base_grid = self.grid_cache[res_key]
        
        # 将流体矢量场应用到基础坐标系
        # 注意：flow 通道 0 是 X，通道 1 是 Y
        sampling_grid = base_grid + flow.permute(0, 2, 3, 1)
        
        return F.grid_sample(x, sampling_grid, mode='bilinear', padding_mode='reflection', align_corners=True)