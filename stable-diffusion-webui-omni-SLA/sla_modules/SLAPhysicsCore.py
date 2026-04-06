"""
SLAPhysicsCore.py - V13.1.0 Zenith-Unified (P1-P3 事实调用版)
🧬 [SLA-Core] 职责：物理算子调度中心。
修正说明：
1. 事实调用机制：不主动探测模型环境，而是读取 payload 携带的 model_type 事实。
2. 压强自动校准：针对 v15 和 xl 的 Latent 能量分布差异，执行基础压强 (p_base) 的分流计算。
3. 跨算子同步：将 model_type 事实透传给下游 Rigid/Cloth/Fluid 算子。
"""
import torch
import logging
import random

# 路径自愈导入逻辑
try:
    from .rigid_dynamics import RigidDynamicsManager
    from .cloth_collision import SLAClothCollisionManager
    from .fluid_consistency import FluidConsistencyManager
except ImportError:
    from rigid_dynamics import RigidDynamicsManager
    from cloth_collision import SLAClothCollisionManager
    from fluid_consistency import FluidConsistencyManager

logger = logging.getLogger("SLA_Zenith")

class SLAPhysicsCore:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16
        # 初始化三大物理支柱
        self.rigid = RigidDynamicsManager(self.device, self.dtype)
        self.cloth = SLAClothCollisionManager(self.device, self.dtype)
        self.fluid = FluidConsistencyManager(self.device, self.dtype)

    def process_physics(self, q, payload, model_options):
        """
        核心物理执行函数
        q: Latent Tensor (采样中的潜在空间数据)
        payload: P1 审计出的 ID 资产锦囊（现在包含 model_type 事实）
        model_options: 包含 P2/P3 全局参数的容器
        """
        # --- 1. 信号探测 (Zenith 协议联动) ---
        sla_payload = model_options.get("sla_payload", {})
        
        # 核心比率：Ratio 越大（步数少），物理挤压越强；Ratio 越小（步数多），物理越细腻
        ratio = sla_payload.get("ratio", 1.5)
        
        # 获取 P1 节点设置的基础物理强度
        p_base = model_options.get("sla_base_pressure", 1.414)
        
        # [事实调用]：从 payload 获取由 Manager 确定的模型事实，默认为 v15
        model_type = payload.get("model_type", "v15")
        
        # 种子同步：确保物理碰撞的随机性在采样步中保持稳定
        seed = sla_payload.get("sla_seed_entropy", 0)
        random.seed(seed)

        # --- 2. 动态压强重构 (事实驱动分流) ---
        # 逻辑：XL 的 Latent 空间在高频细节上更敏感，需适当降低基础压强系数
        if model_type == "xl":
            p_adjusted = p_base * 0.85  # XL 专用压强校准系数
        else:
            p_adjusted = p_base         # v15 保持原始压强

        # 最终压强 = 校准后压强 * (当前比率 / 1.5)
        active_pressure = p_adjusted * (ratio / 1.5)

        # 获取审计过的 ID 列表 (R/B/F/E)
        sorted_ids = payload.get("id_list", [])
        if not sorted_ids:
            return q

        # --- 3. 物理场循环拓印 ---
        for rid in sorted_ids:
            prefix = rid[0]
            
            # 准备下发给子算子的增强载荷，包含模型事实
            payload["active_p"] = active_pressure
            payload["current_ratio"] = ratio
            payload["current_seed"] = seed
            payload["model_type"] = model_type  # 确保下游算子感知事实
            
            if prefix in ["R", "B"]:
                # 【刚体算子】：处理骨骼与生理肌肉挤压
                delta = self.rigid.process_rigid_delta(q, rid, payload)
                q = self.rigid.apply_rigid_stabilizer(q, delta)
            
            elif prefix == "F":
                # 【布料算子】：处理褶皱碰撞
                payload["current_id"] = rid
                q = q + self.cloth.calculate_collision_offset(q, payload) * active_pressure
            
            elif prefix == "E":
                # 【流体算子】：处理发丝与动量
                if ratio > 1.1:
                    q = self.fluid.apply_fluid_flow(q, payload)

        return q

    def _get_priority(self, rid):
        """定义物理计算的层级顺序"""
        p_map = {"R": 0, "B": 1, "F": 2, "E": 3}
        return p_map.get(rid[0], 99)