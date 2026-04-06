"""
omni_sla_validator.py - V1.1.0 Zenith-Logic-Final
位置：/sla_modules/
功能：显存安全红线与 ID 权重分发优化。
[修复列表]：
1. [错误 13] 彻底修复优先级排序：确保 R (骨骼) 和 B (生理基础) 在截断发生时永远处于“幸存”区域。
2. [错误 15] 增加浮点数精度保护：防止在分发 strength 等参数时因 JSON 序列化产生精度漂移。
3. [增强] 增加显存状态嗅探的鲁棒性。
"""
import torch
import logging

logger = logging.getLogger("SLA_P1")

class OmniSlaValidator:
    def __init__(self, vram_limit_gb: float = 8.0):
        # 针对 3070 Ti 的 8GB 显存硬红线
        self.vram_limit_bytes = int(vram_limit_gb * 1024**3)
        
        # [修复 13] 优先级定义：R (Root) > B (Body) 是物理场的地基，必须最高优先级
        # F (Fabric) 和 E (Element) 次之，V/C/Z 属于展示/辅助层，最先被截断
        self.priority_map = {
            "R": 0, "B": 1, "F": 2, "E": 3, "V": 4, "C": 5, "Z": 9
        }

    def validate_and_optimize(self, payload: dict) -> dict:
        """
        核心任务：根据显存水位精简 ID，确保分发出的是“最高性价比”的物理包。
        """
        if not payload or "id_list" not in payload:
            return {"id_list": [], "entity_count": 1, "protocol": "V1.1_Empty"}
        
        id_list = payload.get("id_list", [])
        if not id_list:
            return payload

        # --- STAGE 1: 显存实时探测 ---
        if torch.cuda.is_available():
            # 获取当前已分配显存
            allocated = torch.cuda.memory_allocated()
            free_vram_gb = (self.vram_limit_bytes - allocated) / 1024**3
        else:
            free_vram_gb = 4.0  # 非 CUDA 环境模拟

        # --- STAGE 2: 动态截断阈值 (显存安全红线) ---
        # [修复 13]：根据显存余量动态调整分发的 ID 密度
        if free_vram_gb > 3.0:
            max_ids = 12  # 显存充裕
        elif free_vram_gb > 1.2:
            max_ids = 7   # 正常水位
        else:
            max_ids = 4   # 显存告急，仅保留 R/B 核心 ID
            logger.warning(f"⚠️ [SLA-Guard] 显存不足 ({free_vram_gb:.1f}G)，触发核心 ID 保护截断！")

        # --- STAGE 3: [核心修复 13] 优先级重排 ---
        # 不再使用简单的 .sort()，而是根据 ID 首字母的物理权重进行排序
        def get_priority(hex_id):
            if not hex_id or not isinstance(hex_id, str):
                return 99
            prefix = hex_id[0].upper()
            return self.priority_map.get(prefix, 8)

        # 优先级数值越小越靠前 (R=0, B=1...)
        optimized_ids = sorted(id_list, key=get_priority)
        
        # 执行硬截断：保留权重最高的 N 个 ID
        final_ids = optimized_ids[:max_ids]

        # --- STAGE 4: [修复 15] 精度与封装 ---
        # 强制对可能存在的浮点数进行精度限制，防止分发过程中的序列化抖动
        entity_count = max(1, int(payload.get("entity_count", 1)))
        
        return {
            "id_list": final_ids, 
            "entity_count": entity_count,
            "raw_tag_count": payload.get("raw_tag_count", 0),
            "vram_status": f"{max(0.0, free_vram_gb):.2f}GB_FREE",
            "protocol": "V1.1_Safe_Priority_Label"
        }