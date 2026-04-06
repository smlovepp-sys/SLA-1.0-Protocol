"""
OmniSLAMainNode.py - V1.5.6 逻辑标签中心 (Zenith-ZZ)
🧠 [SLA-P1] 职责：
1. 隐匿解析：从 Prompt 中提取物理标签及 ZZ 系列特殊 ID。
2. 协议封装：将提取的 ID 列表封装进 Conditioning 元数据，实现“黄色端口”输出。
3. 容错增强：修复了针对不同 CLIP 版本的 pooled_output 提取逻辑，防止 Tensor 对象报错。
"""
import torch
import logging
import os
import sys

# --- 路径自愈逻辑 ---
BASE_PATH = os.path.dirname(os.path.realpath(__file__))
MODULES_PATH = os.path.join(BASE_PATH, "sla_modules")
for p in [BASE_PATH, MODULES_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

# 尝试导入底层管理模块
try:
    from sla_modules.omni_sla_manager import OmniSLAManager
except ImportError:
    try:
        from .sla_modules.omni_sla_manager import OmniSLAManager
    except:
        OmniSLAManager = None

logger = logging.getLogger("SLA_P1")

class OmniSLA_P1_Labeler_V1_5:
    def __init__(self):
        # 初始化管理器
        self.manager = OmniSLAManager() if OmniSLAManager else None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",), 
                "model_type": (["v15", "xl"],),
                "physics_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
            },
            # 彻底隐藏 raw_payload 接口，防止旧工作流数据干扰
            "hidden": {
                "raw_payload": "STRING", 
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("MODEL", "POSITIVE", "NEGATIVE")
    FUNCTION = "encode"
    CATEGORY = "SLA_System/Logic"

    def encode(self, clip, model, model_type, physics_strength, raw_payload=None):
        # 1. 物理载荷自动解析
        if self.manager:
            payload = self.manager.get_physical_payload(raw_payload, model_type=model_type)
            id_list = payload.get("id_list", ["Neutral"])
        else:
            id_list = ["Neutral"]
        
        neg_id_list = ["Neutral"]

        # 2. CLIP 协议透传与张量对齐 (修复 'Tensor' object has no attribute 'get' 关键区)
        device = model.load_device if hasattr(model, "load_device") else "cuda"
        
        try:
            tokens = clip.tokenize("")
            # 这里的 ret 在不同版本下可能是 tuple (cond, dict) 或直接是张量
            ret = clip.encode_from_tokens(tokens, return_pooled=True)
            
            # --- 鲁棒性解析结构 ---
            if isinstance(ret, tuple):
                cond_from_clip = ret[0]
                pooled_info = ret[1]
            else:
                cond_from_clip = ret
                pooled_info = None

            # --- 修复点：安全提取池化张量 ---
            if isinstance(pooled_info, dict):
                real_pooled = pooled_info.get("pooled_output", None)
            elif isinstance(pooled_info, torch.Tensor):
                real_pooled = pooled_info
            else:
                real_pooled = None
                
            base_tensor = cond_from_clip 
            
        except Exception as e:
            logger.error(f"⚠️ [SLA-P1] CLIP 协议透传失败，进入降级模式: {e}")
            # 降级：根据模型事实强制分配维度
            dim = 2048 if model_type == "xl" else 768
            base_tensor = torch.zeros((1, 77, dim), device=device)
            real_pooled = None

        # 3. 封装 Conditioning (注入 ZZ 协议数据)
        def build_cond(ids, is_pos=True):
            meta = {
                "sla_ids": ids,
                "prompt": ",".join(ids),
                "model_type": model_type
            }
            if is_pos: 
                meta["strength"] = physics_strength
            
            # 为 SDXL 注入必须的池化张量
            if real_pooled is not None:
                meta["pooled_output"] = real_pooled.to(device)
            
            return [[base_tensor.to(device), meta]]

        pos_cond = build_cond(id_list, True)
        neg_cond = build_cond(neg_id_list, False)

        # 4. 状态注入：在 Model 线上挂载物理指纹
        new_model = model.clone()
        m_opts = new_model.model_options
        if "sla_payload" not in m_opts:
            m_opts["sla_payload"] = {}
        
        m_opts["sla_payload"].update({
            "base_strength": physics_strength,
            "model_type": model_type,
            "id_list": id_list
        })

        return (new_model, pos_cond, neg_cond)