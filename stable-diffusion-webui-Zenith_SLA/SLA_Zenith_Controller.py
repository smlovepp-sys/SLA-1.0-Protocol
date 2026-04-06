"""
SLA_Zenith_Controller.py - V1.0 Origin
📡 [SLA-P2-Core]:
1. 参数注入：负责将 Seed, CFG, Sampler 等物理参数注入 MODEL 载荷。
2. 逻辑解耦：不再处理任何 Conditioning（提示词）流，保持管线纯净。
3. 状态锚定：为 P4 提供执行采样的物理环境配置。
"""
import torch
import copy
import hashlib
import comfy.samplers 

class SLA_Zenith_Controller_V1_0:
    @classmethod
    def INPUT_TYPES(s):
        # 接口回归：只接收模型和基础采样参数
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ), 
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    # 输出保持纯净，转发模型和物理参数
    RETURN_TYPES = ("MODEL", "INT", "FLOAT", "INT", "COMBO", "COMBO")
    RETURN_NAMES = ("MODEL", "steps", "cfg", "seed", "sampler_name", "scheduler")
    FUNCTION = "sync_zenith_context"
    CATEGORY = "SLA/V1.0_Sync"

    def sync_zenith_context(self, model, steps, cfg, seed, sampler_name, scheduler, denoise):
        # 克隆模型以防污染原始管线
        new_model = model.clone()
        m_opts = new_model.model_options.copy()

        # 1. 构建采样上下文元数据 (仅包含物理参数，无提示词)
        sampling_meta = {
            "seed": seed,
            "cfg": cfg,
            "steps": steps,
            "denoise": denoise,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "protocol": "V1.0_Zenith_Origin"
        }

        # 2. 注入核心协议通道，供 P4 认领物理参数
        m_opts["sla_sampling_metadata"] = sampling_meta
        new_model.model_options = m_opts

        return (new_model, steps, cfg, seed, sampler_name, scheduler)