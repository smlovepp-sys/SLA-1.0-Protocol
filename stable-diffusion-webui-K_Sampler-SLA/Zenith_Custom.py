import os
import sys
import torch
import nodes 
import comfy.utils # 导入工具包以支持 SDXL

NODE_DIR = os.path.dirname(os.path.realpath(__file__))
if NODE_DIR not in sys.path:
    sys.path.append(NODE_DIR)

class SLA_P4_Physical_Reproduction:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",), 
                "physics_bus": ("SLA_PHYSICS",), 
                "latent_image": ("LATENT",),     
            }
        }

    RETURN_TYPES = ("LATENT", "STATUS")
    RETURN_NAMES = ("Latent", "STATUS")
    FUNCTION = "execute_reproduction"
    CATEGORY = "SLA/Logic"

    def execute_reproduction(self, model, physics_bus, latent_image):
        # 1. 协议拆包
        s_pkg = physics_bus.get("sketch_package")
        pos = physics_bus.get("pos_id_cond")
        neg = physics_bus.get("neg_id_cond")
        
        # --- 🕵️‍♂️ 核心修正：从紫线 (MODEL) 提取 P2 实时参数 ---
        m_opts = getattr(model, "model_options", {})
        sla_meta = m_opts.get("sla_sampling_metadata", {})
        
        phys_seed = sla_meta.get("seed", physics_bus.get("phys_seed", 0))
        active_cfg = sla_meta.get("cfg", 7.0)
        active_steps = sla_meta.get("steps", 20)
        sampler_name = sla_meta.get("sampler_name", "euler")
        scheduler = sla_meta.get("scheduler", "simple")

        device = model.load_device
        samples = latent_image["samples"].to(device)

        # 2. 物理残差稳压注入
        if len(s_pkg) > 0:
            last_frame = s_pkg[-1].to(device)
            # 使用残差模式叠加，并进行轻微稳压
            samples = samples + (last_frame * 0.05) 

        print(f"🎯 [SLA-P4] 物理复刻启动 | 种子: {phys_seed} | 压强同步: {active_cfg}")
        
        try:
            # --- 🛡️ 稳健采样逻辑 (解决 pooled_output 崩溃) ---
            # 使用 comfy.utils.common_upscale 的类似思路，确保 cond 完整性
            out_latent = nodes.common_ksampler(
                model, 
                phys_seed, 
                active_steps, 
                active_cfg, 
                sampler_name, 
                scheduler, 
                pos, 
                neg, 
                {"samples": samples}, 
                denoise=0.65
            )[0]

            return (out_latent, f"SUCCESS: ALIGNED-{active_cfg}")

        except Exception as e:
            # 针对 SDXL 的特定错误捕获
            error_msg = str(e)
            if "pooled_output" in error_msg:
                error_msg = "SDXL池化校验失败：请确保 P1 节点的 CLIP 正确连接并输出了有效 ID。"
            print(f"❌ [SLA-P4] 采样崩溃: {error_msg}")
            return (latent_image, f"ERROR: {error_msg}")

NODE_CLASS_MAPPINGS = {"SLA_P4_Physical_Reproduction": SLA_P4_Physical_Reproduction}