import torch
import math
import os
import importlib.util
import logging
from K_SLA_utils import K_SLA_Common
from SLA_Pure_Tensor_Vault import SLA_Latent_Vault

logger = logging.getLogger("SLA_P3")

class SLA_Physical_Commander_Hub:
    def __init__(self):
        # 1. 初始化底层工具与存储金库
        self.vault = SLA_Latent_Vault(max_mem_slots=6)
        self.common = K_SLA_Common()
        
        # 2. 尝试装载三大核心算子
        try:
            from .SLA_Kinetic_Solve import SLA_Kinetic_Solve
            from .SLA_Perceptual_Adaptor import SLA_Perceptual_Adaptor
            from .SLA_Isolation_Anchor import SLA_Isolation_Anchor
            self.kinetic = SLA_Kinetic_Solve()
            self.perceptual = SLA_Perceptual_Adaptor()
            self.anchor = SLA_Isolation_Anchor()
        except Exception as e:
            print(f"⚠️ [SLA-P3] 算子装载异常: {e}")
            self.kinetic = self.perceptual = self.anchor = None

    @classmethod
    def INPUT_TYPES(s):
        styles_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "styles")
        # 自动获取 styles 文件夹下的风格配置文件
        style_files = []
        if os.path.exists(styles_path):
            style_files = [f[:-3] for f in os.listdir(styles_path) if f.endswith(".py") and f != "__init__.py"]
        
        return {
            "required": {
                "model": ("MODEL",),            # 接收 P2 经过加工的模型线 (紫线)
                "latent_inner": ("LATENT",),    # 仅获取维度
                "pos_id_cond": ("CONDITIONING",), # 来自 P1 的物理 ID (红线)
                "neg_id_cond": ("CONDITIONING",), 
                "physical_style": (style_files, {"default": style_files[0] if style_files else ""}),
            },
        }

    RETURN_TYPES = ("SLA_PHYSICS", "LATENT", "MODEL")
    RETURN_NAMES = ("physics_bus", "latent_out", "model")
    FUNCTION = "commander_execute"
    CATEGORY = "SLA/Logic"

    def commander_execute(self, model, latent_inner, pos_id_cond, neg_id_cond, physical_style):
        # --- 1. 侦听 P2 的紫线信号 (核心上下文提取) ---
        m_opts = getattr(model, "model_options", {})
        # 从模型配置中提取 P2 (SLA_Zenith_Controller) 注入的采样元数据
        sla_meta = m_opts.get("sla_sampling_metadata", {})
        
        active_seed = sla_meta.get("seed", 0) 
        active_cfg = sla_meta.get("cfg", 7.0)
        active_steps = sla_meta.get("steps", 20)

        if active_seed == 0:
            print("⚠️ [SLA-P3] 离线警告：紫线上未发现 P2 协议数据，请检查 P2 节点是否位于上游。")

        device = model.load_device 
        batch, channel, height, width = latent_inner["samples"].shape 
        
        # --- 2. 物理场同步 (强制对齐 P2 种子) ---
        torch.manual_seed(active_seed)
        p3_latent = torch.randn((batch, channel, height, width), device=device) * 0.18215
        
        # --- 3. 提取物理 ID (核心修正：精准对接 P1 的 Conditioning 结构) ---
        raw_ids = []
        try:
            for (t, meta) in pos_id_cond:
                if isinstance(meta, dict) and "sla_ids" in meta:
                    found_ids = meta["sla_ids"]
                    if isinstance(found_ids, list):
                        raw_ids.extend(found_ids)
                    else:
                        raw_ids.append(str(found_ids))
        except Exception as e:
            print(f"⚠️ [SLA-P3] ID 协议解析失败: {e}")

        # 如果没有抓到任何 ID，回退到 Neutral
        if not raw_ids:
            raw_ids = ["Neutral"]
            
        standard_ids = self.common.protocol_standardize(raw_ids)
        print(f"🛰️ [SLA-P3] 物理载荷确认: {standard_ids} | 模式: {sla_meta.get('model_type', 'unknown')}")

        # --- 4. 动态加载风格配置 ---
        p_mult = 1.0
        try:
            style_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "styles", f"{physical_style}.py")
            spec = importlib.util.spec_from_file_location("style_node", style_path)
            style_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(style_mod)
            style_cfg = style_mod.SLA_Style_Node().get_config()
            p_mult = style_cfg.get("p_mult", 1.0)
        except Exception as e:
            print(f"⚠️ [SLA-P3] 风格加载失败: {e}")

        # --- 5. 核心物理演化循环 ---
        s_pkg_list = []
        # 预设 20 步演化，模拟物理场生长
        for i in range(20):
            step_ratio = i / 19.0
            is_phase_b = i > 12 
            
            # 压强补偿：物理场强度随 P2 的 CFG 缩放
            current_gain = p_mult * (active_cfg / 7.0) * (1.0 + math.sin(step_ratio * math.pi))

            if self.anchor:
                p3_latent = self.anchor.bake_isolation_step(p3_latent, standard_ids, current_gain, is_phase_b)
            if self.kinetic:
                p3_latent = self.kinetic.bake_kinetic_step(p3_latent, standard_ids, current_gain, is_phase_b, step_ratio)
            if self.perceptual:
                p3_latent = self.perceptual.bake_perceptual_step(p3_latent, standard_ids, current_gain, is_phase_b, step_ratio)

            s_pkg_list.append(p3_latent.detach().cpu() * 0.1)

        # --- 6. 统计学稳压与存储 ---
        p3_latent = self.common.tensor_stabilizer(p3_latent)
        self.vault.save_latent(0, p3_latent, active_seed, active_cfg, "Zenith_Core")

        # --- 7. 总线封装与显存清理 ---
        physics_bus = {
            "sketch_package": s_pkg_list,
            "pos_id_cond": pos_id_cond,
            "neg_id_cond": neg_id_cond,
            "phys_seed": active_seed
        }

        print(f"🛰️ [SLA-P3] 同步 P2 成功 | 种子: {active_seed} | 压强: {active_cfg}")
        
        # 深度垃圾回收，防止物理场演化撑爆显存
        self.common.deep_garbage_collect()

        return (physics_bus, {"samples": p3_latent}, model)

# 节点映射
NODE_CLASS_MAPPINGS = {"SLA_P3_Commander_Hub": SLA_Physical_Commander_Hub}
NODE_DISPLAY_NAME_MAPPINGS = {"SLA_P3_Commander_Hub": "🛰️ SLA P3: Physical Commander Hub"}