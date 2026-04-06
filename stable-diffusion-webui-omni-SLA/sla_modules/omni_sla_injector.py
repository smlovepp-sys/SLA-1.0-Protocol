"""
omni_sla_injector.py - V1.0 Zenith-Pure (Restructured)
功能：模型元数据（Metadata）安全注入。
职责：
1. 接收解析后的 ID 列表和压强数值。
2. 修复协议键名，确保与 P3 Receiver Hub 完美对接。
[注意]：保留原 protocol_version 避免注册失效。
"""
import torch
import logging

logger = logging.getLogger("SLA_P1")

class OmniSLAInjector:
    def __init__(self):
        # 严禁修改此版本号，否则会触发 P1 注册拦截
        self.protocol_version = "1.0_ZENITH_PURE"

    def inject_metadata(self, model, optimized_payload: dict, physics_strength: float):
        """
        核心任务：将 P1 的扫描结果注入模型实例，供后续 P3/P4 节点读取。
        """
        try:
            # 1. 安全执行模型克隆，避免污染原始模型引用
            new_model = model.clone()
            m_opts = new_model.model_options.copy()

            # 2. 封装物理锦囊
            # 这里的结构必须严格对齐 P3 的解析逻辑
            zenith_metadata = {
                "id_list": optimized_payload.get("id_list", []),
                "entity_count": optimized_payload.get("entity_count", 1),
                "base_strength": physics_strength,
                "vram_status": optimized_payload.get("vram_status", "UNKNOWN"),
                "protocol": self.protocol_version
            }

            # 3. 【核心修复】：协议对齐
            # 将旧版的 sla_physics_metadata 修改为 P3 唯一识别的 sla_payload
            m_opts["sla_payload"] = zenith_metadata
            
            # 同时保留压强基准映射，防止某些旧版算子报错
            m_opts["sla_base_pressure"] = physics_strength

            # 4. 更新模型选项并回传
            new_model.model_options = m_opts
            
            logger.info(f"✅ [SLA-Injector] 物理协议注入成功: {len(zenith_metadata['id_list'])} IDs")
            return new_model

        except Exception as e:
            logger.error(f"🚨 [SLA-Injector] 注入过程崩溃: {e}")
            return model # 发生异常时返回原模型，防止整个工作流中断