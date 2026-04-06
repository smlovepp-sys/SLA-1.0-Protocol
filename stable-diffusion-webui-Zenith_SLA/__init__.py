"""
P2/__init__.py - V1.0 Origin
📡 [SLA-P2] 模块挂载中心：将 V1.0 纯净参数控制器注册至 ComfyUI。
重构点：
1. 接口回归：彻底移除 positive/negative 提示词输入点，回归 V1.0 极简参数流。
2. 逻辑对齐：指向仅处理物理采样参数（Seed, CFG, Sampler）的 V1.0 类。
3. 强制刷新：更新注册 KEY 为 V1_0_Origin，确保 UI 彻底清除旧版 Payload 接口。
"""
import os
import sys

# 1. 环境自愈：确保 P2 核心组件在 Python 搜索路径中
NODE_DIR = os.path.dirname(os.path.realpath(__file__))
if NODE_DIR not in sys.path:
    sys.path.append(NODE_DIR)

try:
    # 2. 导入回归后的 V1.0 核心控制器类
    from .SLA_Zenith_Controller import SLA_Zenith_Controller_V1_0

    # 3. 定义节点映射
    # 💡 使用 _V1_0_Origin 后缀以强制 ComfyUI 刷新节点外观，移除正负提示词紫色圆点
    NODE_CLASS_MAPPINGS = {
        "SLA_P2_Zenith_Controller_V1_0_Origin": SLA_Zenith_Controller_V1_0
    }

    # 4. 定义 UI 显示名称
    NODE_DISPLAY_NAME_MAPPINGS = {
        "SLA_P2_Zenith_Controller_V1_0_Origin": "📡 SLA P2: Zenith Controller (V1.0 Origin)"
    }
    
    print("✅ [SLA-System] P2 控制器 (V1.0 Origin) 挂载成功 | 物理参数流已锁定")

except Exception as e:
    import logging
    logging.error(f"🚨 [SLA-P2] 注册失败，请检查 SLA_Zenith_Controller.py: {e}")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']