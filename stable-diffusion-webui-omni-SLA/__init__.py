"""
__init__.py - SLA-P1-Logic 标签逻辑中心注册文件
功能：
1. 注册 V1.5.5 重构版 P1 Labeler。
2. 协议升级：通过隐藏式 raw_payload 隔离 UI 与逻辑，防止工作流冲突。
3. 自动同步 XL/v15 物理事实至下游 P3/P4。
"""
import os
import sys
import logging

# 1. 路径锚定：确保 ComfyUI 能够正确导入 sla_modules
NODE_DIR = os.path.dirname(os.path.realpath(__file__))
if NODE_DIR not in sys.path:
    sys.path.append(NODE_DIR)

# 2. 导入重构后的 P1 主节点类
try:
    from .OmniSLAMainNode import OmniSLA_P1_Labeler_V1_5
    
    # --- 核心重构：节点映射表 ---
    # 🛠️ 这里的 Key "OmniSLA_P1_Labeler_V1" 必须保持不变！
    # 这样用户打开旧的工作流时，ComfyUI 会自动调用我们重构后的新逻辑，而不会报“Node Not Found”。
    NODE_CLASS_MAPPINGS = {
        "OmniSLA_P1_Labeler_V1": OmniSLA_P1_Labeler_V1_5
    }
    
    # 前端显示名称映射（在 ComfyUI 节点菜单里看到的名称）
    NODE_DISPLAY_NAME_MAPPINGS = {
        "OmniSLA_P1_Labeler_V1": "🧬 SLA P1: Logical Labeler (Zenith-ZZ)"
    }
    
    print("✅ [SLA-System] P1 标签逻辑中心 (Zenith-ZZ) 注册成功")
    print("   >> 模式: UI 隐藏式索引 / 双表自动切换 (v15 & XL)")

except Exception as e:
    print(f"🚨 [SLA-System] P1 注册失败: {e}")
    # 导出为空字典，防止 ComfyUI 启动因单节点错误而崩溃
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}