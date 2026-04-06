import os
import sys

# 路径锚定
NODE_DIR = os.path.dirname(os.path.realpath(__file__))
if NODE_DIR not in sys.path:
    sys.path.append(NODE_DIR)

try:
    # 导入我们重构后的 P4 执行器类
    from .Zenith_Custom import SLA_P4_Physical_Reproduction
    
    NODE_CLASS_MAPPINGS = {
        "SLA_P4_Physical_Reproduction": SLA_P4_Physical_Reproduction
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "SLA_P4_Physical_Reproduction": "⚙️ SLA P4: Physical Reproduction (Zenith)"
    }
    
    print("✅ [SLA-System] P4 注册成功 | 物理复刻协议已激活")

except Exception as e:
    import logging
    logging.error(f"🚨 [SLA-P4] 注册失败，请检查 Zenith_Custom.py 代码逻辑: {e}")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']