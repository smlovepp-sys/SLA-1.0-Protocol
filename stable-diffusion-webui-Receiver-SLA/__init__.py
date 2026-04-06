import os
import sys

# 自动将当前目录加入系统路径，确保 Hub 能找到同目录下的算子文件
NODE_DIR = os.path.dirname(os.path.realpath(__file__))
if NODE_DIR not in sys.path:
    sys.path.append(NODE_DIR)

from .SLA_Receiver_Hub import SLA_Physical_Commander_Hub

# 建立 ComfyUI 节点类映射
NODE_CLASS_MAPPINGS = {
    "SLA_P3_Commander_Hub": SLA_Physical_Commander_Hub
}

# 设置节点在界面菜单中的显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "SLA_P3_Commander_Hub": "🚀 SLA P3: Physical Commander Hub (V2-Zenith)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']