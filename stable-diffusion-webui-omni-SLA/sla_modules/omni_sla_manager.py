"""
omni_sla_manager.py - V1.2.5 Zenith-Executive (Stability Update)
功能：物理 ID 翻译与事实分发中心。
职责：
1. 修复了 torch.hub._hash_tensor 缺失导致的兼容性报错。
2. 适配 UI 隐藏模式，增强了对空载荷（None/Empty）的防御。
3. 确保 XL/v15 模式下的物理实体识别准确性。
"""
import os
import re
import logging
import hashlib
from typing import List, Dict, Any

# 路径自愈导入逻辑
try:
    from .danbooru_tag_processor import DanbooruTagProcessor
except ImportError:
    from danbooru_tag_processor import DanbooruTagProcessor

logger = logging.getLogger("SLA_P1")

class OmniSLAManager:
    def __init__(self):
        # 挂载具备双向索引能力的处理器
        self.processor = DanbooruTagProcessor()
        self._reset_state()

    def _reset_state(self):
        """状态重置，防止跨批次数据污染"""
        self.current_found_ids = []
        self.last_payload_hash = None

    def _get_safe_hash(self, data: Any) -> str:
        """
        替代已失效的 torch.hub._hash_tensor。
        使用 hashlib 对输入内容生成唯一指纹。
        """
        try:
            hash_machine = hashlib.sha256()
            hash_machine.update(str(data).encode('utf-8'))
            return hash_machine.hexdigest()
        except:
            return "hash_failed"

    def get_physical_payload(self, input_data: Any, model_type: str = "v15") -> Dict[str, Any]:
        """
        核心调度方法：
        1. 接收来自 UI（即使已隐藏）或上游的原始数据。
        2. 自动判定输入格式（Token ID 或 标签文本）。
        3. 翻译并分发物理载荷。
        """
        self._reset_state()
        seen_physical_ids = [] 

        # --- 1. 数据清洗与空值容错 ---
        # 修复：当 UI 隐藏 raw_payload 时，input_data 可能为 None
        if input_data is None or str(input_data).strip() == "":
            # 隐藏模式下，如果不连线，默认返回中性场
            search_list = ["neutral"]
        elif isinstance(input_data, list):
            # 模式 A：Token ID 流模式
            search_list = [str(x) for x in input_data]
        else:
            # 模式 B：文本正则/字符串模式 (适配隐藏 UI 后的工作流反序列化)
            search_list = re.findall(r'\[(.*?)\\]', str(input_data))
            if not search_list:
                # 尝试对整个字符串进行逗号分词
                search_list = str(input_data).split(',')

        # --- 2. 核心翻译阶段 ---
        for item in search_list:
            item = item.strip().lower()
            if not item:
                continue
                
            # 向处理器请求 ID。processor 现在支持同时查找 ID (如 "1024") 和 标签 (如 "dress")
            phys_id = self.processor.get_id_type(item, model_type=model_type)
            
            # 只有非中性 (R/B/F/E) 的 ID 才会进入物理流
            if phys_id != "Z" and phys_id not in seen_physical_ids:
                seen_physical_ids.append(phys_id)

        # --- 3. 兜底逻辑 ---
        if not seen_physical_ids:
            seen_physical_ids = ["Neutral"]

        # --- 4. 封装载荷 ---
        # 这里的结构必须对应下游 P3 Injector 的预期
        result = {
            "id_list": seen_physical_ids,
            "entity_count": len(seen_physical_ids),
            "model_type": model_type,
            "vram_status": "NORMAL",
            "hash": self._get_safe_hash(input_data)
        }

        # 打印关键日志，方便在控制台确认物理场是否激活
        logger.info(f"🧬 [SLA-Manager] 逻辑解析完成 | 模式: {model_type} | 物理实体: {seen_physical_ids}")
        
        return result

    def validate_payload(self, payload: Dict[str, Any]) -> bool:
        """验证载荷完整性，防止非法篡改导致后续算子崩溃"""
        required_keys = ["id_list", "model_type"]
        return all(k in payload for k in required_keys)