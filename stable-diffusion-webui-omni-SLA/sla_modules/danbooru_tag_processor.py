import json
import os
import logging
import re

logger = logging.getLogger("SLA_P1")

class DanbooruTagProcessor:
    def __init__(self):
        # 1. 定义物理属性探测器（用于对 ZZ 系列 ID 进行强制“染色”）
        self.phys_heuristics = {
            'F': r"dress|cloth|skirt|shirt|fabric|apron|suit|bra|pant|garment|wear",
            'E': r"hair|fringe|bangs|fluid|water|fire|smoke|magic|liquid|gas",
            'B': r"body|muscle|skin|thigh|breast|belly|hand|arm|leg|face|eye",
            'R': r"bone|skeleton|joint|structure|frame|mecha"
        }
        
        self.module_path = os.path.dirname(os.path.realpath(__file__))
        self.base_path = os.path.dirname(self.module_path)
        
        # 预加载双表
        self.v15_map = self._load_map("p1_map_v15.json")
        self.xl_map = self._load_map("p1_map_xl.json")

    def _load_map(self, file_name):
        """修复变量名错误并支持双索引"""
        json_path = os.path.join(self.base_path, file_name)
        cleaned_data = {}
        
        if not os.path.exists(json_path):
            logger.warning(f"⚠️ [SLA-P1] 映射表缺失: {json_path}")
            return {}

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                
            for token_id, info in raw_data.items():
                if isinstance(info, dict):
                    # 统一使用 tag_str 变量
                    tag_str = info.get("tag", "").lower().strip()
                    raw_id = info.get("id", "ZZ")
                    
                    assigned_phys = "Z"
                    # 启发式染色
                    for phys_code, pattern in self.phys_heuristics.items():
                        if re.search(pattern, tag_str):
                            assigned_phys = phys_code
                            break
                    
                    # ID 前缀补充
                    if assigned_phys == "Z" and len(raw_id) > 0 and raw_id[0] in ["R", "B", "F", "E"]:
                        assigned_phys = raw_id[0]
                    
                    # --- 注入双向索引 ---
                    # 索引 A：数字 ID (适配旧逻辑)
                    cleaned_data[str(token_id)] = assigned_phys
                    # 索引 B：标签文本 (适配 XL 和 隐藏 UI 后的语义匹配)
                    if tag_str and tag_str not in cleaned_data:
                        cleaned_data[tag_str] = assigned_phys
            
            logger.info(f"✅ [SLA-P1] 物理映射激活: {file_name} (有效条目: {len(cleaned_data)})")
            return cleaned_data
        except Exception as e:
            logger.error(f"🚨 [SLA-P1] 加载异常 ({file_name}): {e}")
            return {}

    def get_id_type(self, item, model_type="v15"):
        """item 既可以是数字 ID 也可以是标签字符串"""
        target_map = self.xl_map if model_type == "xl" else self.v15_map
        key = str(item).strip().lower()
        return target_map.get(key, "Z")