import os
import re
import gc
import torch

class K_SLA_Common:
    """
    SLA 跨模块通用工具箱 - V2.0 Zenith
    """
    
    @staticmethod
    def protocol_standardize(p1_id_input):
        """
        统一协议校验：强制将物理 ID 转换为格式 [(str, float)]。
        解决漏洞 5：消除 Anchor 和 Adaptor 对 ID 结构认知的差异。
        """
        standard_list = []
        
        # 处理字符串输入
        if isinstance(p1_id_input, str):
            standard_list = [(p1_id_input, 1.0)]
        
        # 处理列表输入
        elif isinstance(p1_id_input, list):
            for item in p1_id_input:
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    # 已经是标准格式 (ID, 极性)
                    standard_list.append((str(item[0]), float(item[1])))
                else:
                    # 只有字符串，补齐默认极性
                    standard_list.append((str(item), 1.0))
        
        # 兜底：防止 None 或非法对象导致崩溃
        if not standard_list:
            standard_list = [("Neutral", 1.0)]
            
        return standard_list

    @staticmethod
    def tensor_stabilizer(x, target_std=0.18215):
        """
        物理稳压器：在物理算子原地修改后，强制将张量拉回正常的统计分布。
        解决漏洞 1：防止方差坍塌导致的“灰块”生成。
        """
        with torch.no_grad():
            curr_mean = x.mean()
            curr_std = x.std()
            
            # 极低方差保护，防止除以零
            if curr_std < 1e-5:
                return x 
            
            # 重新归一化至 SD 官方 Latent 标准分布
            x.sub_(curr_mean).div_(curr_std).mul_(target_std)
        return x

    @staticmethod
    def deep_garbage_collect():
        """执行显存与内存的双重清理"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()