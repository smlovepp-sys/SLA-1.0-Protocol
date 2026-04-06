import torch
import os
import gc
from collections import OrderedDict
from datetime import datetime

class SLA_Latent_Vault:
    def __init__(self, vault_path="physics_vault", max_files=2000, max_mem_slots=6):
        self.vault_path = vault_path
        self.max_files = max_files
        self.max_mem_slots = max_mem_slots
        
        # 确保路径存在
        os.makedirs(self.vault_path, exist_ok=True)
        
        # 核心修复：使用 OrderedDict 记录存入顺序，用于 LRU 内存淘汰
        self.memory_slots = OrderedDict()

    def save_latent(self, index, tensor, seed, cfg, text_id):
        """
        保存 128 级物理张量。
        解决漏洞 2：通过 popitem(last=False) 强制移除最早的张量，释放 RAM。
        """
        # 1. 内存阈值检查
        if len(self.memory_slots) >= self.max_mem_slots:
            # 弹出最旧的插槽数据
            _, old_tensor = self.memory_slots.popitem(last=False)
            del old_tensor
            # 配合 gc 确保系统真正回收内存
            gc.collect()

        # 2. 存入内存 (强制转换至 CPU 并 detach，切断所有潜在的隐式梯度链)
        self.memory_slots[index] = tensor.detach().cpu()

        # 3. 硬盘持久化逻辑 (保持原结构，但进行安全检查)
        self._check_limit()
        try:
            fingerprint = torch.hub._hash_tensor(tensor)[:8]
            date_str = datetime.now().strftime("%m%d")
            file_name = f"slot{index}_{seed}_{cfg}_{date_str}_{fingerprint}.pt"
            save_path = os.path.join(self.vault_path, file_name)
            # 生产环境建议按需开启硬盘保存，默认仅驻留内存以保速
            # torch.save(self.memory_slots[index], save_path) 
        except Exception as e:
            print(f"⚠️ [SLA-Vault] 硬盘备份跳过: {e}")

    def _check_limit(self):
        """自动清理过期磁盘文件，防止硬盘占满"""
        try:
            files = [os.path.join(self.vault_path, f) for f in os.listdir(self.vault_path) if f.endswith('.pt')]
            if len(files) > self.max_files:
                files.sort(key=os.path.getmtime)
                for i in range(len(files) - self.max_files):
                    os.remove(files[i])
        except Exception:
            pass

    def get_latent(self, index):
        """供 P4 安全调用"""
        return self.memory_slots.get(index)