import torch
import numpy as np
from K_SLA_utils import K_SLA_Common

class SLA_Perceptual_Adaptor:
    def __init__(self):
        self.version = "2.0.0-Zenith-Stable"
        # 韦伯常数基准（感知阈值），用于控制能量补偿的起步强度
        self.k_base = 0.018

    def bake_perceptual_step(self, x, p1_id_input, current_gain, is_phase_b, step_ratio):
        """
        感知内核：执行非线性稳压与能量对冲。
        解决漏洞 1：废除局部通道切片，改用全通道分布保护。
        解决漏洞 4：引入残差能量对冲，防止方差坍塌。
        """
        with torch.no_grad():
            # 1. 协议标准化：确保拿到格式统一的 [(ID, 极性)]
            standard_ids = K_SLA_Common.protocol_standardize(p1_id_input)
            
            # 2. 统计当前物理场能量差分
            pos_count = sum(1 for _, pol in standard_ids if pol > 0)
            neg_count = sum(1 for _, pol in standard_ids if pol < 0)
            
            # 湮灭因子：模拟正负能量对冲后的残留压强
            annihilation_factor = 1.0 + (neg_count * 0.05) - (pos_count * 0.02)

            # 3. 计算动态感知系数 K
            # 随步数（step_ratio）对数衰减，确保后期上色阶段细节不再波动
            remaining_ratio = 1.0 - step_ratio
            dynamic_k = self.k_base * current_gain * np.log1p(remaining_ratio + 0.05) * annihilation_factor
            
            # 4. 执行能量平衡逻辑 (解决漏洞 1 & 4)
            # 💡 核心改变：不再对 Channel 0 进行 sub_ 减法！
            # 我们通过调整“张量张力”来修正能量，而不是抹平它。
            
            # 记录原始统计量
            orig_mean = x.mean()
            orig_std = x.std()
            
            # 修正强度系数
            comp_factor = dynamic_k * 2.0
            
            if is_phase_b:
                # 在 Phase B (上色阶段)，执行“细节结晶”
                # 我们轻微拉高标准差（对比度），而不是降低它，防止灰化
                crystal_gain = 1.0 + (comp_factor * 0.5)
                x.sub_(orig_mean).mul_(crystal_gain).add_(orig_mean)
            else:
                # 在 Phase A (草图阶段)，执行“能量稳流”
                # 仅做极微小的中心化偏移，不触动方差
                x.sub_(orig_mean * comp_factor * 0.1)

            # 5. 终极统计学稳压 (解决漏洞 1：方差坍塌)
            # 无论前面怎么算，最后强行拉回到标准 Latent 分布 (STD=0.18215)
            # 这一步是保证不出一张“灰板”的生死红线
            x = K_SLA_Common.tensor_stabilizer(x)
            
        return x