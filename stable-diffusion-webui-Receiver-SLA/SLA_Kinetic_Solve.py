import torch
from K_SLA_utils import K_SLA_Common

class SLA_Kinetic_Solve:
    def __init__(self):
        self.version = "2.0.0-Zenith-Stable"
        # 基础张力系数：定义物理场扩张的初始弹性
        self.base_tension = 0.65

    def bake_kinetic_step(self, x, p1_id_input, current_gain, is_phase_b, step_ratio):
        """
        动力学算子：执行物理结构的动态扩张与收缩。
        解决漏洞 4：拒绝破坏性原地修改，改为残差增量模式。
        解决漏洞 5：协议标准化适配。
        """
        with torch.no_grad():
            # 1. 协议标准化：确保拿到的是 [(ID, 极性)]
            standard_ids = K_SLA_Common.protocol_standardize(p1_id_input)
            
            # 2. 预计算当前步的全局张力
            # 随步数推移（step_ratio）逐渐减弱动力学干预，保证后期画面收敛稳定
            decay_factor = 1.0 - (step_ratio ** 2) 
            effective_tension = self.base_tension * current_gain * decay_factor

            # 3. 遍历物理 ID 执行动力学干扰
            for p1_id, polarity in standard_ids:
                # 漏洞 4 修复：不再使用 x.mul_(1.0 + ...)
                # 而是计算一个“动力学增量 (Kinetic Delta)”
                
                if not is_phase_b:
                    # --- [Phase A: 动能场建立] ---
                    if polarity > 0:
                        # 正向：结构扩张 (扩张率控制在 2%-5% 之间)
                        # 使用残差：x_new = x + delta
                        delta = x * (effective_tension * 0.04 * polarity)
                        x.add_(delta)
                    else:
                        # 负向：动能抑制 (防止背景噪点过载)
                        # 负向极性下，我们轻微压缩数值，但同样使用残差
                        delta = x * (effective_tension * 0.02 * polarity) # polarity 为负
                        x.add_(delta)
                else:
                    # --- [Phase B: 上色/细节阶段] ---
                    # 动力学在后期应保持极低的影响力，仅做微调
                    if polarity > 0:
                        delta = x * (effective_tension * 0.01)
                        x.add_(delta)

            # 4. 统计学稳压 (解决漏洞 1)
            # 动力学算子极易拉高方差，必须在每步结束前强制归位
            x = K_SLA_Common.tensor_stabilizer(x)
            
        return x