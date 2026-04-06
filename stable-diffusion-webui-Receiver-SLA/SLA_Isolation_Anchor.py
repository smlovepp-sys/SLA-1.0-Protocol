import torch
from K_SLA_utils import K_SLA_Common

class SLA_Isolation_Anchor:
    def __init__(self):
        self.version = "2.0.0-Zenith-Stable"
        self.base_suppression = 0.85
        
        # 核心物理主权权重 (与 P1 协议标准化对齐)
        self.attr_weights = {
            'R': 1.65,      # Root: 骨架结构
            'B': 1.45,      # Body: 生物肌肉
            'F': 1.15,      # Fabric: 布料纹理
            'E': 1.05,      # Element: 流体/粒子
            'V': 1.20,      # Vital: 视觉中心
            'Neutral': 1.0, # 默认平放
        }

    def bake_isolation_step(self, x, p1_id_input, current_gain, is_phase_b):
        """
        接收 Hub 的指令：执行结构稳固与物理隔离。
        解决漏洞 5：通过协议标准化函数处理 p1_id，彻底消除类型报错。
        解决漏洞 1 & 4：引入增量修改模式，保护 ODE 链条。
        """
        with torch.no_grad():
            # 1. 协议标准化：将输入转为统一的 [(ID, 极性)] 列表
            standard_ids = K_SLA_Common.protocol_standardize(p1_id_input)
            
            # 2. 遍历标准 ID 组进行物理干预
            for p1_id, polarity in standard_ids:
                # 安全提取路由标识
                lookup_key = p1_id[0].upper() if p1_id else 'Neutral'
                sovereignty_power = self.attr_weights.get(lookup_key, self.attr_weights['Neutral'])

                # 3. 执行物理压强对冲 (非 Phase B 阶段)
                if not is_phase_b:
                    # 计算当前步的活跃影响力
                    active_influence = current_gain * sovereignty_power * polarity
                    
                    # 漏洞 4 修复：使用残差增量，而非暴力覆盖原有张量
                    # 这种方式允许 ODE 求解器识别出数值的变化趋势，而不是突变
                    if polarity > 0:
                        # 正向：结构稳固增量
                        delta = x * (active_influence * 0.1)
                        x.add_(delta) 
                    else:
                        # 负向：背景抑制增量
                        # 限制抑制强度，防止方差坍塌
                        suppression = max(self.base_suppression, 1.0 - active_influence.abs() * 0.05)
                        x.mul_(suppression)

            # 4. 统计学稳压 (解决漏洞 1)
            # 每执行完一次锚定，强制检查并拉回标准 Latent 分布
            x = K_SLA_Common.tensor_stabilizer(x)
            
        return x