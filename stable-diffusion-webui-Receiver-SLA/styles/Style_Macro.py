class SLA_Style_Node:
    def __init__(self):
        self.style_name = "Macro_Detail_V1.0_Origin"

    def get_config(self, seed=0):
        return {
            "offset": 25, 
            "p_mult": 1.5,
            "whiplash_bias": 0.6,  # 早期截断以固化微观结构
            "crystallize": 1.8,    # 爆发式亚像素注入
            "logic_mode": "mult"   # 极致乘法增益
        }