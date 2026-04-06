class SLA_Style_Node:
    def __init__(self):
        self.style_name = "Ultra_Realism_V1.0_Origin"

    def get_config(self, seed=0):
        # 引入随机物理位移，增加生物演化感
        stochastic_shift = (seed % 50) / 500.0
        return {
            "offset": 12, 
            "p_mult": 1.1 + stochastic_shift,
            "whiplash_bias": 0.7,  # 留出 30% 步数给结晶渲染
            "crystallize": 1.25,   # 增强 Phase B 的法线深度
            "logic_mode": "mult"   # 修正拼写：multi -> mult
        }