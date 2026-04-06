class SLA_Style_Node:
    def __init__(self):
        self.style_name = "Anime_Flat_V1.0_Origin"

    def get_config(self, seed=0):
        return {
            "offset": 5, 
            "p_mult": 0.8,
            "whiplash_bias": 0.85, # 极晚截断，保持线条稳定
            "crystallize": 0.5,    # 抑制 Phase B 的质感结晶
            "logic_mode": "div"    # 除法模式：净化色块
        }