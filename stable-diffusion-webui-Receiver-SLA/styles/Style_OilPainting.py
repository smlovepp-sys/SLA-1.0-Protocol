class SLA_Style_Node:
    def __init__(self):
        self.style_name = "Oil_Painting_V1.0_Origin"

    def get_config(self, seed=0):
        return {
            "offset": 10, 
            "p_mult": 0.9,
            "whiplash_bias": 0.9,  # 极晚截断，允许物理场抖动
            "crystallize": 0.7,    # 弱化法线，保留平涂感
            "logic_mode": "div"
        }