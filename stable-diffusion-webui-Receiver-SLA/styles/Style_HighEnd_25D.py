class SLA_Style_Node:
    def __init__(self):
        self.style_name = "HighEnd_25D_V1.0_Origin"

    def get_config(self, seed=0):
        return {
            "offset": 22, 
            "p_mult": 1.05,
            "whiplash_bias": 0.7, 
            "crystallize": 1.1,    # 轻微强化材质感
            "logic_mode": "mult"
        }