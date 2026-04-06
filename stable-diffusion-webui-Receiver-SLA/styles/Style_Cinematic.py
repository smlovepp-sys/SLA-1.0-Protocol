class SLA_Style_Node:
    def __init__(self):
        self.style_name = "Cinematic_Flow_V1.0_Origin"

    def get_config(self, seed=0):
        return {
            "offset": 20, 
            "p_mult": 1.0,
            "whiplash_bias": 0.75, # 标准分段
            "crystallize": 1.0,    # 标准光影
            "logic_mode": "mult"   # 修正拼写：multi -> mult
        }