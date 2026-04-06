class SLA_Style_Node:
    def __init__(self):
        self.style_name = "OVA_Classic_V1.0_Origin"

    def get_config(self, seed=0):
        return {
            "offset": 8, 
            "p_mult": 0.85,
            "whiplash_bias": 0.8,
            "crystallize": 0.6,
            "logic_mode": "div"    # 除法模式保持画面通透
        }