class SLA_Style_Node:
    def __init__(self):
        self.style_name = "Sculpt_3D_V1.0_Origin"

    def get_config(self, seed=0):
        return {
            "offset": 35, 
            "p_mult": 1.15,
            "whiplash_bias": 0.5,  # 50% 步数即锁定结构，防止坍塌
            "crystallize": 1.4,    # 强化硬质表面反射
            "logic_mode": "mult"
        }