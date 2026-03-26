class SLA_RainyNight_Shard(nn.Module):
    """
    SLA 1.1-Beta 专用：雨夜环境逻辑碎片
    核心：强制执行地面法线与光源倒影的几何对称性
    """
    def forward(self, source_light_map, reflection_map):
        # 1. 执行空间翻转 (Vertical Flip)
        # 根据 SLA 1.0 物理先验，倒影必须是光源在地面高度下的镜像
        expected_reflection = torch.flip(source_light_map, dims=[2]) 
        
        # 2. 逻辑对齐损耗 (Reference Alignment Loss)
        # 强制 AI 生成的倒影与物理镜像重合
        alignment_loss = F.mse_loss(reflection_map, expected_reflection)
        
        return alignment_loss * 20.0 # 高权重干预