"""ConvGRU 时序融合模块"""
import torch, torch.nn as nn


class ConvGRUCell(nn.Module):
    """单步 ConvGRU 单元，在特征图空间维做卷积门控。
    
    input: (B, C, H, W) feature map
    hidden: (B, C, H, W) hidden state
    """
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.channels = channels
        pad = kernel_size // 2
        self.reset_conv = nn.Conv2d(channels * 2, channels, kernel_size, padding=pad)
        self.update_conv = nn.Conv2d(channels * 2, channels, kernel_size, padding=pad)
        self.candidate_conv = nn.Conv2d(channels * 2, channels, kernel_size, padding=pad)
        # 初始化策略：初始保持 key 特征不变
        # update gate: 负偏置 → z≈0 → new_h≈h (保留历史), 训练后逐渐学会接受 ref
        # reset gate: 零偏置 → r≈0.5 → 初始时 key 和 ref 各一半影响
        # candidate: 正常初始化
        nn.init.kaiming_normal_(self.reset_conv.weight, mode='fan_out', nonlinearity='sigmoid')
        nn.init.zeros_(self.reset_conv.bias)
        nn.init.kaiming_normal_(self.update_conv.weight, mode='fan_out', nonlinearity='sigmoid')
        nn.init.constant_(self.update_conv.bias, -4.0)  # sigmoid(-4)≈0.018 → 保留 98% key
        nn.init.kaiming_normal_(self.candidate_conv.weight, mode='fan_out', nonlinearity='tanh')
        nn.init.zeros_(self.candidate_conv.bias)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        """单步 ConvGRU。
        
        Args:
            x: (B,C,H,W) 当前帧特征
            h: (B,C,H,W) 历史状态
        Returns:
            new_h: (B,C,H,W) 更新后状态
        """
        cat = torch.cat([h, x], dim=1)  # (B, 2C, H, W)
        r = torch.sigmoid(self.reset_conv(cat))       # reset gate
        z = torch.sigmoid(self.update_conv(cat))      # update gate
        candidate = torch.tanh(self.candidate_conv(torch.cat([r * h, x], dim=1)))
        new_h = (1 - z) * h + z * candidate
        return new_h


class ConvGRUFusion(nn.Module):
    """三帧 ConvGRU 融合模块。
    
    将 DAV2 对齐后的 ref1、ref2 特征依次通过 ConvGRU 融合到 key 特征中。
    输入: key, ref1_aligned, ref2_aligned (各 B,C,H,W)
    输出: fused (B,C,H,W)
    
    时序: h0 = key → (h0, ref1) → h1 → (h1, ref2) → h2 = output
    """
    def __init__(self, channels: int, kernel_size: int = 3, num_layers: int = 1):
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers
        
        # 多层 ConvGRU (每层独立细胞)
        self.cells = nn.ModuleList()
        for _ in range(num_layers):
            self.cells.append(ConvGRUCell(channels, kernel_size))
        
        # 最终 1×1 投影，小权重 + 零偏置 → 初始时输出≈key+小扰动
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='linear')
        nn.init.zeros_(self.proj.bias)
        
        # 输入投影: ref 帧先投影到合适的空间
        self.ref1_proj = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels))
        self.ref2_proj = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels))
        for m in [self.ref1_proj, self.ref2_proj]:
            if isinstance(m[0], nn.Conv2d):
                nn.init.kaiming_normal_(m[0].weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m[0].bias)

    def forward(self, key: torch.Tensor, ref1: torch.Tensor, ref2: torch.Tensor):
        """ConvGRU 时序融合。
        
        Args:
            key:   (B,C,H,W) 关键帧特征 (P4)
            ref1:  (B,C,H,W) DAV2对齐后的ref1特征
            ref2:  (B,C,H,W) DAV2对齐后的ref2特征
        
        Returns:
            fused: (B,C,H,W) 融合特征
        """
        r1 = self.ref1_proj(ref1)
        r2 = self.ref2_proj(ref2)
        
        h = key  # 以key帧状态初始化
        
        # 第一层: ref1 → ref2
        for cell in self.cells:
            h = cell(r1, h)  # 先用ref1更新
            h = cell(r2, h)  # 再用ref2更新
        
        # 残差投影
        out = key + self.proj(h)
        return out


def get_gate_weights_for_convgru(self, key, ref1, ref2):
    """ConvGRU模式下不需要gate权重（权重组装兼容接口）"""
    B, C, H, W = key.shape
    return (
        torch.ones(B, 1, H, W, device=key.device) * 0.5,
        torch.ones(B, 1, H, W, device=key.device) * 0.5,
    )
