import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelMixer(nn.Module):
    """Memory-efficient channel correlation module."""
    
    def __init__(self, d_model, c_out, n_groups=16, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.c_out = c_out
        self.n_groups = min(n_groups, c_out)  # Handle C < n_groups
        
        # Learnable channel embeddings
        self.channel_embed = nn.Embedding(c_out, d_model)
        
        # Group-wise projection: from [B, L, d_model] -> [B, L, c_out]
        # Use low-rank factorization
        rank = min(d_model // 2, c_out // 2, 64)
        self.fc_compress = nn.Linear(d_model, rank)
        self.fc_expand = nn.Linear(rank, c_out)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [B, L, d_model] - encoded features
            mask: [B, L, C] - optional missing mask
        
        Returns:
            out: [B, L, c_out]
        """
        B, L, d_model = x.shape
        
        # Normalize input
        x_norm = self.norm(x)  # [B, L, d_model]
        
        # Low-rank projection
        x_compressed = self.fc_compress(x_norm)  # [B, L, rank]
        x_compressed = F.gelu(x_compressed)
        x_out = self.fc_expand(x_compressed)  # [B, L, c_out]
        x_out = self.dropout(x_out)
        
        # If mask provided, mask missing channels
        if mask is not None:
            # mask: [B, L, C] -> average mask over L for channel-level masking
            channel_mask = mask.mean(dim=1)  # [B, C]
            x_out = x_out * channel_mask.unsqueeze(1)  # broadcast
        
        return x_out  # [B, L, c_out]
