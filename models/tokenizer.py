import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskAwareTokenizer(nn.Module):
    """Multi-scale mask-aware tokenizer."""
    
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.c_in = c_in
        self.d_model = d_model
        
        # Multi-scale Conv1D kernels
        self.conv_kernels = nn.ModuleList([
            nn.Conv1d(c_in + 1, d_model, kernel_size=k, padding=k//2)
            for k in [3, 5, 7, 15]
        ])
        
        # Projection and normalization
        self.proj = nn.Linear(d_model * 4, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask, time_features=None):
        """
        Args:
            x: [B, L, C] - time series
            mask: [B, L, C] - missing mask (1=observed, 0=missing)
            time_features: [B, L] or None
        
        Returns:
            tokens: [B, L, d_model]
        """
        B, L, C = x.shape
        
        # Combine x and mask: [B, L, C+1]
        x_mask = torch.cat([x, mask.float()], dim=-1)  # [B, L, C+1]
        x_mask = x_mask.transpose(1, 2)  # [B, C+1, L]
        
        # Apply multi-scale convolutions
        conv_outputs = []
        for conv in self.conv_kernels:
            out = conv(x_mask)  # [B, d_model, L]
            out = F.gelu(out)
            conv_outputs.append(out)
        
        # Concatenate and project: [B, 4*d_model, L]
        tokens = torch.cat(conv_outputs, dim=1)  # [B, 4*d_model, L]
        tokens = tokens.transpose(1, 2)  # [B, L, 4*d_model]
        
        # Project to d_model
        tokens = self.proj(tokens)  # [B, L, d_model]
        tokens = self.norm(tokens)
        tokens = F.gelu(tokens)
        tokens = self.dropout(tokens)
        
        return tokens
