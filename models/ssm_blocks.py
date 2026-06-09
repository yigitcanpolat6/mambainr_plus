import torch
import torch.nn as nn
import torch.nn.functional as F


class SSMBlock(nn.Module):
    """Fallback SSM block when mamba_ssm is unavailable."""
    
    def __init__(self, d_model, expansion=2, dropout=0.1):
        super().__init__()
        d_inner = d_model * expansion
        
        # Temporal modeling: depthwise Conv1D
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=1, groups=d_model
        )
        
        # Gated MLP
        self.fc1 = nn.Linear(d_model, d_inner)
        self.fc2 = nn.Linear(d_inner, d_model)
        self.gate = nn.Linear(d_model, d_inner)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: [B, L, d_model]
        
        Returns:
            out: [B, L, d_model]
        """
        B, L, d_model = x.shape
        residual = x
        
        # Norm + depthwise conv
        x_norm = self.norm(x)  # [B, L, d_model]
        x_conv = x_norm.transpose(1, 2)  # [B, d_model, L]
        x_conv = self.depthwise_conv(x_conv)  # [B, d_model, L]
        x_conv = x_conv.transpose(1, 2)  # [B, L, d_model]
        x_conv = F.gelu(x_conv)
        
        # Gated MLP
        gate = self.gate(x_norm)  # [B, L, d_inner]
        x_mlp = self.fc1(x_norm)  # [B, L, d_inner]
        x_mlp = F.gelu(x_mlp)
        x_mlp = x_mlp * gate
        x_mlp = self.fc2(x_mlp)  # [B, L, d_model]
        
        out = x_conv + x_mlp + residual
        out = self.dropout(out)
        
        return out


class TemporalEncoder(nn.Module):
    """Optional Mamba or SSM temporal encoder."""
    
    def __init__(self, d_model, n_layers, use_mamba=True, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.use_mamba = use_mamba
        
        if use_mamba:
            try:
                from mamba_ssm import Mamba
                self.layers = nn.ModuleList([Mamba(d_model, bimamba_type="v2") for _ in range(n_layers)])
                self.mamba_available = True
            except ImportError:
                # Fallback to SSM block
                self.layers = nn.ModuleList([SSMBlock(d_model, dropout=dropout) for _ in range(n_layers)])
                self.mamba_available = False
        else:
            self.layers = nn.ModuleList([SSMBlock(d_model, dropout=dropout) for _ in range(n_layers)])
            self.mamba_available = False
    
    def forward(self, x):
        """
        Args:
            x: [B, L, d_model]
        
        Returns:
            out: [B, L, d_model]
        """
        for layer in self.layers:
            x = layer(x)
        return x
