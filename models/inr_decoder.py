import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class INRDecoder(nn.Module):
    """Implicit Neural Representation decoder with trend/seasonal/residual decomposition."""
    
    def __init__(self, d_model, c_out, n_groups=16, seq_len=96, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.c_out = c_out
        self.n_groups = min(n_groups, c_out)  # Handle C < n_groups
        self.seq_len = seq_len
        
        # Trend: polynomial degree 2
        self.trend_poly_degree = 2
        self.trend_coeff = nn.Linear(d_model, c_out * (self.trend_poly_degree + 1))
        
        # Seasonal: Fourier features
        self.n_fourier = 8
        self.seasonal_proj = nn.Linear(d_model + 2 * self.n_fourier, d_model)
        self.seasonal_fc = nn.Linear(d_model, c_out)
        
        # Residual: group-based MLP with soft assignment
        self.residual_fc1 = nn.Linear(d_model, d_model * 2)
        self.residual_fc2 = nn.Linear(d_model * 2, c_out)
        
        # Soft group assignment (learnable)
        self.group_assign = nn.Linear(d_model, self.n_groups)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: [B, L, d_model]
        
        Returns:
            pred: [B, L, c_out]
        """
        B, L, d_model = x.shape
        device = x.device
        
        # Normalize
        x_norm = self.norm(x)  # [B, L, d_model]
        
        # Trend component
        trend_coeff = self.trend_coeff(x_norm)  # [B, L, c_out * (degree+1)]
        trend_coeff = trend_coeff.view(B, L, self.c_out, self.trend_poly_degree + 1)
        
        t = torch.linspace(0, 1, L, device=device)  # [L]
        t_poly = torch.stack([t ** i for i in range(self.trend_poly_degree + 1)], dim=1)  # [L, degree+1]
        trend = torch.einsum('blcd,ld->blc', trend_coeff, t_poly)  # [B, L, c_out]
        
        # Seasonal component with Fourier features
        fourier_feats = []
        for k in range(1, self.n_fourier + 1):
            freq = 2 * np.pi * k / L
            fourier_feats.append(torch.sin(freq * t).to(device))
            fourier_feats.append(torch.cos(freq * t).to(device))
        fourier_feats = torch.stack(fourier_feats, dim=1)  # [L, 2*n_fourier]
        
        # Expand fourier features for batch
        fourier_feats = fourier_feats.unsqueeze(0).expand(B, -1, -1)  # [B, L, 2*n_fourier]
        
        # Combine with x_norm and project
        seasonal_input = torch.cat([x_norm, fourier_feats], dim=-1)  # [B, L, d_model + 2*n_fourier]
        seasonal_hidden = self.seasonal_proj(seasonal_input)  # [B, L, d_model]
        seasonal_hidden = F.gelu(seasonal_hidden)
        seasonal = self.seasonal_fc(seasonal_hidden)  # [B, L, c_out]
        
        # Residual component with soft group assignment
        group_logits = self.group_assign(x_norm)  # [B, L, n_groups]
        group_weights = F.softmax(group_logits, dim=-1)  # [B, L, n_groups]
        
        residual_hidden = self.residual_fc1(x_norm)  # [B, L, d_model*2]
        residual_hidden = F.gelu(residual_hidden)
        residual = self.residual_fc2(residual_hidden)  # [B, L, c_out]
        
        # Combine: output(t, c) = trend(t, c) + seasonal(t, c) + residual(t, c)
        pred = trend + seasonal + residual  # [B, L, c_out]
        pred = self.dropout(pred)
        
        return pred  # [B, L, c_out]
