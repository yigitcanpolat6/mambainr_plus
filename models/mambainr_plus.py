import torch
import torch.nn as nn
from .tokenizer import MaskAwareTokenizer
from .ssm_blocks import TemporalEncoder
from .channel_mixer import ChannelMixer
from .inr_decoder import INRDecoder
from .losses import ImputationLoss


class MambaINRPlus(nn.Module):
    """Complete MambaINRPlus model."""
    
    def __init__(self, c_in, c_out, d_model=128, n_layers=4, n_groups=16, 
                 seq_len=96, use_mamba=True, dropout=0.1, 
                 weight_observed=1.0, weight_diff=0.1, weight_spectral=0.1):
        super().__init__()
        
        self.c_in = c_in
        self.c_out = c_out
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_groups = min(n_groups, c_out)
        self.seq_len = seq_len
        
        # Stage 1: Mask-aware tokenizer
        self.tokenizer = MaskAwareTokenizer(c_in, d_model, dropout=dropout)
        
        # Stage 2: Optional Mamba/SSM temporal encoder
        self.temporal_encoder = TemporalEncoder(
            d_model, n_layers, use_mamba=use_mamba, dropout=dropout
        )
        
        # Stage 3: Channel correlation module
        self.channel_mixer = ChannelMixer(
            d_model, d_model, n_groups=n_groups, dropout=dropout
        )
        
        # Stage 4: INR decoder
        self.inr_decoder = INRDecoder(
            d_model, c_out, n_groups=n_groups, seq_len=seq_len, dropout=dropout
        )
        
        # Loss function
        self.loss_fn = ImputationLoss(
            weight_observed=weight_observed,
            weight_diff=weight_diff,
            weight_spectral=weight_spectral
        )
        
    def forward(self, x, mask, time_features=None):
        """
        Args:
            x: [B, L, C] - input time series
            mask: [B, L, C] - missing mask (1=observed, 0=missing)
            time_features: [B, L] or None
        
        Returns:
            pred: [B, L, C] - imputed time series
        """
        # Stage 1: Tokenize
        tokens = self.tokenizer(x, mask, time_features)  # [B, L, d_model]
        
        # Stage 2: Temporal encoding
        encoded = self.temporal_encoder(tokens)  # [B, L, d_model]
        
        # Stage 3: Channel mixing
        mixed = self.channel_mixer(encoded, mask)  # [B, L, d_model]
        
        # Stage 4: INR decoding
        pred = self.inr_decoder(mixed)  # [B, L, c_out]
        
        return pred
    
    def compute_loss(self, pred, target, mask):
        """
        Args:
            pred: [B, L, C]
            target: [B, L, C]
            mask: [B, L, C]
        
        Returns:
            loss: scalar
        """
        return self.loss_fn(pred, target, mask)
