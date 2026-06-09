import torch
import torch.nn as nn
import torch.nn.functional as F


class ImputationLoss(nn.Module):
    """Configurable imputation loss with mask awareness."""
    
    def __init__(self, weight_observed=1.0, weight_diff=0.1, weight_spectral=0.1):
        super().__init__()
        self.weight_observed = weight_observed
        self.weight_diff = weight_diff
        self.weight_spectral = weight_spectral
        
    def forward(self, pred, target, mask):
        """
        Args:
            pred: [B, L, C] - model prediction
            target: [B, L, C] - ground truth
            mask: [B, L, C] - missing mask (1=observed, 0=missing)
        
        Returns:
            loss: scalar
        """
        # Missing loss: calculated only where mask == 0
        missing_loss = self._missing_loss(pred, target, mask)
        
        # Observed loss: calculated only where mask == 1
        observed_loss = self._observed_loss(pred, target, mask)
        
        # First difference loss
        diff_loss = self._diff_loss(pred, target, mask)
        
        # Spectral loss
        spectral_loss = self._spectral_loss(pred, target, mask)
        
        total_loss = (
            missing_loss +
            self.weight_observed * observed_loss +
            self.weight_diff * diff_loss +
            self.weight_spectral * spectral_loss
        )
        
        return total_loss
    
    def _missing_loss(self, pred, target, mask):
        """MSE loss only on missing values (mask == 0)."""
        missing_mask = (mask == 0).float()  # [B, L, C]
        if missing_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        loss = F.mse_loss(pred * missing_mask, target * missing_mask, reduction='sum')
        loss = loss / (missing_mask.sum() + 1e-6)
        return loss
    
    def _observed_loss(self, pred, target, mask):
        """MSE loss only on observed values (mask == 1)."""
        observed_mask = (mask == 1).float()  # [B, L, C]
        if observed_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        loss = F.mse_loss(pred * observed_mask, target * observed_mask, reduction='sum')
        loss = loss / (observed_mask.sum() + 1e-6)
        return loss
    
    def _diff_loss(self, pred, target, mask):
        """First-order difference loss."""
        # Compute first differences along time dimension
        pred_diff = pred[:, 1:, :] - pred[:, :-1, :]  # [B, L-1, C]
        target_diff = target[:, 1:, :] - target[:, :-1, :]  # [B, L-1, C]
        
        # Mask at time t and t+1 must both indicate missing
        mask_diff = mask[:, 1:, :] * mask[:, :-1, :]  # [B, L-1, C]
        if mask_diff.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        loss = F.mse_loss(pred_diff * mask_diff, target_diff * mask_diff, reduction='sum')
        loss = loss / (mask_diff.sum() + 1e-6)
        return loss
    
    def _spectral_loss(self, pred, target, mask):
        """Spectral loss using FFT."""
        # Compute FFT along time dimension
        pred_fft = torch.fft.rfft(pred, dim=1)  # [B, L//2+1, C]
        target_fft = torch.fft.rfft(target, dim=1)
        
        # Mask in frequency domain
        mask_fft = mask[:, :pred_fft.shape[1], :].float()  # align dimensions
        if mask_fft.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        loss = F.mse_loss(
            torch.abs(pred_fft) * mask_fft,
            torch.abs(target_fft) * mask_fft,
            reduction='sum'
        )
        loss = loss / (mask_fft.sum() + 1e-6)
        return loss
