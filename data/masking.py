import numpy as np


def create_missing_mask(data_shape, missing_rate=0.3, missing_type='mcar'):
    """
    Create missing mask for time series data.
    
    Args:
        data_shape: tuple (N, C) or (B, L, C)
        missing_rate: float between 0 and 1
        missing_type: 'mcar', 'block', 'variable', or 'mixed'
    
    Returns:
        mask: numpy array same shape as data_shape (1=observed, 0=missing)
    """
    mask = np.ones(data_shape, dtype=np.float32)
    
    if missing_type == 'mcar':
        # Missing Completely At Random
        missing_indices = np.random.rand(*data_shape) < missing_rate
        mask[missing_indices] = 0.0
    
    elif missing_type == 'block':
        # Block missing (contiguous in time)
        if len(data_shape) == 2:
            N, C = data_shape
            block_size = max(1, int(N * missing_rate))
            n_blocks = max(1, int(N / (block_size * 2)))
            
            for _ in range(n_blocks):
                start = np.random.randint(0, N - block_size + 1)
                mask[start:start + block_size, :] = 0.0
        
        elif len(data_shape) == 3:
            B, L, C = data_shape
            block_size = max(1, int(L * missing_rate))
            
            for b in range(B):
                start = np.random.randint(0, max(1, L - block_size + 1))
                mask[b, start:start + block_size, :] = 0.0
    
    elif missing_type == 'variable':
        # Variable-wise missing
        if len(data_shape) == 2:
            N, C = data_shape
            n_missing_vars = max(1, int(C * missing_rate))
            missing_vars = np.random.choice(C, n_missing_vars, replace=False)
            mask[:, missing_vars] = 0.0
        
        elif len(data_shape) == 3:
            B, L, C = data_shape
            n_missing_vars = max(1, int(C * missing_rate))
            missing_vars = np.random.choice(C, n_missing_vars, replace=False)
            mask[:, :, missing_vars] = 0.0
    
    elif missing_type == 'mixed':
        # Combination of MCAR and block missing
        # First apply MCAR with reduced rate
        mcar_rate = missing_rate * 0.5
        missing_indices = np.random.rand(*data_shape) < mcar_rate
        mask[missing_indices] = 0.0
        
        # Then apply block missing with reduced rate
        if len(data_shape) == 2:
            N, C = data_shape
            block_size = max(1, int(N * (missing_rate * 0.5)))
            start = np.random.randint(0, max(1, N - block_size + 1))
            mask[start:start + block_size, :] = 0.0
        
        elif len(data_shape) == 3:
            B, L, C = data_shape
            block_size = max(1, int(L * (missing_rate * 0.5)))
            for b in range(B):
                start = np.random.randint(0, max(1, L - block_size + 1))
                mask[b, start:start + block_size, :] = 0.0
    
    return mask
