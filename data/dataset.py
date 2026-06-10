import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os


class TSDataset(Dataset):
    """Time series dataset with chronological split."""
    
    def __init__(self, data, mask, times, seq_len, mode='train'):
        """
        Args:
            data: [N, C] - normalized time series
            mask: [N, C] - missing mask (1=observed, 0=missing)
            times: [N] - time indices
            seq_len: sequence length
            mode: 'train', 'val', or 'test'
        """
        self.data = data
        self.mask = mask
        self.times = times
        self.seq_len = seq_len
        self.mode = mode
        
    def __len__(self):
        return max(0, len(self.data) - self.seq_len + 1)
    
    def __getitem__(self, idx):
        # Extract sequence
        end_idx = idx + self.seq_len
        x = self.data[idx:end_idx]  # [seq_len, C]
        mask = self.mask[idx:end_idx]  # [seq_len, C]
        time = self.times[idx:end_idx]  # [seq_len]
        
        # Convert to tensors
        x = torch.tensor(x, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        time = torch.tensor(time, dtype=torch.float32)
        
        return x, mask, time


def load_ts_data(data_path, seq_len=96, train_ratio=0.6, val_ratio=0.2):
    """
    Load and split time series data chronologically.
    
    Args:
        data_path: path to CSV file
        seq_len: sequence length
        train_ratio: proportion of training data
        val_ratio: proportion of validation data
    
    Returns:
        train_dataset, val_dataset, test_dataset, scaler
    """
    # Load data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Remove date/time column if present
    date_cols = ['date', 'time', 'Date', 'Time', 'datetime', 'Datetime']
    for col in date_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
            break
    
    # Extract numpy array
    data = df.values.astype(np.float32)  # [N, C]
    n_samples, n_channels = data.shape
    
    # Chronological split
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    # Fit scaler on train data only
    scaler = StandardScaler()
    scaler.fit(train_data)
    
    # Normalize all splits using train statistics
    train_data = scaler.transform(train_data)
    val_data = scaler.transform(val_data)
    test_data = scaler.transform(test_data)
    
    # Time features: normalized indices
    train_times = np.arange(len(train_data)) / len(train_data)
    val_times = np.arange(len(val_data)) / len(val_data)
    test_times = np.arange(len(test_data)) / len(test_data)
    
    # Create dummy masks (no missing values initially)
    train_mask = np.ones_like(train_data)
    val_mask = np.ones_like(val_data)
    test_mask = np.ones_like(test_data)
    
    # Create datasets
    train_dataset = TSDataset(train_data, train_mask, train_times, seq_len, mode='train')
    val_dataset = TSDataset(val_data, val_mask, val_times, seq_len, mode='val')
    test_dataset = TSDataset(test_data, test_mask, test_times, seq_len, mode='test')
    
    return train_dataset, val_dataset, test_dataset, scaler
