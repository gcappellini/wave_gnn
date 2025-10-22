"""
Data scaling utilities for normalizing inputs and outputs.
"""

import torch
import numpy as np
from pathlib import Path
import pickle
import logging

log = logging.getLogger(__name__)


class DataScaler:
    """
    Handles input and output scaling for neural network training.
    
    Supports:
    - Standard normalization: (x - mean) / std
    - Min-max normalization: (x - min) / (max - min)
    - Per-feature scaling (each feature scaled independently)
    - Global scaling (all features scaled together)
    """
    
    def __init__(self, method='standard', per_feature=True, epsilon=1e-8):
        """
        Args:
            method: 'standard' (z-score) or 'minmax'
            per_feature: If True, scale each feature independently
            epsilon: Small value to prevent division by zero
        """
        self.method = method
        self.per_feature = per_feature
        self.epsilon = epsilon
        
        # Statistics will be computed from training data
        self.input_mean = None
        self.input_std = None
        self.input_min = None
        self.input_max = None
        
        self.output_mean = None
        self.output_std = None
        self.output_min = None
        self.output_max = None
        
        self.is_fitted = False
    
    def fit(self, train_loader, input_indices=None, output_indices=None):
        """
        Compute scaling statistics from training data.
        
        Args:
            train_loader: DataLoader with training data
            input_indices: List of indices for input features (default: all)
            output_indices: List of indices for output features (default: [0, 1])
        """
        log.info("Computing scaling statistics from training data...")
        
        all_inputs = []
        all_outputs = []
        
        for batch in train_loader:
            data_list = batch.to_data_list()
            for data in data_list:
                x = data.x.cpu().numpy()
                all_inputs.append(x)
                # Outputs are typically the first 2 features in next timestep
                # For now, we'll use the same as input
                all_outputs.append(x[:, output_indices if output_indices else [0, 1]])
        
        # Concatenate all data
        all_inputs = np.concatenate(all_inputs, axis=0)  # (N_total, F_in)
        all_outputs = np.concatenate(all_outputs, axis=0)  # (N_total, F_out)
        
        if input_indices is not None:
            all_inputs = all_inputs[:, input_indices]
        
        # Compute statistics
        if self.method == 'standard':
            if self.per_feature:
                self.input_mean = np.mean(all_inputs, axis=0)
                self.input_std = np.std(all_inputs, axis=0) + self.epsilon
                self.output_mean = np.mean(all_outputs, axis=0)
                self.output_std = np.std(all_outputs, axis=0) + self.epsilon
            else:
                self.input_mean = np.mean(all_inputs)
                self.input_std = np.std(all_inputs) + self.epsilon
                self.output_mean = np.mean(all_outputs)
                self.output_std = np.std(all_outputs) + self.epsilon
            
            log.info(f"Input scaling: mean={self.input_mean}, std={self.input_std}")
            log.info(f"Output scaling: mean={self.output_mean}, std={self.output_std}")
            
        elif self.method == 'minmax':
            if self.per_feature:
                self.input_min = np.min(all_inputs, axis=0)
                self.input_max = np.max(all_inputs, axis=0)
                self.output_min = np.min(all_outputs, axis=0)
                self.output_max = np.max(all_outputs, axis=0)
            else:
                self.input_min = np.min(all_inputs)
                self.input_max = np.max(all_inputs)
                self.output_min = np.min(all_outputs)
                self.output_max = np.max(all_outputs)
            
            # Ensure max != min
            if self.per_feature:
                self.input_max = np.maximum(self.input_max, self.input_min + self.epsilon)
                self.output_max = np.maximum(self.output_max, self.output_min + self.epsilon)
            else:
                self.input_max = max(self.input_max, self.input_min + self.epsilon)
                self.output_max = max(self.output_max, self.output_min + self.epsilon)
            
            log.info(f"Input scaling: min={self.input_min}, max={self.input_max}")
            log.info(f"Output scaling: min={self.output_min}, max={self.output_max}")
        
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
        
        self.is_fitted = True
        log.info("✓ Scaling statistics computed successfully")
    
    def transform_input(self, x):
        """
        Scale input features.
        
        Args:
            x: Input tensor of shape (N, F_in)
            
        Returns:
            Scaled input tensor
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before transform")
        
        is_tensor = torch.is_tensor(x)
        device = x.device if is_tensor else None
        
        if is_tensor:
            x = x.detach().cpu().numpy()
        
        if self.method == 'standard':
            x_scaled = (x - self.input_mean) / self.input_std
        elif self.method == 'minmax':
            x_scaled = (x - self.input_min) / (self.input_max - self.input_min)
        
        if is_tensor:
            x_scaled = torch.tensor(x_scaled, dtype=torch.float32, device=device)
        
        return x_scaled
    
    def inverse_transform_input(self, x_scaled):
        """
        Inverse scale input features (denormalize).
        
        Args:
            x_scaled: Scaled input tensor
            
        Returns:
            Original scale input tensor
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before inverse transform")
        
        is_tensor = torch.is_tensor(x_scaled)
        device = x_scaled.device if is_tensor else None
        
        if is_tensor:
            x_scaled = x_scaled.cpu().numpy()
        
        if self.method == 'standard':
            x = x_scaled * self.input_std + self.input_mean
        elif self.method == 'minmax':
            x = x_scaled * (self.input_max - self.input_min) + self.input_min
        
        if is_tensor:
            x = torch.tensor(x, dtype=torch.float32, device=device)
        
        return x
    
    def transform_output(self, y):
        """
        Scale output features.
        
        Args:
            y: Output tensor of shape (N, F_out)
            
        Returns:
            Scaled output tensor
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before transform")
        
        is_tensor = torch.is_tensor(y)
        device = y.device if is_tensor else None
        
        if is_tensor:
            y = y.cpu().numpy()
        
        if self.method == 'standard':
            y_scaled = (y - self.output_mean) / self.output_std
        elif self.method == 'minmax':
            y_scaled = (y - self.output_min) / (self.output_max - self.output_min)
        
        if is_tensor:
            y_scaled = torch.tensor(y_scaled, dtype=torch.float32, device=device)
        
        return y_scaled
    
    def inverse_transform_output(self, y_scaled):
        """
        Inverse scale output features (denormalize).
        
        Args:
            y_scaled: Scaled output tensor
            
        Returns:
            Original scale output tensor
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before inverse transform")
        
        is_tensor = torch.is_tensor(y_scaled)
        device = y_scaled.device if is_tensor else None
        
        if is_tensor:
            y_scaled_np = y_scaled.detach().cpu().numpy()
        else:
            y_scaled_np = y_scaled
        
        # Check for extreme values before inverse transform
        if np.abs(y_scaled_np).max() > 100:
            log.warning(f"Extremely large scaled values detected (max: {np.abs(y_scaled_np).max():.3e}). "
                       f"This may cause numerical overflow during inverse transform.")
        
        if self.method == 'standard':
            y = y_scaled_np * self.output_std + self.output_mean
        elif self.method == 'minmax':
            y = y_scaled_np * (self.output_max - self.output_min) + self.output_min
        
        # Clip extreme values to prevent overflow (safety measure)
        max_safe_value = 1e10  # Adjust based on your domain
        if np.abs(y).max() > max_safe_value:
            log.warning(f"Clipping extreme values in inverse transform! Max: {np.abs(y).max():.3e}")
            y = np.clip(y, -max_safe_value, max_safe_value)
        
        if is_tensor:
            y = torch.tensor(y, dtype=torch.float32, device=device)
        
        return y
    
    def save(self, path):
        """Save scaler parameters to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'method': self.method,
            'per_feature': self.per_feature,
            'epsilon': self.epsilon,
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'input_min': self.input_min,
            'input_max': self.input_max,
            'output_mean': self.output_mean,
            'output_std': self.output_std,
            'output_min': self.output_min,
            'output_max': self.output_max,
            'is_fitted': self.is_fitted,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        log.info(f"Scaler saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load scaler parameters from file."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        scaler = cls(
            method=state['method'],
            per_feature=state['per_feature'],
            epsilon=state['epsilon']
        )
        
        scaler.input_mean = state['input_mean']
        scaler.input_std = state['input_std']
        scaler.input_min = state['input_min']
        scaler.input_max = state['input_max']
        scaler.output_mean = state['output_mean']
        scaler.output_std = state['output_std']
        scaler.output_min = state['output_min']
        scaler.output_max = state['output_max']
        scaler.is_fitted = state['is_fitted']
        
        log.info(f"Scaler loaded from {path}")
        return scaler
