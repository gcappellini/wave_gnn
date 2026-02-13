import torch
import torch.nn as nn
import numpy as np


class FourierFeatureTransform(nn.Module):
    """
    Fourier Feature Mapping for PINNs to overcome spectral bias.
    
    Transforms input [x, t] using:
    \lambda(X) = [cos(BX), sin(BX)]
    
    where B is sampled from N(0, σ²)
    
    Args:
        input_dim: Dimension of input (e.g., 2 for [x, t])
        m_spatial: Number of Fourier features for spatial dimension
        m_temporal: Number of Fourier features for temporal dimension
        sigma_spatial: Standard deviation for spatial features (controls frequency)
        sigma_temporal_list: List of standard deviations for temporal features
        seed: Random seed for reproducibility (optional)
    """
    
    def __init__(self, 
                 input_dim=2,
                 m_spatial=64,
                 m_temporal=64,
                 sigma_spatial=1.0,
                 sigma_temporal_list=[1.0, 10.0],
                 seed=None):
        super().__init__()
        
        self.input_dim = input_dim
        self.m_spatial = m_spatial
        self.m_temporal = m_temporal
        self.sigma_spatial = sigma_spatial
        self.sigma_temporal_list = sigma_temporal_list
        
        # Set seed for reproducibility if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Create B matrices for each feature type
        # Spatial features: B_spatial of shape (m_spatial, 1) for x dimension
        self.register_buffer(
            'B_spatial',
            torch.randn(m_spatial, 1) * sigma_spatial
        )
        
        # Temporal features: list of B matrices with different sigmas
        self.B_temporal_list = []
        for idx, sigma_t in enumerate(sigma_temporal_list):
            B_t = torch.randn(m_temporal, 1) * sigma_t
            # Use index instead of sigma value to avoid dots in buffer name
            buffer_name = f'B_temporal_sigma_{idx}'
            self.register_buffer(buffer_name, B_t)
            self.B_temporal_list.append(B_t)
        
        # Calculate output dimension
        # For each spatial: 2*m_spatial (cos + sin)
        # For each temporal: 2*m_temporal (cos + sin) per sigma
        self.output_dim = 2 * m_spatial + len(sigma_temporal_list) * 2 * m_temporal
    
    def forward(self, X):
        """
        Apply Fourier feature mapping to input.
        
        Args:
            X: Input tensor of shape (batch_size, 2) where X[:, 0] is x and X[:, 1] is t
            
        Returns:
            Transformed features of shape (batch_size, output_dim)
        """
        # Split into spatial (x) and temporal (t) components
        x = X[:, 0:1]  # Shape: (batch_size, 1)
        t = X[:, 1:2]  # Shape: (batch_size, 1)
        
        features = []
        
        # Apply spatial Fourier features
        x_proj = torch.matmul(x, self.B_spatial.T)  # (batch_size, m_spatial)
        x_features = torch.cat([torch.cos(2 * np.pi * x_proj), 
                                torch.sin(2 * np.pi * x_proj)], dim=1)
        features.append(x_features)
        
        # Apply temporal Fourier features for each sigma
        for i, sigma_t in enumerate(self.sigma_temporal_list):
            # Use index instead of sigma value to access the buffer
            B_t = getattr(self, f'B_temporal_sigma_{i}')
            t_proj = torch.matmul(t, B_t.T)  # (batch_size, m_temporal)
            t_features = torch.cat([torch.cos(2 * np.pi * t_proj),
                                    torch.sin(2 * np.pi * t_proj)], dim=1)
            features.append(t_features)
        
        # Concatenate all features
        return torch.cat(features, dim=1)
    
    def get_output_dim(self):
        """Returns the output dimension after transformation."""
        return self.output_dim


# Example usage with your PINN
if __name__ == "__main__":
    # Create the Fourier feature transform
    # For 1D wave equation with c=10:
    # - Multiple temporal scales to capture fast propagation
    # - Spatial features for discontinuities/sharp gradients
    
    # CONSERVATIVE SETTING (faster training, less memory):
    feature_transform = FourierFeatureTransform(
        input_dim=2,
        m_spatial=32,       
        m_temporal=32,      
        sigma_spatial=5.0,  # Moderate spatial frequencies
        sigma_temporal_list=[1.0, 10.0],  # Low and high temporal frequencies
        seed=42
    )
    
    # RECOMMENDED SETTING (balanced):
    feature_transform = FourierFeatureTransform(
        input_dim=2,
        m_spatial=64,       
        m_temporal=64,      
        sigma_spatial=5.0,  
        sigma_temporal_list=[1.0, 10.0, 100.0],  # Multi-scale for c=10
        seed=42
    )
    
    # AGGRESSIVE SETTING (for very high-frequency solutions):
    feature_transform = FourierFeatureTransform(
        input_dim=2,
        m_spatial=128,       
        m_temporal=128,      
        sigma_spatial=10.0,  
        sigma_temporal_list=[1.0, 10.0, 100.0],
        seed=42
    )
    
    # Test the transform
    batch_size = 100
    X_test = torch.randn(batch_size, 2)  # [x, t] inputs
    
    transformed = feature_transform(X_test)
    print(f"Input shape: {X_test.shape}")
    print(f"Output shape: {transformed.shape}")
    print(f"Output dimension: {feature_transform.get_output_dim()}")
    
    # SCALING RULES FOR YOUR PROBLEM:
    # If spatial domain is [0, L] and time is [0, T]:
    # - For highest spatial frequency k_max in your solution:
    #   σ_spatial ~ k_max (or slightly higher)
    # - For wave speed c=10, temporal frequencies: ω = c*k
    #   σ_temporal should include c*k_max ≈ 10*k_max
    #
    # Example: if k_max ≈ 10 (10 waves in your domain):
    #   σ_spatial ~ 5-10
    #   σ_temporal ~ [1, 10, 100] to cover ω ~ 100
    
    # To apply to your PINN:
    # net._apply_feature_transform(feature_transform)
    # or if your PINN accepts it in the constructor:
    # net = YourPINN(feature_transform=feature_transform)