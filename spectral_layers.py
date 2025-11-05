"""
Spectral layers for Graph Neural Networks using Laplacian eigenbasis.

This module implements spectral graph convolutions using the Fourier transform
via Laplacian eigendecomposition. The key idea is:
  1. Transform node features to spectral domain using eigenvectors
  2. Apply learnable filters in frequency space
  3. Transform back to spatial domain

Reference: Spectral Networks and Deep Locally Connected Networks (Bruna et al., 2013)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import sparse as sp


def compute_laplacian_eigendecomposition(L, k=None, normalized=True):
    """
    Compute eigendecomposition of graph Laplacian.
    
    For a graph Laplacian L, we compute:
        L @ U = U @ Λ
    where U contains eigenvectors (columns) and Λ is diagonal with eigenvalues.
    
    Args:
        L: Laplacian matrix (scipy sparse or torch tensor)
        k: Number of smallest eigenvalues/vectors to compute. 
           If None, compute all (for small graphs only!)
        normalized: If True, use normalized Laplacian L_norm = D^(-1/2) L D^(-1/2)
    
    Returns:
        eigenvalues: [k] or [N] eigenvalues in ascending order
        eigenvectors: [N, k] or [N, N] matrix U where columns are eigenvectors
    """
    # Convert to scipy sparse if needed
    if torch.is_tensor(L):
        if L.is_sparse:
            # Convert torch sparse to scipy sparse
            indices = L._indices().cpu().numpy()
            values = L._values().cpu().numpy()
            shape = L.shape
            L_scipy = sp.coo_matrix((values, (indices[0], indices[1])), shape=shape)
        else:
            L_scipy = sp.coo_matrix(L.cpu().numpy())
    else:
        L_scipy = L
    
    N = L_scipy.shape[0]
    
    # Optional: normalize the Laplacian
    if normalized:
        # Compute degree matrix
        degrees = np.array(L_scipy.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1  # Avoid division by zero
        D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees))
        L_norm = D_inv_sqrt @ L_scipy @ D_inv_sqrt
        L_scipy = L_norm
    
    # Compute eigendecomposition
    if k is None or k >= N - 1:
        # Compute all eigenvalues/vectors for small graphs
        L_dense = L_scipy.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
    else:
        # Compute k smallest eigenvalues/vectors for large graphs
        # Note: For Laplacian, smallest eigenvalues correspond to low-frequency modes
        from scipy.sparse.linalg import eigsh
        eigenvalues, eigenvectors = eigsh(L_scipy, k=k, which='SM')
    
    # Sort by eigenvalue (should already be sorted, but ensure it)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Convert to torch tensors
    eigenvalues = torch.from_numpy(eigenvalues).float()
    eigenvectors = torch.from_numpy(eigenvectors).float()
    
    return eigenvalues, eigenvectors


class SpectralConv(nn.Module):
    """
    Spectral graph convolution layer.
    
    Performs convolution in spectral domain:
        h_spatial -> U^T @ h_spatial (to frequency)
        -> element-wise multiply with learnable filter
        -> U @ h_filtered (back to spatial)
    
    This is equivalent to: h_out = U @ diag(filter) @ U^T @ h_in
    """
    
    def __init__(self, in_channels, out_channels, num_modes, dropout=0.0):
        """
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            num_modes: Number of spectral modes (eigenvectors) to use
            dropout: Dropout probability
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_modes = num_modes
        
        # Learnable spectral filters for each input-output channel pair
        # Shape: [out_channels, in_channels, num_modes]
        self.spectral_weights = nn.Parameter(
            torch.randn(out_channels, in_channels, num_modes) * 0.02
        )
        
        # Bias in spatial domain
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x, U):
        """
        Args:
            x: [N, in_channels] node features in spatial domain
            U: [N, num_modes] eigenvector matrix
        
        Returns:
            h_out: [N, out_channels] filtered features in spatial domain
        """
        N = x.shape[0]
        
        # Transform to spectral domain: [N, in_channels] -> [num_modes, in_channels]
        x_spectral = U.T @ x  # [num_modes, in_channels]
        
        # Apply learnable filters in frequency space
        # Broadcasting: [out_channels, in_channels, num_modes] * [1, in_channels, num_modes]
        # -> sum over in_channels -> [out_channels, num_modes]
        x_spectral_expanded = x_spectral.T.unsqueeze(0)  # [1, in_channels, num_modes]
        filtered = (self.spectral_weights * x_spectral_expanded).sum(dim=1)  # [out_channels, num_modes]
        
        # Transform back to spatial domain: [out_channels, num_modes] -> [N, out_channels]
        x_spatial = U @ filtered.T  # [N, out_channels]
        
        # Add bias and apply dropout
        x_spatial = x_spatial + self.bias
        x_spatial = self.dropout(x_spatial)
        
        return x_spatial


class SpectralLayer(nn.Module):
    """
    Complete spectral layer with normalization and activation.
    
    Architecture:
        Input -> SpectralConv -> LayerNorm -> Activation -> Output
    """
    
    def __init__(self, in_channels, out_channels, num_modes, 
                 activation='relu', dropout=0.0, use_norm=True):
        super().__init__()
        
        self.conv = SpectralConv(in_channels, out_channels, num_modes, dropout)
        
        if use_norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = nn.Identity()
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()
    
    def forward(self, x, U):
        """
        Args:
            x: [N, in_channels] node features
            U: [N, num_modes] eigenvector matrix
        
        Returns:
            h: [N, out_channels] output features
        """
        h = self.conv(x, U)
        h = self.norm(h)
        h = self.activation(h)
        return h


class SpectralMessagePassing(nn.Module):
    """
    Spectral global communication layer.
    
    Replaces the global node broadcast with spectral (Fourier) transform:
      1. Transform to frequency domain via eigenvectors
      2. Process each frequency mode with learnable filters
      3. Transform back to spatial domain
      
    This provides a more principled global communication mechanism
    based on the graph structure (encoded in Laplacian eigenvectors).
    """
    
    def __init__(self, hidden_dim, num_modes, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        
        # Spectral convolution
        self.spectral_conv = SpectralConv(hidden_dim, hidden_dim, num_modes, dropout)
        
        # Node update MLP
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, U):
        """
        Args:
            x: [N, hidden_dim] node features in spatial domain
            U: [N, num_modes] Laplacian eigenvectors
        
        Returns:
            x_new: [N, hidden_dim] updated node features
        """
        # Spectral processing (global communication via Fourier transform)
        x_spectral = self.spectral_conv(x, U)
        
        # Combine original and spectral features
        x_combined = torch.cat([x, x_spectral], dim=-1)
        x_update = self.update_mlp(x_combined)
        
        # Residual connection and normalization
        x_new = self.layer_norm(x + x_update)
        
        return x_new


class MultiScaleSpectralLayer(nn.Module):
    """
    Multi-scale spectral layer that processes different frequency bands.
    
    Low frequencies: global, smooth patterns (long-range)
    High frequencies: local, detailed patterns (short-range)
    """
    
    def __init__(self, hidden_dim, num_modes_low, num_modes_high, dropout=0.1):
        super().__init__()
        
        # Separate processing for low and high frequency modes
        self.low_freq_conv = SpectralConv(hidden_dim, hidden_dim // 2, num_modes_low, dropout)
        self.high_freq_conv = SpectralConv(hidden_dim, hidden_dim // 2, num_modes_high, dropout)
        
        # Combine multi-scale features
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, U):
        """
        Args:
            x: [N, hidden_dim] node features
            U: [N, K] eigenvectors (K >= num_modes_low + num_modes_high)
        
        Returns:
            x_new: [N, hidden_dim] multi-scale features
        """
        # Split eigenvectors into low and high frequency modes
        U_low = U[:, :self.low_freq_conv.num_modes]
        # Take some high-frequency modes from the middle-to-end range
        start_idx = U.shape[1] - self.high_freq_conv.num_modes
        U_high = U[:, start_idx:]
        
        # Process each scale
        x_low = self.low_freq_conv(x, U_low)
        x_high = self.high_freq_conv(x, U_high)
        
        # Concatenate and combine
        x_multi = torch.cat([x_low, x_high], dim=-1)
        x_combined = self.combine(x_multi)
        
        # Residual and norm
        x_new = self.layer_norm(x + x_combined)
        
        return x_new
