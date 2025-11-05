"""
Phase 2 and Phase 3 models for wave equation prediction.

Phase 2: Pure spectral model using Laplacian eigenbasis
Phase 3: Hybrid model combining spatial message passing and spectral layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from spectral_layers import SpectralConv, SpectralLayer, SpectralMessagePassing


class Normalizer:
    """Handles normalization and denormalization of features."""
    
    def __init__(self, u_scale=0.04, v_scale=0.08, f_scale=3.0):
        self.u_scale = u_scale
        self.v_scale = v_scale
        self.f_scale = f_scale
    
    def normalize_input(self, u, v, f):
        """Normalize input features to roughly [-1, 1] range."""
        u_norm = u / self.u_scale
        v_norm = v / self.v_scale
        f_norm = f / self.f_scale
        return u_norm, v_norm, f_norm
    
    def denormalize_displacement(self, u_norm):
        """Denormalize predicted displacement change."""
        return u_norm * self.u_scale
    
    def denormalize_velocity(self, v_norm):
        """Denormalize predicted velocity change."""
        return v_norm * self.v_scale

class BoundaryCondition(nn.Module):
    """Applies hard boundary conditions (Dirichlet: u=0, v=0)."""
    
    def forward(self, u, v, boundary_mask):
        """
        Args:
            u: [N] displacement
            v: [N] velocity
            boundary_mask: [N] boolean mask (True for boundary nodes)
        Returns:
            u, v with boundary conditions applied
        """
        u = u * (~boundary_mask).float()
        v = v * (~boundary_mask).float()
        return u, v


# ==================== PHASE 2: PURE SPECTRAL MODEL ====================

class SpectralWaveGNN(nn.Module):
    """
    Phase 2: Pure spectral model using Laplacian eigenbasis.
    
    Architecture:
        Lifting → Spectral Layers → Projection → Physics
    
    Spectral Layer:
        h_spatial = node features in spatial domain
        ↓
        h_spectral = U^T @ h_spatial    (transform to frequency)
        ↓
        h_filtered = Learnable_Filter(h_spectral)  (process each mode)
        ↓
        h_spatial_new = U @ h_filtered   (transform back)
    """
    
    def __init__(self, cfg):
        super().__init__()
        cfg_model = cfg.model
        self.hidden_dim = cfg_model.hidden_dim
        self.num_layers = cfg_model.num_layers
        self.num_spectral_modes = cfg_model.num_spectral_modes
        self.dt = cfg.dataset.dt
        
        # Store for compatibility
        self.bc_mask = None
        
        # Normalization
        self.normalizer = Normalizer(
            cfg.dataset.u_scale, 
            cfg.dataset.v_scale, 
            cfg.dataset.f_scale
        )
        
        # Lifting: 3 input features -> hidden_dim
        self.lifting = nn.Sequential(
            nn.Linear(3, cfg_model.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg_model.hidden_dim, cfg_model.hidden_dim)
        )
        
        # Spectral layers
        self.spectral_layers = nn.ModuleList([
            SpectralMessagePassing(
                cfg_model.hidden_dim, 
                self.num_spectral_modes, 
                cfg_model.dropout
            )
            for _ in range(cfg_model.num_layers)
        ])
        
        # Projection: hidden_dim -> 2 (displacement change, velocity change)
        self.projection = nn.Sequential(
            nn.Linear(cfg_model.hidden_dim, cfg_model.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg_model.dropout),
            nn.Linear(cfg_model.hidden_dim, 2)
        )
        
        # Boundary conditions
        self.boundary = BoundaryCondition()
    
    def forward(self, x, edge_index=None, bc_mask=None, eigenvectors=None, **kwargs):
        """
        Forward pass - COMPATIBLE WITH DeepGCN INTERFACE.
        
        Args:
            x: [N, 3] node features with [u, v, f]
            edge_index: [2, E] graph connectivity (not used in pure spectral)
            bc_mask: [N] boolean mask for boundary nodes
            eigenvectors: [N, K] Laplacian eigenvectors (U matrix)
            **kwargs: Additional arguments (for compatibility)
        
        Returns:
            output: [N, 2] tensor with [u_next, v_next]
        """
        # Use bc_mask from argument or from self.bc_mask
        if bc_mask is None:
            bc_mask = self.bc_mask
        if bc_mask is None:
            raise ValueError("bc_mask must be provided")
        
        if eigenvectors is None:
            raise ValueError("eigenvectors must be provided for spectral model")
        
        # Extract individual components
        u = x[:, 0]
        v = x[:, 1]
        f = x[:, 2]
        
        # Normalize inputs
        u_norm, v_norm, f_norm = self.normalizer.normalize_input(u, v, f)
        
        # Lift to hidden dimension
        x_input = torch.stack([u_norm, v_norm, f_norm], dim=-1)
        h = self.lifting(x_input)
        
        # Spectral processing layers
        for spectral_layer in self.spectral_layers:
            h = spectral_layer(h, eigenvectors)
        
        # Project to [du_norm, dv_norm]
        delta_norm = self.projection(h)
        du_norm = delta_norm[:, 0]
        dv_norm = delta_norm[:, 1]
        
        # Denormalize
        du = self.normalizer.denormalize_displacement(du_norm)
        dv = self.normalizer.denormalize_velocity(dv_norm)
        
        # Direct integration
        u_next = u + du
        v_next = v + dv
        
        # Apply boundary conditions
        u_next, v_next = self.boundary(u_next, v_next, bc_mask)
        
        # Return in standard format
        output = torch.stack([u_next, v_next], dim=-1)
        return output


# ==================== PHASE 3: HYBRID MODEL ====================

class LocalMessagePassing(MessagePassing):
    """
    Local message passing layer for spatial branch.
    Communicates only with graph neighbors (edge-based).
    """
    
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__(aggr='add')
        
        # Local message MLP
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Node update MLP
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, edge_index):
        """
        Args:
            x: [N, hidden_dim] node features
            edge_index: [2, E] graph connectivity
        
        Returns:
            x_new: [N, hidden_dim] updated features
        """
        # Message passing
        x_messages = self.propagate(edge_index, x=x)
        
        # Update
        x_combined = torch.cat([x, x_messages], dim=-1)
        x_update = self.update_mlp(x_combined)
        
        # Residual + norm
        x_new = self.layer_norm(x + x_update)
        
        return x_new
    
    def message(self, x_i, x_j):
        """Compute messages from neighbors."""
        msg = torch.cat([x_i, x_j], dim=-1)
        return self.message_mlp(msg)


class HybridWaveGNN(nn.Module):
    """
    Phase 3: Hybrid model combining spatial and spectral processing.
    
    Architecture:
        Lifting → Parallel Branches → Combine → Projection → Physics
        
    Branch 1 (Spatial): Local message passing layers
    Branch 2 (Spectral): Fourier layers using Laplacian eigenbasis
    ↓
    Combine: h_final = MLP(concat[h_spatial, h_spectral])
    
    This combines:
    - Spatial branch: captures local interactions via graph structure
    - Spectral branch: captures global patterns via Fourier modes
    """
    
    def __init__(self, cfg):
        super().__init__()
        cfg_model = cfg.model
        self.hidden_dim = cfg_model.hidden_dim
        self.num_layers = cfg_model.num_layers
        self.num_spectral_modes = cfg_model.num_spectral_modes
        self.dt = cfg.dataset.dt
        
        # Hybrid branch weights
        self.spatial_weight = getattr(cfg_model, 'spatial_weight', 0.5)
        self.spectral_weight = getattr(cfg_model, 'spectral_weight', 0.5)
        
        # Store for compatibility
        self.bc_mask = None
        
        # Normalization
        self.normalizer = Normalizer(
            cfg.dataset.u_scale, 
            cfg.dataset.v_scale, 
            cfg.dataset.f_scale
        )
        
        # Shared lifting: 3 input features -> hidden_dim
        self.lifting = nn.Sequential(
            nn.Linear(3, cfg_model.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg_model.hidden_dim, cfg_model.hidden_dim)
        )
        
        # Branch 1: Spatial message passing layers
        self.spatial_layers = nn.ModuleList([
            LocalMessagePassing(cfg_model.hidden_dim, cfg_model.dropout)
            for _ in range(cfg_model.num_layers)
        ])
        
        # Branch 2: Spectral layers
        self.spectral_layers = nn.ModuleList([
            SpectralMessagePassing(
                cfg_model.hidden_dim, 
                self.num_spectral_modes, 
                cfg_model.dropout
            )
            for _ in range(cfg_model.num_layers)
        ])
        
        # Combine branches
        self.combine = nn.Sequential(
            nn.Linear(2 * cfg_model.hidden_dim, cfg_model.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg_model.dropout),
            nn.Linear(cfg_model.hidden_dim, cfg_model.hidden_dim)
        )
        
        # Projection: hidden_dim -> 2
        self.projection = nn.Sequential(
            nn.Linear(cfg_model.hidden_dim, cfg_model.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg_model.dropout),
            nn.Linear(cfg_model.hidden_dim, 2)
        )
        
        # Boundary conditions
        self.boundary = BoundaryCondition()
    
    def forward(self, x, edge_index, bc_mask=None, eigenvectors=None, **kwargs):
        """
        Forward pass with parallel spatial and spectral branches.
        
        Args:
            x: [N, 3] node features with [u, v, f]
            edge_index: [2, E] graph connectivity
            bc_mask: [N] boolean mask for boundary nodes
            eigenvectors: [N, K] Laplacian eigenvectors
            **kwargs: Additional arguments
        
        Returns:
            output: [N, 2] tensor with [u_next, v_next]
        """
        # Use bc_mask from argument or from self.bc_mask
        if bc_mask is None:
            bc_mask = self.bc_mask
        if bc_mask is None:
            raise ValueError("bc_mask must be provided")
        
        if eigenvectors is None:
            raise ValueError("eigenvectors must be provided for hybrid model")
        
        # Extract components
        u = x[:, 0]
        v = x[:, 1]
        f = x[:, 2]
        
        # Normalize inputs
        u_norm, v_norm, f_norm = self.normalizer.normalize_input(u, v, f)
        
        # Lift to hidden dimension (shared by both branches)
        x_input = torch.stack([u_norm, v_norm, f_norm], dim=-1)
        h = self.lifting(x_input)
        
        # Branch 1: Spatial processing
        h_spatial = h
        for spatial_layer in self.spatial_layers:
            h_spatial = spatial_layer(h_spatial, edge_index)
        
        # Branch 2: Spectral processing
        h_spectral = h
        for spectral_layer in self.spectral_layers:
            h_spectral = spectral_layer(h_spectral, eigenvectors)
        
        # Combine branches
        h_combined = torch.cat([h_spatial, h_spectral], dim=-1)
        h_final = self.combine(h_combined)
        
        # Residual connection from original lifted features
        h_final = h_final + h
        
        # Project to [du_norm, dv_norm]
        delta_norm = self.projection(h_final)
        du_norm = delta_norm[:, 0]
        dv_norm = delta_norm[:, 1]
        
        # Denormalize
        du = self.normalizer.denormalize_displacement(du_norm)
        dv = self.normalizer.denormalize_velocity(dv_norm)
        
        # Direct integration
        u_next = u + du
        v_next = v + dv
        
        # Apply boundary conditions
        u_next, v_next = self.boundary(u_next, v_next, bc_mask)
        
        # Return in standard format
        output = torch.stack([u_next, v_next], dim=-1)
        return output


# ==================== MODEL FACTORY ====================

def create_spectral_model(cfg, phase='hybrid'):
    """
    Factory function to create spectral-based models.
    
    Args:
        cfg: Hydra config object
        phase: 'spectral' for Phase 2, 'hybrid' for Phase 3
    
    Returns:
        model: SpectralWaveGNN or HybridWaveGNN instance
    """
    if phase == 'spectral' or phase == 'phase2':
        return SpectralWaveGNN(cfg)
    elif phase == 'hybrid' or phase == 'phase3':
        return HybridWaveGNN(cfg)
    else:
        raise ValueError(f"Unknown phase: {phase}. Use 'spectral' or 'hybrid'")


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("Phase 2 and Phase 3 models for spectral graph neural networks")
    print("\nPhase 2: Pure spectral model")
    print("  - Uses only Laplacian eigenbasis for global communication")
    print("  - Architecture: Lifting → Spectral Layers → Projection")
    print("\nPhase 3: Hybrid model")
    print("  - Combines local message passing with spectral processing")
    print("  - Architecture: Lifting → [Spatial + Spectral] → Combine → Projection")
    print("\nUsage:")
    print("  from spectral_models import create_spectral_model")
    print("  model = create_spectral_model(cfg, phase='hybrid')")
