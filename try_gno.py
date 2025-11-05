"""
Graph Neural Operator (GNO) for Wave Equation Prediction

This module implements three phases of global communication in GNNs:

PHASE 1 (Current): Global Node Broadcast
    - Uses a virtual global node that aggregates information from all nodes
    - Broadcasts global information back to all nodes
    - Simple but not based on graph structure

PHASE 2: Spectral Global Communication
    - Replaces global node with Fourier transform via Laplacian eigenbasis
    - Architecture: Lifting → Spectral Layers → Projection → Physics
    - Spectral Layer:
        h_spatial → U^T @ h_spatial (to frequency)
        → Learnable_Filter(h_spectral) (process modes)
        → U @ h_filtered (back to spatial)
    - See spectral_models.py for implementation

PHASE 3: Hybrid Spatial-Spectral Model
    - Combines local message passing with spectral processing
    - Architecture: Lifting → Parallel[Spatial, Spectral] → Combine → Projection
    - Branch 1: Local MP layers (captures local structure)
    - Branch 2: Spectral layers (captures global patterns)
    - Combine: h_final = MLP(concat[h_spatial, h_spectral])
    - See spectral_models.py for implementation

To use:
    Phase 1: model = WaveGNN(cfg)
    Phase 2: from spectral_models import SpectralWaveGNN; model = SpectralWaveGNN(cfg)
    Phase 3: from spectral_models import HybridWaveGNN; model = HybridWaveGNN(cfg)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


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


class GlobalMessagePassing(MessagePassing):
    """
    PHASE 1: Message passing layer with virtual global node.
    
    Each node communicates with:
    - Its local neighbors (via graph edges)
    - A global node (broadcast communication)
    
    This provides a baseline for global communication that will be
    replaced by spectral methods in Phase 2, and combined with
    local methods in Phase 3.
    """
    
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__(aggr='add')
        
        # Local message passing
        self.local_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Global node communication
        self.to_global = nn.Linear(hidden_dim, hidden_dim)
        self.from_global = nn.Linear(hidden_dim, hidden_dim)
        self.global_update = nn.Linear(hidden_dim, hidden_dim)
        
        # Node update
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, edge_index, global_node):
        """
        Args:
            x: [N, hidden_dim] node features
            edge_index: [2, E] graph connectivity
            global_node: [1, hidden_dim] global node feature
        Returns:
            x_new: [N, hidden_dim] updated node features
            global_new: [1, hidden_dim] updated global node
        """
        # Local message passing
        x_local = self.propagate(edge_index, x=x)
        
        # Global communication: nodes -> global
        global_message = self.to_global(x).mean(dim=0, keepdim=True)  # [1, hidden_dim]
        global_new = global_node + self.global_update(global_message)
        
        # Global communication: global -> nodes
        global_broadcast = self.from_global(global_new).expand_as(x)  # [N, hidden_dim]
        
        # Combine local and global messages
        x_combined = torch.cat([x_local, global_broadcast], dim=-1)
        x_update = self.update_mlp(x_combined)
        
        # Residual connection and normalization
        x_new = self.layer_norm(x + x_update)
        
        return x_new, global_new
    
    def message(self, x_i, x_j):
        """Compute messages from neighbors."""
        msg = torch.cat([x_i, x_j], dim=-1)
        return self.local_mlp(msg)


class PhysicsIntegrator(nn.Module):
    """Enforces physics-based time integration."""
    
    def __init__(self, dt):
        super().__init__()
        self.dt = dt
    
    def forward(self, u, v, dv):
        """
        Integrate state using trapezoidal rule.
        Args:
            u: [N] displacement at time t
            v: [N] velocity at time t
            dv: [N] predicted velocity change
        Returns:
            u_next: [N] displacement at t+dt
            v_next: [N] velocity at t+dt
        """
        v_next = v + dv
        u_next = u + self.dt * (v + v_next) / 2.0  # Trapezoidal rule
        return u_next, v_next


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


class WaveGNN(nn.Module):
    """
    PHASE 1: Graph Neural Network for wave equation with global node.
    
    Predicts both displacement and velocity changes directly using
    a virtual global node for global communication.
    
    DROP-IN REPLACEMENT for DeepGCN:
    - Compatible with existing training loop
    - Same forward signature: forward(x, edge_index, bc_mask)
    - Returns [N, 2] tensor with [u_next, v_next]
    
    For Phase 2 (spectral) and Phase 3 (hybrid), see spectral_models.py
    """
    
    def __init__(
        self,
        cfg
    ):
        super().__init__()
        cfg_model = cfg.model
        self.hidden_dim = cfg_model.hidden_dim
        self.num_layers = cfg_model.num_layers
        self.dt = cfg.dataset.dt
        
        # Store for compatibility with train.py
        # self.in_channels = cfg_model.in_channels
        # self.out_channels = cfg_model.out_channels
        self.bc_mask = None  # Will be set externally like DeepGCN
        
        # Normalization
        self.normalizer = Normalizer(cfg.dataset.u_scale, cfg.dataset.v_scale, cfg.dataset.f_scale)
        
        # Lifting: 3 input features -> hidden_dim
        self.lifting = nn.Sequential(
            nn.Linear(3, cfg_model.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg_model.hidden_dim, cfg_model.hidden_dim)
        )
        
        # Initialize global node
        self.global_node_init = nn.Parameter(torch.randn(1, cfg_model.hidden_dim))
        
        # Message passing layers
        self.mp_layers = nn.ModuleList([
            GlobalMessagePassing(cfg_model.hidden_dim, cfg_model.dropout) 
            for _ in range(cfg_model.num_layers)
        ])
        
        # Projection: hidden_dim -> 2 (displacement change, velocity change)
        self.projection = nn.Sequential(
            nn.Linear(cfg_model.hidden_dim, cfg_model.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg_model.dropout),
            nn.Linear(cfg_model.hidden_dim, 2)  # Predicts [du_norm, dv_norm]
        )
        
        # Boundary conditions
        self.boundary = BoundaryCondition()
    
    def forward(self, x, edge_index, bc_mask=None, batch=None, **kwargs):
        """
        Forward pass - COMPATIBLE WITH DeepGCN INTERFACE.
        
        Args:
            x: [N, 3] node features with [u, v, f]
            edge_index: [2, E] graph connectivity
            bc_mask: [N] boolean mask for boundary nodes (or use self.bc_mask)
            batch: [N] batch vector (optional, for multi-graph batching)
            **kwargs: Additional arguments (for compatibility)
        
        Returns:
            output: [N, 2] tensor with [u_next, v_next]
        """
        # Use bc_mask from argument or from self.bc_mask (set externally)
        if bc_mask is None:
            bc_mask = self.bc_mask
        if bc_mask is None:
            raise ValueError("bc_mask must be provided either as argument or set as self.bc_mask")
        
        # Extract individual components from input
        u = x[:, 0]  # [N]
        v = x[:, 1]  # [N]
        f = x[:, 2]  # [N]
        
        # Normalize inputs
        u_norm, v_norm, f_norm = self.normalizer.normalize_input(u, v, f)
        
        # Lift to hidden dimension
        x_input = torch.stack([u_norm, v_norm, f_norm], dim=-1)  # [N, 3]
        h = self.lifting(x_input)  # [N, hidden_dim]
        
        # Initialize global node
        batch_size = 1  # Single graph (or use batch if provided)
        global_node = self.global_node_init.expand(batch_size, -1)  # [1, hidden_dim]
        
        # Message passing with global node
        for mp_layer in self.mp_layers:
            h, global_node = mp_layer(h, edge_index, global_node)
        
        # Project to [du_norm, dv_norm] in normalized space
        delta_norm = self.projection(h)  # [N, 2]
        du_norm = delta_norm[:, 0]  # [N]
        dv_norm = delta_norm[:, 1]  # [N]
        
        # Denormalize
        du = self.normalizer.denormalize_displacement(du_norm)
        dv = self.normalizer.denormalize_velocity(dv_norm)
        
        # Direct integration (network learns the dynamics)
        u_next = u + du
        v_next = v + dv
        
        # Apply boundary conditions
        u_next, v_next = self.boundary(u_next, v_next, bc_mask)
        
        # Return in DeepGCN format: [N, 2] tensor
        output = torch.stack([u_next, v_next], dim=-1)
        return output
    
    def rollout(self, u, v, f_sequence, edge_index, boundary_mask, steps):
        """
        Multi-step autoregressive rollout.
        Args:
            u: [N] initial displacement
            v: [N] initial velocity
            f_sequence: [steps, N] force trajectory
            edge_index: [2, E] graph connectivity
            boundary_mask: [N] boundary mask
            steps: number of timesteps to predict
        Returns:
            u_trajectory: [steps, N] predicted displacements
            v_trajectory: [steps, N] predicted velocities
        """
        u_traj = []
        v_traj = []
        
        u_current = u
        v_current = v
        
        for t in range(steps):
            f_t = f_sequence[t]
            u_next, v_next = self.forward(
                u_current, v_current, f_t, edge_index, boundary_mask
            )
            
            u_traj.append(u_next)
            v_traj.append(v_next)
            
            u_current = u_next
            v_current = v_next
        
        return torch.stack(u_traj), torch.stack(v_traj)


# ==================== LOSS FUNCTIONS ====================

def physics_informed_loss(
    interior_mask, 
    input_state, 
    output_state, 
    laplacian, 
    dt, 
    c, 
    k, 
    w1=1.0, 
    w2=1.0
):
    """
    Physics-informed loss based on wave equation discretization.
    Args:
        interior_mask: [N] boolean mask for interior nodes
        input_state: [N, 3] with [u, v, f] at time t
        output_state: [N, 2] with [u_next, v_next] at time t+dt
        laplacian: [N, N] sparse graph Laplacian matrix
        dt: timestep
        c: wave speed (c² = stiffness/mass)
        k: damping coefficient
        w1: weight for position residual
        w2: weight for velocity residual
    Returns:
        loss: total physics loss
        loss_1: position residual loss
        loss_2: velocity residual loss
    """
    u = input_state[:, 0]
    v = input_state[:, 1]
    f = input_state[:, 2]
    u_next = output_state[:, 0]
    v_next = output_state[:, 1]
    
    # Apply Laplacian
    if torch.is_tensor(laplacian):
        Lu = laplacian @ u
    else:
        # Handle scipy sparse matrix
        Lu = torch.tensor(laplacian @ u.cpu().numpy(), 
                         dtype=u.dtype, device=u.device)
    
    # PDE Residual 1: Position update (trapezoidal rule)
    # u_next = u + dt * (v + v_next) / 2
    pde_res1 = (u_next[interior_mask] - u[interior_mask] - 
                dt * (v[interior_mask] + v_next[interior_mask]) / 2)
    
    # PDE Residual 2: Velocity update (wave equation)
    # v_next = v + dt * (c² * ∇²u - k*v + f)
    pde_res2 = (v_next[interior_mask] - v[interior_mask] - 
                dt * (c**2 * Lu[interior_mask] - k * v[interior_mask] + 
                      f[interior_mask]))
    
    loss_1 = (pde_res1 ** 2).mean()
    loss_2 = (pde_res2 ** 2).mean()
    pde_loss = w1 * loss_1 + w2 * loss_2
    
    return pde_loss, float(loss_1.detach().item()), float(loss_2.detach().item())


def energy_loss(
    interior_mask,
    input_state,
    output_state,
    laplacian,
    dt,
    c,
    k,
    w=1.0
):
    """
    Energy conservation loss. Penalizes unphysical energy increase.
    Args:
        interior_mask: not used but kept for interface consistency
        input_state: [N, 3] with [u, v, f] at time t
        output_state: [N, 2] with [u_next, v_next] at time t+dt
        laplacian: not used but kept for interface consistency
        dt: timestep
        c: wave speed
        k: damping coefficient
        w: weight for energy loss
    Returns:
        weighted energy loss
    """
    u = input_state[:, 0]
    v = input_state[:, 1]
    force = input_state[:, 2]
    u_next = output_state[:, 0]
    v_next = output_state[:, 1]
    
    # Mechanical energy at t
    KE_current = 0.5 * torch.sum(v**2)
    PE_current = 0.5 * (c**2) * torch.sum(u**2)
    energy_current = KE_current + PE_current
    
    # Mechanical energy at t+dt
    KE_next = 0.5 * torch.sum(v_next**2)
    PE_next = 0.5 * (c**2) * torch.sum(u_next**2)
    energy_next = KE_next + PE_next
    
    # Work done by external forcing: W = ∫F·v dt ≈ F·v_avg·dt
    v_avg = (v + v_next) / 2
    work_forcing = torch.sum(force * v_avg) * dt
    
    # Energy dissipated by damping: D = ∫k*v² dt ≈ k*v_avg²*dt
    dissipation = k * torch.sum(v_avg**2) * dt
    
    # Energy balance: E_next = E_current + W - D
    energy_expected = energy_current + work_forcing - dissipation
    energy_violation = energy_next - energy_expected
    
    # Only penalize unphysical energy increase
    loss_energy = F.relu(energy_violation)
    
    return w * loss_energy


def combined_loss(
    u, v, f,
    u_next_pred, v_next_pred,
    u_next_true, v_next_true,
    interior_mask,
    laplacian,
    dt, c, k,
    lambda_mse=1.0,
    lambda_pde=0.1,
    lambda_energy=0.01,
    w1_pde=1.0,
    w2_pde=1.0
):
    """
    Combined loss function for training.
    Args:
        u, v, f: input state
        u_next_pred, v_next_pred: predicted next state
        u_next_true, v_next_true: ground truth next state
        interior_mask: boolean mask for interior nodes
        laplacian: graph Laplacian
        dt, c, k: physical parameters
        lambda_*: loss weights
        w1_pde, w2_pde: physics loss component weights
    Returns:
        total_loss, dict of individual losses
    """
    # MSE loss
    loss_mse_u = F.mse_loss(u_next_pred, u_next_true)
    loss_mse_v = F.mse_loss(v_next_pred, v_next_true)
    loss_mse = loss_mse_u + loss_mse_v
    
    # Physics-informed loss
    input_state = torch.stack([u, v, f], dim=-1)
    output_state = torch.stack([u_next_pred, v_next_pred], dim=-1)
    loss_pde, loss_pde1, loss_pde2 = physics_informed_loss(
        interior_mask, input_state, output_state, 
        laplacian, dt, c, k, w1_pde, w2_pde
    )
    
    # Energy loss
    loss_eng = energy_loss(
        interior_mask, input_state, output_state,
        laplacian, dt, c, k, w=1.0
    )
    
    # Total loss
    total_loss = (lambda_mse * loss_mse + 
                  lambda_pde * loss_pde + 
                  lambda_energy * loss_eng)
    
    losses_dict = {
        'total': total_loss.item(),
        'mse': loss_mse.item(),
        'mse_u': loss_mse_u.item(),
        'mse_v': loss_mse_v.item(),
        'pde': loss_pde.item(),
        'pde1': loss_pde1,
        'pde2': loss_pde2,
        'energy': loss_eng.item()
    }
    
    return total_loss, losses_dict


# ==================== TRAINING UTILITIES ====================

def train_step(model, optimizer, batch, laplacian, dt, c, k, loss_weights):
    """
    Single training step.
    Args:
        model: WaveGNN model
        optimizer: torch optimizer
        batch: dict with 'u', 'v', 'f', 'u_next', 'v_next', 
               'edge_index', 'boundary_mask'
        laplacian: graph Laplacian
        dt, c, k: physical parameters
        loss_weights: dict with lambda values
    Returns:
        losses_dict: dictionary of loss values
    """
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    u_next_pred, v_next_pred = model(
        batch['u'], batch['v'], batch['f'],
        batch['edge_index'], batch['boundary_mask']
    )
    
    # Compute loss
    interior_mask = ~batch['boundary_mask']
    total_loss, losses_dict = combined_loss(
        batch['u'], batch['v'], batch['f'],
        u_next_pred, v_next_pred,
        batch['u_next'], batch['v_next'],
        interior_mask, laplacian, dt, c, k,
        **loss_weights
    )
    
    # Backward pass
    total_loss.backward()
    optimizer.step()
    
    return losses_dict


def validate(model, val_loader, laplacian, dt, c, k, loss_weights):
    """
    Validation loop.
    Args:
        model: WaveGNN model
        val_loader: validation dataloader
        laplacian: graph Laplacian
        dt, c, k: physical parameters
        loss_weights: dict with lambda values
    Returns:
        avg_losses: dict of averaged loss values
    """
    model.eval()
    
    total_losses = {
        'total': 0, 'mse': 0, 'mse_u': 0, 'mse_v': 0,
        'pde': 0, 'pde1': 0, 'pde2': 0, 'energy': 0
    }
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            u_next_pred, v_next_pred = model(
                batch['u'], batch['v'], batch['f'],
                batch['edge_index'], batch['boundary_mask']
            )
            
            interior_mask = ~batch['boundary_mask']
            _, losses_dict = combined_loss(
                batch['u'], batch['v'], batch['f'],
                u_next_pred, v_next_pred,
                batch['u_next'], batch['v_next'],
                interior_mask, laplacian, dt, c, k,
                **loss_weights
            )
            
            for key in total_losses:
                total_losses[key] += losses_dict[key]
            num_batches += 1
    
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    return avg_losses


# ==================== MODEL FACTORY ====================

def create_wavegnn_from_config(cfg):
    """
    Create WaveGNN model from Hydra config (drop-in replacement for DeepGCN).
    
    Args:
        cfg: Hydra config object with model, dataset, and training sections
    
    Returns:
        model: WaveGNN instance compatible with existing training loop
    
    Example usage in train.py:
        # Replace:
        # model = DeepGCN(...)
        # With:
        from try_gno import create_wavegnn_from_config
        model = create_wavegnn_from_config(cfg)
    """
    model = WaveGNN(
        cfg_model=cfg.model
    )
    return model


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Example: Create model and dummy data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model parameters
    model = WaveGNN(
        hidden_dim=128,
        num_layers=3,
        dt=0.01,
        u_scale=0.04,
        v_scale=0.08,
        f_scale=3.0,
        dropout=0.1,
        in_channels=3,
        out_channels=2
    ).to(device)
    