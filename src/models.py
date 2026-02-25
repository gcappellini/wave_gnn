"""
Neural Network Models for DeepONet

Defines:
- MLP: Simple feedforward network
- DeepONet: Operator network (trunk + branch)
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple feedforward MLP network."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int, input_scale: float = 1.0):
        super().__init__()
        self.input_scale = input_scale
        # n_layers = total number of Linear layers
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 2):  # n_layers - 2 because we have 1 input + 1 output
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x * self.input_scale
        return self.net(x)


class DeepONet(nn.Module):
    """
    DeepONet operator network: trunk + branch (IC or force).
    
    Supports two problem types:
    - 'free_evolution': branch takes IC measurements
    - 'constant_force': branch takes force measurements
    """
    
    def __init__(self, 
                 trunk: MLP,
                 branch_ic: MLP = None,
                 branch_force: MLP = None,
                 problem_type: str = 'free_evolution',
                 wave_speed: float = 1.0,
                 damping: float = 1.0):
        """
        Args:
            trunk: Trunk network (x, y, t) -> n_modes
            branch_ic: Branch network for IC measurements (free_evolution)
            branch_force: Branch network for force measurements (constant_force)
            problem_type: 'free_evolution' or 'constant_force'
            wave_speed: Wave speed parameter c
            damping: Damping parameter k
        """
        super().__init__()
        self.trunk = trunk
        self.branch_ic = branch_ic
        self.branch_force = branch_force
        self.problem_type = problem_type
        self.c = wave_speed
        self.k = damping
        
        # Validate problem type
        if problem_type not in ['free_evolution', 'constant_force']:
            raise ValueError(f"problem_type must be 'free_evolution' or 'constant_force', got {problem_type}")
    
    def get_branch(self):
        """Get the active branch based on problem type."""
        if self.problem_type == 'free_evolution':
            if self.branch_ic is None:
                raise RuntimeError("branch_ic is None for free_evolution problem")
            return self.branch_ic
        else:  # constant_force
            if self.branch_force is None:
                raise RuntimeError("branch_force is None for constant_force problem")
            return self.branch_force
    
    def forward(self, measurements: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            measurements: IC or force measurements, shape (B, n_measurements)
            coords: Spatial-temporal coordinates (x, y, t), shape (B, 3)
        
        Returns:
            u: Predictions, shape (B,)
        """
        trunk_out = self.trunk(coords)  # (B, N_MODES)
        branch = self.get_branch()
        branch_out = branch(measurements)  # (B, N_MODES)
        u = torch.sum(trunk_out * branch_out, dim=1)  # (B,)
        return u
    
    def compute_pde_residual(self, 
                            measurements: torch.Tensor, 
                            xyt: torch.Tensor,
                            force_field: torch.Tensor = None) -> torch.Tensor:
        """
        Compute PDE residual.
        
        Free Evolution (no forcing):
            u_tt + k*u_t - c^2*(u_xx+u_yy) = 0
        
        Constant Force (with forcing):
            u_tt + k*u_t - c^2*(u_xx+u_yy) - f(x,y) = 0
        
        Args:
            measurements: IC or force measurements
            xyt: Coordinates (x, y, t) requiring gradients, shape (B, 3)
            force_field: Force field f(x, y), shape (B,). If None, assumes free evolution.
        
        Returns:
            residual: PDE residual, shape (B,)
        """
        xyt_grad = xyt.clone().requires_grad_(True)
        u = self.forward(measurements, xyt_grad)
        
        # First derivatives
        grad_u = torch.autograd.grad(
            u, xyt_grad,
            torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        u_x, u_y, u_t = grad_u[:, 0], grad_u[:, 1], grad_u[:, 2]
        
        # Second derivatives
        u_xx = torch.autograd.grad(u_x, xyt_grad, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0][:, 0]
        u_yy = torch.autograd.grad(u_y, xyt_grad, torch.ones_like(u_y), create_graph=True, retain_graph=True)[0][:, 1]
        u_tt = torch.autograd.grad(u_t, xyt_grad, torch.ones_like(u_t), create_graph=True)[0][:, 2]
        
        # Wave equation: u_tt + k*u_t - c^2*(u_xx+u_yy) = f (or 0 if no forcing)
        residual = u_tt + self.k * u_t - self.c**2 * (u_xx + u_yy)
        
        # Subtract forcing term if provided (constant_force case)
        if force_field is not None:
            residual = residual - force_field
        
        return residual
