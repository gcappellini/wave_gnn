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
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int):
        super().__init__()
        # n_layers = total number of Linear layers
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 2):  # n_layers - 2 because we have 1 input + 1 output
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class DeepONet(nn.Module):
    """DeepONet operator network: trunk + branch."""
    
    def __init__(self, trunk: MLP, branch: MLP, wave_speed: float = 1.0, damping: float = 1.0):
        super().__init__()
        self.trunk = trunk
        self.branch = branch
        self.c = wave_speed
        self.k = damping
    
    def forward(self, ic: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        trunk_out = self.trunk(coords)  # (B, N_MODES)
        branch_out = self.branch(ic)    # (B, N_MODES)
        u = torch.sum(trunk_out * branch_out, dim=1)  # (B,)
        return u
    
    def compute_pde_residual(self, ic: torch.Tensor, xyt: torch.Tensor) -> torch.Tensor:
        """Compute 2D wave equation residual: u_tt + k*u_t - c^2*(u_xx+u_yy)."""
        xyt_grad = xyt.clone().requires_grad_(True)
        u = self.forward(ic, xyt_grad)
        
        grad_u = torch.autograd.grad(
            u, xyt_grad,
            torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        u_x, u_y, u_t = grad_u[:, 0], grad_u[:, 1], grad_u[:, 2]
        
        u_xx = torch.autograd.grad(u_x, xyt_grad, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0][:, 0]
        u_yy = torch.autograd.grad(u_y, xyt_grad, torch.ones_like(u_y), create_graph=True, retain_graph=True)[0][:, 1]
        u_tt = torch.autograd.grad(u_t, xyt_grad, torch.ones_like(u_t), create_graph=True)[0][:, 2]
        
        residual = u_tt + self.k * u_t - self.c**2 * (u_xx + u_yy)
        return residual
