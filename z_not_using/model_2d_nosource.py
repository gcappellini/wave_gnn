"""
Simplified PINN model for wave equation WITHOUT source term.
Subclass of PINNDeepONet_Wave2D with reduced complexity.
"""

import torch
import torch.nn as nn
import logging
from model_2d import PINNDeepONet_Wave2D

log = logging.getLogger(__name__)


class PINNDeepONet_Wave2D_NoSource(PINNDeepONet_Wave2D):
    """
    Simplified DeepONet for 2D damped wave equation WITHOUT source term.
    
    Inherits all infrastructure from parent (branch_ic, trunk, etc.)
    Only differs in PDE residual computation: no source term.
    
    PDE: u_tt + k*u_t - c²*(u_xx + u_yy) = 0
    """
    
    def __init__(self, cfg):
        """
        Initialize using parent class directly (cfg-based signature).
        
        Args:
            cfg: OmegaConf config object with all model parameters
        """
        # Initialize parent normally - inherits branch_ic, trunk, everything
        super().__init__(cfg)
        log.info("✓ Initialized PINNDeepONet_Wave2D_NoSource (source term disabled in PDE)")
    
    def compute_pde_residual(self, u0_sensors, v0_sensors, src_sensors, xyt, src_values=None):
        """
        Compute PDE residual WITHOUT source term.
        
        PDE: u_tt + k*u_t - c²*(u_xx + u_yy) = 0
        
        Args:
            u0_sensors: Initial velocity
            v0_sensors: Initial acceleration
            src_sensors: Measurements (not used in residual)
            xyt: Collocation points (n_colloc, 3)
            src_values: Source values (ignored)
        
        Returns:
            residual: PDE residual at collocation points (n_colloc,)
        """
        xyt_grad = xyt.clone().requires_grad_(True)
        u = self.forward(u0_sensors, v0_sensors, src_sensors, xyt_grad)
        
        # First derivatives
        grad_u = torch.autograd.grad(
            u, xyt_grad,
            torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        u_x = grad_u[:, 0]
        u_y = grad_u[:, 1]
        u_t = grad_u[:, 2]
        
        # Second derivatives in space
        u_xx = torch.autograd.grad(
            u_x, xyt_grad,
            torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0][:, 0]
        
        u_yy = torch.autograd.grad(
            u_y, xyt_grad,
            torch.ones_like(u_y),
            create_graph=True,
            retain_graph=True
        )[0][:, 1]
        
        # Second derivative in time
        u_tt = torch.autograd.grad(
            u_t, xyt_grad,
            torch.ones_like(u_t),
            create_graph=True
        )[0][:, 2]
        
        # PDE residual WITHOUT source: u_tt + k*u_t - c²*(u_xx + u_yy) = 0
        residual = u_tt + self.k * u_t - self.c**2 * (u_xx + u_yy)
        
        return residual
