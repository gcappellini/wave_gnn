import numpy as np
import torch
import torch.nn as nn
import warnings
import logging
from adaptive_weights import AdaptiveLossWeights
import os
from plot_2d import plot_ic_reconstruction
warnings.filterwarnings('ignore')

log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def sample_coeffs(n, base_range, p=1.5, as_2d=False, device='cpu'):
    """
    Sample coefficients with decreasing range per mode.
    GPU optimized: directly creates tensors on target device.
    
    Args:
        n: Number of modes (for 1D) or modes per dimension (for 2D)
        base_range: (min, max) range for first mode
        p: Power law decay exponent
        as_2d: If True, returns (n, n) 2D tensor with different x,y modes
        device: torch.device to create tensors on
    
    Returns:
        1D tensor of shape (n,) or 2D tensor of shape (n, n)
    """
    coeffs = []
    for k in range(n):
        # Range decreases with k (first coeff full range, then shrinks)
        scale = 1.0 / (k + 1) ** p
        r0, r1 = base_range
        coeff_range = (r0 * scale, r1 * scale) if k > 0 else (r0, r1)
        coeffs.append(torch.FloatTensor(1).uniform_(*coeff_range).item())
    
    coeffs_1d = torch.tensor(coeffs, device=device)
    
    if as_2d:
        # Create 2D coefficient matrix: different values for each (k, l) mode
        coeffs_2d = torch.zeros(n, n, device=device)
        for k in range(n):
            for l in range(n):
                # Use product of 1D scales for each dimension
                scale_k = 1.0 / (k + 1) ** p
                scale_l = 1.0 / (l + 1) ** p
                scale_combined = scale_k * scale_l
                r0, r1 = base_range
                coeff_range = (r0 * scale_combined, r1 * scale_combined) if k > 0 and l > 0 else (r0, r1)
                coeffs_2d[k, l] = torch.FloatTensor(1).uniform_(*coeff_range).item()
        return coeffs_2d
    else:
        return coeffs_1d 

class BranchNet(nn.Module):
    """Branch network: encodes function inputs from sensor measurements"""
    def __init__(self, n_sensors, hidden_dim, output_dim, n_hidden_layers=2, residual=False, activation=None, fft_transform=None):
        super().__init__()
        self.fft_transform = fft_transform
        self.input_layer = nn.Linear(
            fft_transform.get_output_dim() if fft_transform is not None else n_sensors,
            hidden_dim
        )
        self.n_hidden_layers = n_hidden_layers
        self.residual = residual
        self.activation = activation if activation is not None else nn.LeakyReLU()
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    
    def forward(self, sensors):
        x = sensors
        if self.fft_transform is not None:
            x = self.fft_transform(x)
        x = self.input_layer(x)
        x = self.activation(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)
        x = self.output_layer(x)
        return x

class TrunkNet(nn.Module):
    def __init__(self, hidden_dim, output_dim, n_hidden_layers=4, residual=False, activation=None, fft_transform=None):
        super().__init__()
        self.fft_transform = fft_transform
        self.input_layer = nn.Linear(
            fft_transform.get_output_dim() if fft_transform is not None else 3,
            hidden_dim
        )
        self.n_hidden_layers = n_hidden_layers
        self.residual = residual
        self.activation = activation if activation is not None else nn.Tanh()
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, xyt):
        """
        Args:
            xyt: (n_points, 3) with columns [x, y, t]
        """
        x = xyt
        if self.fft_transform is not None:
            x = self.fft_transform(x)
        x = self.input_layer(x)
        x = self.activation(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)
        x = self.output_layer(x)
        return x


class PINNDeepONet_Wave2D(nn.Module):
    """
    Physics-Informed DeepONet for the 2D damped wave equation:
        u_tt + k*u_t = c²*(u_xx + u_yy) + f(x,y)
        
    with boundary conditions:
        u = 0 on all four edges
        
    and initial conditions:
        u(x,y,0) = a * sin(π*x) * sin(π*y)
        u_t(x,y,0) = b * sin(π*x) * sin(π*y)
        
    Args:
        n_sensors_ic: number of sensor points per dimension for IC (total: n²)
        n_sensors_src: number of sensor points per dimension for source (total: n²)
        branch_hidden: hidden layer size in branch networks
        trunk_hidden: hidden layer size in trunk network
        p: embedding dimension
        wave_speed: wave propagation speed c
        damping_coeff: damping coefficient k
    """
    
    def __init__(self, cfg):
        super().__init__()
        
        self.n_sensors_ic = cfg.model.n_sensors_ic
        self.n_sensors_src = cfg.model.n_sensors_src
        self.p = cfg.model.p
        self.c = cfg.model.wave_speed
        self.k = cfg.model.damping_coeff
        self.domain = [0.0, 1.0]  # Square domain [0,1] × [0,1]

        fft_trunk = FourierFeatureTransform(**(cfg.model.fft_trunk_args or {})) if cfg.model.use_fft_trunk else None
        
        # IC branch: takes BOTH u0 and v0 sensors (2D grids flattened)
        self.branch_ic = BranchNet(2 * cfg.model.n_sensors_ic * cfg.model.n_sensors_ic, cfg.model.branch_width, 2 * cfg.model.p, n_hidden_layers=cfg.model.branch_depth, activation=eval(cfg.model.branch_act))
        
        # Source branch: 2D grid flattened
        self.branch_source = BranchNet(cfg.model.n_sensors_src * cfg.model.n_sensors_src, cfg.model.branch_width, cfg.model.p, n_hidden_layers=cfg.model.branch_depth, activation=eval(cfg.model.branch_act))
        
        # Trunk network for (x, y, t)
        self.trunk = TrunkNet(cfg.model.trunk_width, cfg.model.p, n_hidden_layers=cfg.model.trunk_depth, fft_transform=fft_trunk, activation=eval(cfg.model.trunk_act))
        
        # Bias term
        self.bias = nn.Parameter(torch.zeros(1))
        self.fusion = cfg.model.fusion

        if cfg.model.fusion in ['comb_mlp', 'pure_mlp', 'hc']:
            self.fusion_mlp = nn.Sequential(
                nn.Linear(3 * self.p, 2 * self.p), # Hidden layer (e.g., 1024 neurons)
                nn.Tanh(),                         # Use Tanh as it worked best for your Branch
                nn.Linear(2 * self.p, self.p)      # Project back to rank p
            )
        if cfg.model.fusion == 'hc':
            self.trunk_spatial = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, self.p)
        )

        
        # Fixed sensor locations (2D grid)
        x_1d = torch.linspace(self.domain[0], self.domain[1], cfg.model.n_sensors_ic)
        y_1d = torch.linspace(self.domain[0], self.domain[1], cfg.model.n_sensors_ic)
        X_grid, Y_grid = torch.meshgrid(x_1d, y_1d, indexing='ij')
        self.register_buffer('sensor_x_ic', X_grid.flatten())
        self.register_buffer('sensor_y_ic', Y_grid.flatten())
        
        x_1d_src = torch.linspace(self.domain[0], self.domain[1], cfg.model.n_sensors_src)
        y_1d_src = torch.linspace(self.domain[0], self.domain[1], cfg.model.n_sensors_src)
        X_grid_src, Y_grid_src = torch.meshgrid(x_1d_src, y_1d_src, indexing='ij')
        self.register_buffer('sensor_x_src', X_grid_src.flatten())
        self.register_buffer('sensor_y_src', Y_grid_src.flatten())
    
    def predict_ic(self, u0_sensors, v0_sensors, xyt):
            """
            Pre-training helper for Standard DeepONet (Sum/MLP Mixing).
            Predicts u0 and v0 using the MAIN Trunk at t=0.
            BC factor is applied to enforce zero displacement and velocity at boundaries.
            """
            # 1. Branch Encoding
            # Same as before: encode inputs to get latent vectors
            if u0_sensors.dim() == 1: u0_sensors = u0_sensors.unsqueeze(0)
            if v0_sensors.dim() == 1: v0_sensors = v0_sensors.unsqueeze(0)
                
            ic_concat = torch.cat([u0_sensors, v0_sensors], dim=1)
            ic_encoded = self.branch_ic(ic_concat)
            b_u = ic_encoded[:, :self.p]
            b_v = ic_encoded[:, self.p:]

            # 2. Trunk Evaluation at t=0
            # We must enforce t=0 for the IC prediction context
            # Check if input xyt already has t=0, or force it.
            # Ideally, pass xyt where t is explicitly 0.0
            
            if xyt.dim() == 2:
                tau = self.trunk(xyt) # Shape: (n_points, p)
                # Reconstruct: (batch, p) @ (p, n_points)
                u0_pred = torch.matmul(b_u, tau.T)
                v0_pred = torch.matmul(b_v, tau.T)
                
            elif xyt.dim() == 3:
                tau = self.trunk(xyt) # Shape: (batch, n_points, p)
                # Reconstruct: Element-wise sum
                u0_pred = torch.sum(b_u.unsqueeze(1) * tau, dim=-1)
                v0_pred = torch.sum(b_v.unsqueeze(1) * tau, dim=-1)

            # 3. Apply BC factor for hard constraint enforcement
            # Extract spatial coordinates and apply boundary condition factor
            x = xyt[..., 0:1]  # Shape: (..., 1)
            y = xyt[..., 1:2]  # Shape: (..., 1)
            
            # BC factor: (x(1-x)y(1-y))^2, rescaled by 16.0
            bc_factor_sq = (x * (1.0 - x) * y * (1.0 - y)) ** 2
            u0_pred_rescaled = 16.0 * u0_pred
            v0_pred_rescaled = 16.0 * v0_pred
            
            u0_pred = bc_factor_sq * u0_pred_rescaled
            v0_pred = bc_factor_sq * v0_pred_rescaled

            return u0_pred, v0_pred
    
    def forward(self, u0_sensors, v0_sensors, src_sensors, xyt):
            """
            Args:
                u0_sensors: (n_sensors²,) - displacement IC on 2D grid (flattened)
                v0_sensors: (n_sensors²,) - velocity IC on 2D grid (flattened)
                src_sensors: (n_sensors²,) - source on 2D grid (flattened)
                xyt: (n_points, 3) spatiotemporal coordinates [x, y, t]
            
            Returns:
                u: (n_points,) - displacement
            """
            single_sample = u0_sensors.dim() == 1
            if single_sample:
                u0_sensors = u0_sensors.unsqueeze(0)
                v0_sensors = v0_sensors.unsqueeze(0)
                src_sensors = src_sensors.unsqueeze(0)
            
            # 1. Coordinate Extraction (Needed for HC and BCs)
            x = xyt[:, 0]
            y = xyt[:, 1]
            t = xyt[:, 2]

            # 2. Branch Encoding (Common to all methods)
            ic_concat = torch.cat([u0_sensors, v0_sensors], dim=1)
            ic_encoded = self.branch_ic(ic_concat)
            
            b_u = ic_encoded[:, :self.p]
            b_v = ic_encoded[:, self.p:]
            b_src = self.branch_source(src_sensors)
            
            # 3. Fusion Strategy Switch
            if self.fusion == 'hc':
                # --- HARD CONSTRAINT ANSATZ (Neural Ansatz) ---
                
                # A. Static Reconstruction (u0, v0)
                # Use separate Spatial Trunk (inputs: x,y)
                xy = xyt[:, :2]
                tau_x = self.trunk_spatial(xy)  # Shape: (n_points, p)
                
                # Reconstruct ICs: (batch, p) @ (p, n_points) -> (batch, n_points)
                u0_pred = torch.matmul(b_u, tau_x.T)
                v0_pred = torch.matmul(b_v, tau_x.T)
                
                # B. Dynamic Correction Part
                # We use the Residual MLP mixing for the dynamic coefficients
                b_concat = torch.cat([b_u, b_v, b_src], dim=1)
                b_correction = self.fusion_mlp(b_concat)
                b_dynamic = b_u + b_v + b_src + b_correction
                
                # Spatiotemporal Trunk (inputs: x,y,t)
                tau = self.trunk(xyt)  # Shape: (n_points, p)
                
                # Compute dynamic term
                u_dynamic = torch.matmul(b_dynamic, tau.T) + self.bias
                
                # C. The Ansatz Combination
                # u = u0 + t*v0 + (1 - exp(-t)) * u_dynamic
                time_factor = 1.0 - torch.exp(-t)
                
                u_net = u0_pred + (t * v0_pred) + (time_factor * u_dynamic)

            else:
                # --- STANDARD DEEPONET (Soft Constraints) ---
                
                if self.fusion == 'comb_mlp':
                    b_concat = torch.cat([b_u, b_v, b_src], dim=1)
                    b_correction = self.fusion_mlp(b_concat)
                    b_combined = b_u + b_v + b_src + b_correction
                if self.fusion == 'pure_mlp':
                    b_concat = torch.cat([b_u, b_v, b_src], dim=1)
                    b_combined = self.fusion_mlp(b_concat)
                elif self.fusion == 'sum':
                    b_combined = b_u + b_v + b_src
                
                # Encode spatiotemporal locations
                tau = self.trunk(xyt)
                
                # Standard DeepONet operation
                u_net = torch.matmul(b_combined, tau.T) + self.bias
            
            # 4. Global Boundary Condition Enforcement (Sharp polynomial BC)
            # Uses polynomial: x(1-x)*y(1-y) which ensures:
            #   - u = 0 at all four boundaries (x=0, x=1, y=0, y=1)
            #   - du/dx = 0 and du/dy = 0 at boundaries (zero normal derivative)
            # This is sharper than sin(πx)*sin(πy) and provides stronger constraints

            # 1. New BC factor with zero derivative at boundaries
            bc_factor_sq = (x * (1.0 - x))**2 * (y * (1.0 - y))**2

            # 2. Rescale u_net by 16.0 to compensate for the smaller bc_factor_sq peak (0.0625)
            u_net_rescaled = 16.0 * u_net
            
            # Ensure dimensionality matches for broadcasting
            if single_sample:
                u = bc_factor_sq * u_net_rescaled
            else:
                u = bc_factor_sq.unsqueeze(0) * u_net_rescaled
            
            return u.squeeze(0) if single_sample else u
    
    def get_velocity(self, u0_sensors, v0_sensors, src_sensors, xyt):
        """
        Compute time derivative of displacement (velocity) at inference.
        
        Args:
            u0_sensors: (n_sensors²,) - displacement IC on 2D grid (flattened)
            v0_sensors: (n_sensors²,) - velocity IC on 2D grid (flattened)
            src_sensors: (n_sensors²,) - source on 2D grid (flattened)
            xyt: (n_points, 3) spatiotemporal coordinates [x, y, t]
        
        Returns:
            u_t: (n_points,) - time derivative of displacement (velocity field)
        """
        # Ensure xyt requires gradients for differentiation
        xyt_grad = xyt.clone().requires_grad_(True)
        
        # Forward pass
        u = self.forward(u0_sensors, v0_sensors, src_sensors, xyt_grad)
        
        # Compute time derivative using autograd
        u_t = torch.autograd.grad(
            u, xyt_grad,
            torch.ones_like(u),
            create_graph=False,
            retain_graph=False
        )[0][:, 2]  # Extract time component (column 2)
        
        return u_t
    
    def compute_pde_residual(self, u0_sensors, v0_sensors, src_sensors, xyt, src_values):
        """
        Compute PDE residual: R = u_tt + k*u_t - c²*(u_xx + u_yy) - f(x,y)
        """
        xyt_grad = xyt.clone().requires_grad_(True)
        
        u = self.forward(u0_sensors, v0_sensors, src_sensors, xyt_grad)
        
        # First derivatives
        grad_u = torch.autograd.grad(u, xyt_grad, torch.ones_like(u), create_graph=True)[0]
        u_x = grad_u[:, 0]
        u_y = grad_u[:, 1]
        u_t = grad_u[:, 2]
        
        # Second derivatives in space
        u_xx = torch.autograd.grad(u_x, xyt_grad, torch.ones_like(u_x), create_graph=True)[0][:, 0]
        u_yy = torch.autograd.grad(u_y, xyt_grad, torch.ones_like(u_y), create_graph=True)[0][:, 1]
        
        # Second derivative in time
        u_tt = torch.autograd.grad(u_t, xyt_grad, torch.ones_like(u_t), create_graph=True)[0][:, 2]
        
        # PDE residual
        residual = u_tt + self.k * u_t - self.c**2 * (u_xx + u_yy) - src_values
        
        return residual
    
    def generate_ic_sine_series(self, coeffs, x=None, y=None):
        """
        Generate initial condition as a 2D sine series:
            - Scalar: u(x, y) = coeff * sin(π*x) * sin(π*y)
            - 1D array: u(x, y) = sum_k coeffs[k] * sin((k+1)*π*x) * sin((k+1)*π*y)
            - 2D array: u(x, y) = sum_k,l coeffs[k,l] * sin((k+1)*π*x) * sin((l+1)*π*y)
        
        Args:
            coeffs: scalar, 1D tensor (shape: n,) or 2D tensor (shape: n, n)
            x: tensor of x sensor locations (default: self.sensor_x_ic)
            y: tensor of y sensor locations (default: self.sensor_y_ic)
        
        Returns:
            u0_sensors: tensor of shape (len(x),)
        """
        if x is None:
            x = self.sensor_x_ic
        if y is None:
            y = self.sensor_y_ic
        
        # Ensure coeffs is a tensor
        coeffs = torch.as_tensor(coeffs, dtype=x.dtype, device=x.device)
        
        # Handle scalar coefficient
        if coeffs.dim() == 0:
            u = coeffs * torch.sin(np.pi * x) * torch.sin(np.pi * y)
        
        # Handle 1D array of coefficients (symmetric x-y modes)
        elif coeffs.dim() == 1:
            u = torch.zeros_like(x)
            for k in range(coeffs.shape[0]):
                u = u + coeffs[k] * torch.sin((k + 1) * np.pi * x) * torch.sin((k + 1) * np.pi * y)
        
        # Handle 2D array of coefficients (independent x-y modes)
        elif coeffs.dim() == 2:
            u = torch.zeros_like(x)
            for k in range(coeffs.shape[0]):
                for l in range(coeffs.shape[1]):
                    u = u + coeffs[k, l] * torch.sin((k + 1) * np.pi * x) * torch.sin((l + 1) * np.pi * y)
        
        else:
            raise ValueError(f"coeffs must be 0D (scalar), 1D, or 2D tensor, got {coeffs.dim()}D")
        
        return u
    
    def generate_source(self, source_type='gaussian', amplitude=15.0, center_x=0.5, center_y=0.5):
        """Generate 2D source on sensor grid"""
        x = self.sensor_x_src
        y = self.sensor_y_src
        
        if source_type == 'gaussian':
            width = 0.3
            r_sq = ((x - center_x)**2 + (y - center_y)**2)
            src = amplitude * torch.exp(-r_sq / width**2)
        elif source_type == 'zero':
            src = torch.zeros_like(x)
        else:
            raise ValueError(f"Unknown source type: {source_type}")
        
        return src
    
    def source_function(self, x, y, source_type='gaussian', amplitude=15.0, center_x=0.5, center_y=0.5):
        """Evaluate source at arbitrary (x,y) locations"""
        if source_type == 'gaussian':
            width = 0.3
            r_sq = ((x - center_x)**2 + (y - center_y)**2)
            f = amplitude * torch.exp(-r_sq / width**2)
        elif source_type == 'zero':
            f = torch.zeros_like(x)
        else:
            raise ValueError(f"Unknown source type: {source_type}")
        
        return f
    
    def _compute_batch_losses(self, u0_sensors, v0_sensors, src_sensors, 
                              n_colloc, n_ic, T_max, source_type, source_amplitude, 
                              cx, cy, a_coeffs, b_coeffs):
        """
        Compute PDE and IC losses for a single batch (IC/source configuration).
        GPU optimized: Tensors created on same device as model parameters.
        
        Returns:
            loss_pde, loss_ic_u, loss_ic_v (scalars)
        """
        # Get device from model parameters
        device = next(self.parameters()).device
        
        # Sample 3D collocation points (on GPU)
        x_colloc = self.domain[0] + (self.domain[1] - self.domain[0]) * torch.rand(n_colloc, device=device)
        y_colloc = self.domain[0] + (self.domain[1] - self.domain[0]) * torch.rand(n_colloc, device=device)
        t_colloc = T_max * torch.rand(n_colloc, device=device)
        xyt_colloc = torch.stack([x_colloc, y_colloc, t_colloc], dim=1)
        
        # Source values at collocation points
        src_colloc = self.source_function(x_colloc, y_colloc, source_type, source_amplitude, cx, cy)
        
        # PDE Loss
        residual = self.compute_pde_residual(u0_sensors, v0_sensors, src_sensors, 
                                             xyt_colloc, src_colloc)
        loss_pde = torch.mean(residual ** 2)
        
        # IC grid (fixed across batches within epoch, on GPU)
        x_ic_1d = torch.linspace(self.domain[0], self.domain[1], n_ic, device=device)
        y_ic_1d = torch.linspace(self.domain[0], self.domain[1], n_ic, device=device)
        X_ic, Y_ic = torch.meshgrid(x_ic_1d, y_ic_1d, indexing='ij')
        x_ic = X_ic.flatten()
        y_ic = Y_ic.flatten()
        t_ic = torch.zeros_like(x_ic)
        xyt_ic = torch.stack([x_ic, y_ic, t_ic], dim=1)
        
        # Displacement IC Loss
        u_ic_pred = self.forward(u0_sensors, v0_sensors, src_sensors, xyt_ic)
        u_ic_true = self.generate_ic_sine_series(a_coeffs, x_ic, y_ic)
        loss_ic_u = torch.mean((u_ic_pred - u_ic_true) ** 2)
        
        # Velocity IC Loss
        xyt_ic_grad = xyt_ic.clone().requires_grad_(True)
        u_ic_grad = self.forward(u0_sensors, v0_sensors, src_sensors, xyt_ic_grad)
        u_t_ic_pred = torch.autograd.grad(u_ic_grad, xyt_ic_grad,
                                          torch.ones_like(u_ic_grad),
                                          create_graph=True)[0][:, 2]
        v_ic_true = self.generate_ic_sine_series(b_coeffs, x_ic, y_ic)
        loss_ic_v = torch.mean((u_t_ic_pred - v_ic_true) ** 2)
        
        return loss_pde, loss_ic_u, loss_ic_v
    
    def _train_epoch_batched(self, n_batches, n_colloc, n_ic, T_max, 
                             a_range, b_range, source_type, source_amplitude,
                             center_x, center_y, center_x_range, center_y_range,
                             n_ic_u, n_ic_v, weights, optimizer, max_grad_norm, device=None):
        """
        Execute one epoch with mini-batch training: sample multiple IC/source configs,
        compute losses, accumulate gradients, then update once per epoch (Adam).
        GPU optimized: all tensors created on device.
        
        Returns:
            epoch_loss_pde, epoch_loss_ic_u, epoch_loss_ic_v (averaged over batches)
        """
        if device is None:
            device = next(self.parameters()).device
            
        loss_pde_accum = 0.0
        loss_ic_u_accum = 0.0
        loss_ic_v_accum = 0.0
        
        for batch_idx in range(n_batches):
            # Sample IC and source for this batch (on GPU)
            a_coeffs = sample_coeffs(n_ic_u, a_range, p=3.0, as_2d=True, device=device)
            b_coeffs = sample_coeffs(n_ic_v, b_range, as_2d=True, device=device)
            
            # Sample source center
            if center_x_range is not None and center_y_range is not None and source_type == 'gaussian':
                cx = center_x_range[0] + (center_x_range[1] - center_x_range[0]) * torch.rand(1).item()
                cy = center_y_range[0] + (center_y_range[1] - center_y_range[0]) * torch.rand(1).item()
            else:
                cx, cy = center_x, center_y
            
            # Generate sensors
            u0_sensors = self.generate_ic_sine_series(a_coeffs)
            v0_sensors = self.generate_ic_sine_series(b_coeffs)
            src_sensors = self.generate_source(source_type, source_amplitude, cx, cy)
            
            # Compute batch losses
            loss_pde, loss_ic_u, loss_ic_v = self._compute_batch_losses(
                u0_sensors, v0_sensors, src_sensors,
                n_colloc, n_ic, T_max, source_type, source_amplitude, cx, cy,
                a_coeffs, b_coeffs
            )
            
            # Weighted total loss
            loss_batch = (weights['PDE'] * loss_pde + 
                          weights['IC_u'] * loss_ic_u + 
                          weights['IC_v'] * loss_ic_v)
            
            # Backward (accumulates gradients across batches)
            loss_batch.backward()
            
            # Accumulate losses
            loss_pde_accum += loss_pde.item()
            loss_ic_u_accum += loss_ic_u.item()
            loss_ic_v_accum += loss_ic_v.item()
        
        # Gradient clipping and single weight update per epoch
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        
        # Return averaged losses
        return (loss_pde_accum / n_batches, 
                loss_ic_u_accum / n_batches, 
                loss_ic_v_accum / n_batches)
    
    def train_pinn(self, cfg, output_fold=os.path.join(SCRIPT_DIR, 'logs_multibranch_wave'), device=None, gt_data=None):
        """
        Train PINN-DeepONet for 2D wave equation with early stopping based on test loss
        
        Args:
            cfg: Config object containing all training parameters
            output_fold: Output directory for checkpoints and logs
            device: torch.device to train on (cuda or cpu)
        """
        # Set device if provided
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self = self.to(device)
        # Extract training parameters from config
        n_epochs = cfg.training.n_epochs
        n_colloc = cfg.data.n_colloc
        n_ic = cfg.data.n_ic
        lr = cfg.training.lr
        strategy = cfg.model.strategy
        use_lbfgs_finetune = cfg.training.get('use_lbfgs_finetune', False)
        lbfgs_n_epochs = cfg.training.get('lbfgs_n_epochs', 100)
        lbfgs_max_iter = cfg.training.get('lbfgs_max_iter', 20)
        lbfgs_lr = cfg.training.get('lbfgs_lr', lr)
        w_pde = cfg.model.w_pde
        w_ic_u = cfg.model.w_ic_u
        w_ic_v = cfg.model.w_ic_v
        val_interval = cfg.training.val_interval
        early_stopping_patience = cfg.training.es_patience
        n_test_cases = cfg.training.n_test_cases
        test_seed = cfg.training.test_seed
        lr_scheduler_gamma = cfg.training.lr_scheduler_gamma
        lr_scheduler_step = cfg.training.lr_scheduler_step
        max_grad_norm = cfg.training.max_grad_norm
        n_batches = cfg.training.n_batches
        log_interval = cfg.training.log_interval
        
        # Extract IC/source parameters from config
        a_range = tuple(cfg.data.a_range)
        b_range = tuple(cfg.data.b_range)
        n_ic_u = cfg.data.n_ic_u
        n_ic_v = cfg.data.n_ic_v
        T_max = cfg.data.T_max
        source_type = cfg.data.source_type
        source_amplitude = cfg.data.source_amplitude
        center_x = cfg.data.center_x
        center_y = cfg.data.center_y
        center_x_range = tuple(cfg.data.center_x_range) if cfg.data.center_x_range else None
        center_y_range = tuple(cfg.data.center_y_range) if cfg.data.center_y_range else None
        pretrain_ic = cfg.training.pretrain_ic 
        n_epochs_pretrain = cfg.training.n_epochs_pretrain
        
        log.info(f"Epochs: {n_epochs}")
        log.info(f"Collocation points: {n_colloc}")
        log.info(f"a range: {a_range}, b range: {b_range}")
        if center_x_range is not None and center_y_range is not None:
            log.info(f"Source: {source_type}, amplitude: {source_amplitude}, center x range: {center_x_range}, center y range: {center_y_range}")
        else:
            log.info(f"Source: {source_type}, amplitude: {source_amplitude}, center: ({center_x}, {center_y})")
        log.info(f"Wave speed: {self.c}, Damping: {self.k}")
        log.info(f"Early stopping: patience={early_stopping_patience}, val_interval={val_interval}, n_test_cases={n_test_cases}")
        
        # Generate deterministic test set with fixed seed (on GPU)
        test_rng = np.random.RandomState(test_seed)
        test_cases = []
        for _ in range(n_test_cases):
            a_test = sample_coeffs(n_ic_u, a_range, p=3.0, as_2d=True, device=device)
            b_test = sample_coeffs(n_ic_v, b_range, p=3.0, as_2d=True, device=device)
            if center_x_range is not None and center_y_range is not None and source_type == 'gaussian':
                cx_test = test_rng.uniform(center_x_range[0], center_x_range[1])
                cy_test = test_rng.uniform(center_y_range[0], center_y_range[1])
            else:
                cx_test, cy_test = center_x, center_y
            test_cases.append({
                'a_coeffs': a_test,
                'b_coeffs': b_test,
                'center_x': cx_test,
                'center_y': cy_test
            })
        # ==============================================================================
        # PHASE 1: IC Pre-training (The "Freeze" Strategy)
        # ==============================================================================
        # Assuming n_epochs_pretrain is set (e.g., 1000)
        pretrain_history = None  # Will be populated if pretrain_ic is True
        if pretrain_ic: 
            log.info(f"=== PHASE 1: Pre-training IC Reconstruction for {n_epochs_pretrain} epochs (Standard DeepONet) ===")
            pretrain_history = {'loss_ic_u': [], 'loss_ic_v': [], 'test_metric': [], 'test_epochs': []}
            
            # 1. Freeze Dynamic Components (All layers initially)
            for param in self.parameters():
                param.requires_grad = False
            
            # 2. Unfreeze components for Standard DeepONet IC Reconstruction:
            trainable_params = []
            
            # a. Unfreeze Branch IC
            if hasattr(self, 'branch_ic'):
                for param in self.branch_ic.parameters():
                    param.requires_grad = True
                    trainable_params.append(param)
                    
            # b. Unfreeze the MAIN TRUNK (The trunk must learn the basis functions)
            if hasattr(self, 'trunk'): 
                for param in self.trunk.parameters():
                    param.requires_grad = True
                    trainable_params.append(param)
            
            # 3. Optimizer for Phase 1 (Tracks only the unfrozen parameters)
            optimizer_pre = torch.optim.Adam(trainable_params, lr=lr)
            
            # Track best pretraining model
            best_pretrain_loss = float('inf')
            best_pretrain_model_state = None
            best_pretrain_epoch = 0
            
            # 4. Pre-training Loop
            for epoch_pre in range(n_epochs_pretrain):
                loss_ic_u_accum = 0.0
                loss_ic_v_accum = 0.0
                
                for _ in range(n_batches):
                    # Sample random IC coefficients (on GPU)
                    a_coeffs_pre = sample_coeffs(n_ic_u, a_range, p=3.0, as_2d=True, device=device)
                    b_coeffs_pre = sample_coeffs(n_ic_v, b_range, as_2d=True, device=device)
                    
                    # Generate Input Sensors
                    u0_sensors_pre = self.generate_ic_sine_series(a_coeffs_pre)
                    v0_sensors_pre = self.generate_ic_sine_series(b_coeffs_pre)
                    
                    # Generate Target Ground Truth Points (IC Grid)
                    x_ic_1d = torch.linspace(self.domain[0], self.domain[1], n_ic, device=device)
                    y_ic_1d = torch.linspace(self.domain[0], self.domain[1], n_ic, device=device)
                    X_ic, Y_ic = torch.meshgrid(x_ic_1d, y_ic_1d, indexing='ij')
                    x_ic = X_ic.flatten()
                    y_ic = Y_ic.flatten()
                    t_ic = torch.zeros_like(x_ic)
                    xyt_ic = torch.stack([x_ic, y_ic, t_ic], dim=1)
                    
                    # Compute Ground Truth Field (u0 and v0)
                    u0_true = self.generate_ic_sine_series(a_coeffs_pre, x_ic, y_ic)
                    v0_true = self.generate_ic_sine_series(b_coeffs_pre, x_ic, y_ic)
                    
                    # Forward Pass: Compute reconstruction (requires custom logic in forward or helper)
                    # NOTE: Since we reverted to the standard model, you MUST ensure
                    # your self.predict_ic (or a similar block) now calls self.trunk (not trunk_spatial).
                    u0_pred, v0_pred = self.predict_ic(u0_sensors_pre, v0_sensors_pre, xyt_ic)
                    
                    # Loss: Compute separate losses for Displacement and Velocity reconstruction
                    loss_ic_u = torch.mean((u0_pred - u0_true)**2)
                    loss_ic_v = torch.mean((v0_pred - v0_true)**2)
                    loss_pre = loss_ic_u + loss_ic_v
                    
                    loss_pre.backward()
                    loss_ic_u_accum += loss_ic_u.item()
                    loss_ic_v_accum += loss_ic_v.item()
                
                # Update
                optimizer_pre.step()
                optimizer_pre.zero_grad()
                
                # Store pretraining history (separate IC_u and IC_v losses)
                loss_ic_u_avg = loss_ic_u_accum / n_batches
                loss_ic_v_avg = loss_ic_v_accum / n_batches
                pretrain_history['loss_ic_u'].append(loss_ic_u_avg)
                pretrain_history['loss_ic_v'].append(loss_ic_v_avg)
                
                # Compute test metric on test set every val_interval epochs (using IC components only)
                if (epoch_pre + 1) % val_interval == 0:
                    pretrain_test_metric = 0.0
                    for test_case in test_cases:
                        u0_test_pre = self.generate_ic_sine_series(test_case['a_coeffs'])
                        v0_test_pre = self.generate_ic_sine_series(test_case['b_coeffs'])
                        
                        # Evaluate IC reconstruction on test set
                        with torch.no_grad():
                            x_test_ic_1d = torch.linspace(self.domain[0], self.domain[1], n_ic, device=device)
                            y_test_ic_1d = torch.linspace(self.domain[0], self.domain[1], n_ic, device=device)
                            X_test_ic, Y_test_ic = torch.meshgrid(x_test_ic_1d, y_test_ic_1d, indexing='ij')
                            x_test_ic = X_test_ic.flatten()
                            y_test_ic = Y_test_ic.flatten()
                            t_test_ic = torch.zeros_like(x_test_ic)
                            xyt_test_ic = torch.stack([x_test_ic, y_test_ic, t_test_ic], dim=1)
                            
                            u0_test_pred, v0_test_pred = self.predict_ic(u0_test_pre, v0_test_pre, xyt_test_ic)
                            u0_test_true = self.generate_ic_sine_series(test_case['a_coeffs'], x_test_ic, y_test_ic)
                            v0_test_true = self.generate_ic_sine_series(test_case['b_coeffs'], x_test_ic, y_test_ic)
                            
                            loss_test_ic_u = torch.mean((u0_test_pred - u0_test_true)**2)
                            loss_test_ic_v = torch.mean((v0_test_pred - v0_test_true)**2)
                            pretrain_test_metric += (loss_test_ic_u + loss_test_ic_v).item()
                    
                    pretrain_test_metric /= n_test_cases
                    pretrain_history['test_metric'].append(pretrain_test_metric)
                    pretrain_history['test_epochs'].append(epoch_pre + 1)
                    
                    # Save best pretraining model
                    if pretrain_test_metric < best_pretrain_loss:
                        best_pretrain_loss = pretrain_test_metric
                        best_pretrain_model_state = {key: value.cpu().clone() for key, value in self.state_dict().items()}
                        best_pretrain_epoch = epoch_pre
                        log.info(f"  ✓ Pretrain test metric improved to {pretrain_test_metric:.6e} at epoch {epoch_pre+1}")
                        
                        # Save checkpoint immediately when new best is achieved (overwrites previous best)
                        checkpoint_path = os.path.join(output_fold, 'model_pretrain.pth')
                        torch.save({
                            'epoch': epoch_pre,
                            'model_state_dict': self.state_dict(),
                            'test_metric': pretrain_test_metric,
                            'pretrain_history': pretrain_history,
                            'config': cfg,
                        }, checkpoint_path)
                    if (epoch_pre + 1) % (val_interval * 5) == 0:
                        log.info(f"  [Pre-train] Epoch {epoch_pre+1}/{n_epochs_pretrain} | IC_u Loss: {loss_ic_u_avg:.6e} | IC_v Loss: {loss_ic_v_avg:.6e} | Test Metric: {pretrain_test_metric:.6e}")
                elif (epoch_pre + 1) % 100 == 0:
                    log.info(f"  [Pre-train] Epoch {epoch_pre+1}/{n_epochs_pretrain} | IC_u Loss: {loss_ic_u_avg:.6e} | IC_v Loss: {loss_ic_v_avg:.6e}")
            
            log.info("=== PHASE 1 COMPLETE. ICs learned. Unfreezing for Phase 2. ===")
            
            # Restore best pretraining model if one was saved
            if best_pretrain_model_state is not None:
                self.load_state_dict(best_pretrain_model_state)
                log.info(f"Restored best pretraining model from epoch {best_pretrain_epoch+1} with test metric {best_pretrain_loss:.6e}")
            plot_ic_reconstruction(
                self, 
                test_cases[0], 
                save_path=os.path.join(output_fold, 'ic_pretrain_diagnostic.png'),
                gt_data=gt_data
            )
            # 5. Unfreeze Everything for Phase 2
            for param in self.parameters():
                param.requires_grad = True


        initial_weights = {'PDE': w_pde, 'IC_u': w_ic_u, 'IC_v': w_ic_v}
        adaptive_weights = AdaptiveLossWeights(
            initial_weights=initial_weights,
            strategy=strategy,  
        )

        # Initialize Adam optimizer (LBFGS will be applied as optional fine-tuning after)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=lr_scheduler_gamma
        )
        log.info(f"Using Adam optimizer with lr={lr}")
        if use_lbfgs_finetune:
            log.info(f"  LBFGS fine-tuning enabled: {lbfgs_n_epochs} epochs, max_iter={lbfgs_max_iter}, lr={lbfgs_lr}")
        
        history = {'total': [], 'pde': [], 'ic_u': [], 'ic_v': [], 'test_metric': [], 'test_epochs': []}
        
        # Track best model based on TEST loss
        best_test_loss = float('inf')
        best_model_state = None
        best_epoch = 0
        best_test_losses = {'pde': float('inf'), 'ic_u': float('inf'), 'ic_v': float('inf')}
        epochs_without_improvement = 0
        
        # Print initial weights only for fixed strategy
        if strategy == 'fixed':
            log.info(f"Initial Weights -> PDE: {initial_weights['PDE']:.2e}, IC_u: {initial_weights['IC_u']:.2e}, IC_v: {initial_weights['IC_v']:.2e}")
        
        for epoch in range(n_epochs):
            # Get weights for this epoch
            if strategy != 'fixed':
                # For adaptive weights, need a single context (use first batch config)
                a_coeffs_temp = sample_coeffs(n_ic_u, a_range, p=3.0, as_2d=True, device=device)
                b_coeffs_temp = sample_coeffs(n_ic_v, b_range, as_2d=True, device=device)
                if center_x_range is not None and center_y_range is not None and source_type == 'gaussian':
                    cx_temp = center_x_range[0] + (center_x_range[1] - center_x_range[0]) * torch.rand(1).item()
                    cy_temp = center_y_range[0] + (center_y_range[1] - center_y_range[0]) * torch.rand(1).item()
                else:
                    cx_temp, cy_temp = center_x, center_y
                u0_sensors_temp = self.generate_ic_sine_series(a_coeffs_temp)
                v0_sensors_temp = self.generate_ic_sine_series(b_coeffs_temp)
                src_sensors_temp = self.generate_source(source_type, source_amplitude, cx_temp, cy_temp)
                x_colloc_temp = self.domain[0] + (self.domain[1] - self.domain[0]) * torch.rand(n_colloc, device=device)
                y_colloc_temp = self.domain[0] + (self.domain[1] - self.domain[0]) * torch.rand(n_colloc, device=device)
                t_colloc_temp = T_max * torch.rand(n_colloc, device=device)
                xyt_colloc_temp = torch.stack([x_colloc_temp, y_colloc_temp, t_colloc_temp], dim=1)
                src_colloc_temp = self.source_function(x_colloc_temp, y_colloc_temp, source_type, source_amplitude, cx_temp, cy_temp)
                x_ic_1d = torch.linspace(self.domain[0], self.domain[1], n_ic, device=device)
                y_ic_1d = torch.linspace(self.domain[0], self.domain[1], n_ic, device=device)
                X_ic_temp, Y_ic_temp = torch.meshgrid(x_ic_1d, y_ic_1d, indexing='ij')
                x_ic_temp = X_ic_temp.flatten()
                y_ic_temp = Y_ic_temp.flatten()
                t_ic_temp = torch.zeros_like(x_ic_temp)
                xyt_ic_temp = torch.stack([x_ic_temp, y_ic_temp, t_ic_temp], dim=1)
                
                # Dummy losses for context
                residual_temp = self.compute_pde_residual(u0_sensors_temp, v0_sensors_temp, src_sensors_temp, 
                                                          xyt_colloc_temp, src_colloc_temp)
                loss_pde_temp = torch.mean(residual_temp ** 2)
                u_ic_pred_temp = self.forward(u0_sensors_temp, v0_sensors_temp, src_sensors_temp, xyt_ic_temp)
                u_ic_true_temp = self.generate_ic_sine_series(a_coeffs_temp, x_ic_temp, y_ic_temp)
                loss_ic_u_temp = torch.mean((u_ic_pred_temp - u_ic_true_temp) ** 2)
                xyt_ic_grad_temp = xyt_ic_temp.clone().requires_grad_(True)
                u_ic_grad_temp = self.forward(u0_sensors_temp, v0_sensors_temp, src_sensors_temp, xyt_ic_grad_temp)
                u_t_ic_pred_temp = torch.autograd.grad(u_ic_grad_temp, xyt_ic_grad_temp,
                                                       torch.ones_like(u_ic_grad_temp),
                                                       create_graph=True)[0][:, 2]
                v_ic_true_temp = self.generate_ic_sine_series(b_coeffs_temp, x_ic_temp, y_ic_temp)
                loss_ic_v_temp = torch.mean((u_t_ic_pred_temp - v_ic_true_temp) ** 2)
                
                context = {
                    'loss_values': {
                        'PDE': loss_pde_temp.item(),
                        'IC_u': loss_ic_u_temp.item(),
                        'IC_v': loss_ic_v_temp.item(),
                    },
                    'model': self,
                    'xyt_colloc': xyt_colloc_temp,
                    'xyt_ic': xyt_ic_temp,
                    'u0_sensors': u0_sensors_temp,
                    'v0_sensors': v0_sensors_temp,
                    'src_sensors': src_sensors_temp,
                    'src_colloc': src_colloc_temp,
                    'a_coeffs': a_coeffs_temp,
                    'b_coeffs': b_coeffs_temp,
                    'output_fold': output_fold
                }
                
                if strategy == 'ntk':
                    n_pde_subsample = min(100, xyt_colloc_temp.shape[0])
                    n_ic_subsample = min(200, xyt_ic_temp.shape[0])
                    idx_pde = torch.randperm(xyt_colloc_temp.shape[0])[:n_pde_subsample]
                    idx_ic = torch.randperm(xyt_ic_temp.shape[0])[:n_ic_subsample]
                    context['xyt_colloc'] = xyt_colloc_temp[idx_pde]
                    context['src_colloc'] = src_colloc_temp[idx_pde]
                    context['xyt_ic'] = xyt_ic_temp[idx_ic]
                
                weights = adaptive_weights.update(epoch, context)
            else:
                weights = initial_weights
            
            # Train with mini-batches
            optimizer.zero_grad()
            loss_pde_avg, loss_ic_u_avg, loss_ic_v_avg = self._train_epoch_batched(
                n_batches=n_batches,
                n_colloc=n_colloc,
                n_ic=n_ic,
                T_max=T_max,
                a_range=a_range,
                b_range=b_range,
                source_type=source_type,
                source_amplitude=source_amplitude,
                center_x=center_x,
                center_y=center_y,
                center_x_range=center_x_range,
                center_y_range=center_y_range,
                n_ic_u=n_ic_u,
                n_ic_v=n_ic_v,
                weights=weights,
                optimizer=optimizer,
                max_grad_norm=max_grad_norm,
                device=device
            )
            
            loss_total_avg = weights['PDE'] * loss_pde_avg + weights['IC_u'] * loss_ic_u_avg + weights['IC_v'] * loss_ic_v_avg
            
            history['total'].append(loss_total_avg)
            history['pde'].append(loss_pde_avg)
            history['ic_u'].append(loss_ic_u_avg)
            history['ic_v'].append(loss_ic_v_avg)
            
            # Update learning rate scheduler every lr_scheduler_step epochs (only for Adam)
            if scheduler is not None and (epoch + 1) % lr_scheduler_step == 0:
                scheduler.step()
            
            # Evaluate on test set every val_interval epochs
            if (epoch + 1) % val_interval == 0:
                test_metric = 0.0
                test_pde_loss_sum = 0.0
                test_ic_u_loss_sum = 0.0
                test_ic_v_loss_sum = 0.0
                for test_case in test_cases:
                    u0_test = self.generate_ic_sine_series(test_case['a_coeffs'])
                    v0_test = self.generate_ic_sine_series(test_case['b_coeffs'])
                    src_test = self.generate_source(source_type, source_amplitude, test_case['center_x'], test_case['center_y'])
                    
                    # Sample test collocation and IC points
                    x_test_colloc = self.domain[0] + (self.domain[1] - self.domain[0]) * torch.rand(n_colloc, device=device)
                    y_test_colloc = self.domain[0] + (self.domain[1] - self.domain[0]) * torch.rand(n_colloc, device=device)
                    t_test_colloc = T_max * torch.rand(n_colloc, device=device)
                    xyt_test_colloc = torch.stack([x_test_colloc, y_test_colloc, t_test_colloc], dim=1)
                    src_test_colloc = self.source_function(x_test_colloc, y_test_colloc, source_type, source_amplitude, test_case['center_x'], test_case['center_y'])
                    
                    # Test PDE loss using same method as training
                    residual_test = self.compute_pde_residual(u0_test, v0_test, src_test, xyt_test_colloc, src_test_colloc)
                    loss_pde_test = torch.mean(residual_test ** 2)
                    
                    # Test IC losses (no gradients needed for prediction, but needed for velocity)
                    with torch.no_grad():
                        x_test_ic_1d = torch.linspace(self.domain[0], self.domain[1], n_ic, device=device)
                        y_test_ic_1d = torch.linspace(self.domain[0], self.domain[1], n_ic, device=device)
                        X_test_ic, Y_test_ic = torch.meshgrid(x_test_ic_1d, y_test_ic_1d, indexing='ij')
                        x_test_ic = X_test_ic.flatten()
                        y_test_ic = Y_test_ic.flatten()
                        t_test_ic = torch.zeros_like(x_test_ic)
                        xyt_test_ic = torch.stack([x_test_ic, y_test_ic, t_test_ic], dim=1)
                        
                        u_test_ic = self.forward(u0_test, v0_test, src_test, xyt_test_ic)
                        u_ic_true_test = self.generate_ic_sine_series(test_case['a_coeffs'], x_test_ic, y_test_ic)
                        loss_ic_u_test = torch.mean((u_test_ic - u_ic_true_test) ** 2)
                    
                    xyt_test_ic_grad = xyt_test_ic.clone().requires_grad_(True)
                    u_test_ic_grad = self.forward(u0_test, v0_test, src_test, xyt_test_ic_grad)
                    u_t_test_ic = torch.autograd.grad(u_test_ic_grad, xyt_test_ic_grad, torch.ones_like(u_test_ic_grad), create_graph=True)[0][:, 2]
                    v_ic_true_test = self.generate_ic_sine_series(test_case['b_coeffs'], x_test_ic, y_test_ic)
                    loss_ic_v_test = torch.mean((u_t_test_ic - v_ic_true_test) ** 2)
                    
                    # Accumulate test metric and individual losses
                    test_pde_loss_sum += loss_pde_test.item()
                    test_ic_u_loss_sum += loss_ic_u_test.item()
                    test_ic_v_loss_sum += loss_ic_v_test.item()
                    test_metric += np.sqrt(loss_pde_test.item()**2 + 10*loss_ic_u_test.item()**2 + loss_ic_v_test.item()**2)
                
                
                # Average test metric and losses over all test cases
                test_metric /= n_test_cases
                test_pde_loss_avg = test_pde_loss_sum / n_test_cases
                test_ic_u_loss_avg = test_ic_u_loss_sum / n_test_cases
                test_ic_v_loss_avg = test_ic_v_loss_sum / n_test_cases
                history['test_metric'].append(test_metric)
                history['test_epochs'].append(epoch + 1)  # Record current epoch
                
                # Check for improvement
                if test_metric < best_test_loss:
                    best_test_loss = test_metric
                    best_model_state = {key: value.cpu().clone() for key, value in self.state_dict().items()}
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    # Store best TEST losses at this epoch
                    best_test_losses = {
                        'pde': test_pde_loss_avg,
                        'ic_u': test_ic_u_loss_avg,
                        'ic_v': test_ic_v_loss_avg
                    }
                    log.info(f"✓ Test metric improved to {test_metric:.6e} at epoch {epoch}")
                    
                    # Save checkpoint immediately when new best is achieved (overwrites previous best)
                    checkpoint_path = os.path.join(output_fold, 'model.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'test_metric': test_metric,
                        'test_losses': best_test_losses,
                        'history': history,
                        'config': cfg,
                    }, checkpoint_path)
                else:
                    epochs_without_improvement += val_interval
                    if epochs_without_improvement % (val_interval * 5) == 0:  # Print every 5 evals
                        log.info(f"  No improvement for {epochs_without_improvement} epochs (best: {best_test_loss:.6e})")
                
                # Early stopping
                if epochs_without_improvement >= early_stopping_patience:
                    log.info(f"Early stopping triggered at epoch {epoch} (patience={early_stopping_patience})")
                    break

            if strategy == 'equal_init' and epoch == 2:
                log.info(f"Initial Weights -> PDE: {weights['PDE']:.2e}, IC_u: {weights['IC_u']:.2e}, IC_v: {weights['IC_v']:.2e}")
                
            if epoch % log_interval == 0 or epoch == n_epochs - 1:
                current_lr = optimizer.param_groups[0]['lr']
                # Diagnostic: check prediction scale (use first batch for sampling)
                a_diag = sample_coeffs(n_ic_u, a_range, p=3.0, as_2d=True, device=device)
                b_diag = sample_coeffs(n_ic_v, b_range, as_2d=True, device=device)
                if center_x_range is not None and center_y_range is not None and source_type == 'gaussian':
                    cx_diag = center_x_range[0] + (center_x_range[1] - center_x_range[0]) * torch.rand(1).item()
                    cy_diag = center_y_range[0] + (center_y_range[1] - center_y_range[0]) * torch.rand(1).item()
                else:
                    cx_diag, cy_diag = center_x, center_y
                u0_diag = self.generate_ic_sine_series(a_diag)
                v0_diag = self.generate_ic_sine_series(b_diag)
                src_diag = self.generate_source(source_type, source_amplitude, cx_diag, cy_diag)
                x_ic_1d_diag = torch.linspace(self.domain[0], self.domain[1], n_ic, device=device)
                y_ic_1d_diag = torch.linspace(self.domain[0], self.domain[1], n_ic, device=device)
                X_ic_diag, Y_ic_diag = torch.meshgrid(x_ic_1d_diag, y_ic_1d_diag, indexing='ij')
                xyt_ic_diag = torch.stack([X_ic_diag.flatten(), Y_ic_diag.flatten(), torch.zeros_like(X_ic_diag.flatten())], dim=1)
                with torch.no_grad():
                    u_sample = self.forward(u0_diag, v0_diag, src_diag, xyt_ic_diag[:100])
                    u_min, u_max = u_sample.min().item(), u_sample.max().item()
                log.info(f"Epoch {epoch:5d} | Loss: {loss_total_avg:.6f} | "
                      f"PDE: {loss_pde_avg:.6f} | IC_u: {loss_ic_u_avg:.6f} | "
                      f"IC_v: {loss_ic_v_avg:.6f} | LR: {current_lr:.2e} | "
                      f"u_range: [{u_min:.3f}, {u_max:.3f}]")
                if strategy not in ['fixed', 'equal_init']:
                    log.info(f"          Weights -> PDE: {weights['PDE']:.2e}, IC_u: {weights['IC_u']:.2e}, IC_v: {weights['IC_v']:.2e}")
        
        if strategy not in ['fixed', 'equal_init']:
            adaptive_weights.plot_weights_evolution(output_fold)
            if strategy == 'ntk':
                adaptive_weights.plot_ntk_traces(output_fold)
        
        # Restore best model
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
            log.info(f"Restored best model from epoch {best_epoch} with test metric {best_test_loss:.6f}")
            log.info(f"  Best model loss components -> PDE: {best_test_losses['pde']:.6e}, IC_u: {best_test_losses['ic_u']:.6e}, IC_v: {best_test_losses['ic_v']:.6e}")
        
        # Optional LBFGS fine-tuning phase
        if use_lbfgs_finetune:
            log.info(f"\n{'='*70}")
            log.info(f"PHASE 3: LBFGS Fine-tuning ({lbfgs_n_epochs} epochs)")
            log.info(f"{'='*70}")
            
            # Create LBFGS optimizer for fine-tuning
            optimizer_lbfgs = torch.optim.LBFGS(self.parameters(), lr=lbfgs_lr, 
                                                max_iter=lbfgs_max_iter,
                                                line_search_fn='strong_wolfe')
            
            for epoch_lbfgs in range(lbfgs_n_epochs):
                # Use current weights (fixed strategy used for LBFGS typically)
                weights_lbfgs = initial_weights if strategy == 'fixed' else adaptive_weights.get_weights()
                
                # Define closure for LBFGS
                def closure_lbfgs():
                    optimizer_lbfgs.zero_grad()
                    loss_total_lbfgs = 0.0
                    
                    # Sample single batch for LBFGS step (on GPU)
                    a_lbfgs = sample_coeffs(n_ic_u, a_range, p=3.0, as_2d=True, device=device)
                    b_lbfgs = sample_coeffs(n_ic_v, b_range, p=3.0, as_2d=True, device=device)
                    
                    if center_x_range is not None and center_y_range is not None and source_type == 'gaussian':
                        cx_lbfgs = center_x_range[0] + (center_x_range[1] - center_x_range[0]) * torch.rand(1).item()
                        cy_lbfgs = center_y_range[0] + (center_y_range[1] - center_y_range[0]) * torch.rand(1).item()
                    else:
                        cx_lbfgs, cy_lbfgs = center_x, center_y
                    
                    u0_lbfgs = self.generate_ic_sine_series(a_lbfgs)
                    v0_lbfgs = self.generate_ic_sine_series(b_lbfgs)
                    src_lbfgs = self.generate_source(source_type, source_amplitude, cx_lbfgs, cy_lbfgs)
                    
                    loss_pde_lbfgs, loss_ic_u_lbfgs, loss_ic_v_lbfgs = self._compute_batch_losses(
                        u0_lbfgs, v0_lbfgs, src_lbfgs,
                        n_colloc, n_ic, T_max, source_type, source_amplitude, cx_lbfgs, cy_lbfgs,
                        a_lbfgs, b_lbfgs
                    )
                    
                    loss_total_lbfgs = (weights_lbfgs['PDE'] * loss_pde_lbfgs + 
                                       weights_lbfgs['IC_u'] * loss_ic_u_lbfgs + 
                                       weights_lbfgs['IC_v'] * loss_ic_v_lbfgs)
                    
                    loss_total_lbfgs.backward()
                    return loss_total_lbfgs
                
                # LBFGS step
                optimizer_lbfgs.step(closure_lbfgs)
                
                # Evaluate test metric every val_interval epochs during LBFGS
                if (epoch_lbfgs + 1) % val_interval == 0:
                    test_metric_lbfgs = 0.0
                    for test_case in test_cases:
                        u0_test_lbfgs = self.generate_ic_sine_series(test_case['a_coeffs'])
                        v0_test_lbfgs = self.generate_ic_sine_series(test_case['b_coeffs'])
                        src_test_lbfgs = self.generate_source(source_type, source_amplitude, test_case['center_x'], test_case['center_y'])
                        
                        x_test_lbfgs = self.domain[0] + (self.domain[1] - self.domain[0]) * torch.rand(n_colloc)
                        y_test_lbfgs = self.domain[0] + (self.domain[1] - self.domain[0]) * torch.rand(n_colloc)
                        t_test_lbfgs = T_max * torch.rand(n_colloc)
                        xyt_test_lbfgs = torch.stack([x_test_lbfgs, y_test_lbfgs, t_test_lbfgs], dim=1)
                        src_test_lbfgs = self.source_function(x_test_lbfgs, y_test_lbfgs, source_type, source_amplitude, test_case['center_x'], test_case['center_y'])
                        
                        residual_test_lbfgs = self.compute_pde_residual(u0_test_lbfgs, v0_test_lbfgs, src_test_lbfgs, xyt_test_lbfgs, src_test_lbfgs)
                        loss_pde_test_lbfgs = torch.mean(residual_test_lbfgs ** 2)
                        
                        with torch.no_grad():
                            x_ic_lbfgs = torch.linspace(self.domain[0], self.domain[1], n_ic)
                            y_ic_lbfgs = torch.linspace(self.domain[0], self.domain[1], n_ic)
                            X_ic_lbfgs, Y_ic_lbfgs = torch.meshgrid(x_ic_lbfgs, y_ic_lbfgs, indexing='ij')
                            xyt_ic_lbfgs = torch.stack([X_ic_lbfgs.flatten(), Y_ic_lbfgs.flatten(), torch.zeros_like(X_ic_lbfgs.flatten())], dim=1)
                            
                            u_ic_lbfgs = self.forward(u0_test_lbfgs, v0_test_lbfgs, src_test_lbfgs, xyt_ic_lbfgs)
                            u_ic_true_lbfgs = self.generate_ic_sine_series(test_case['a_coeffs'], xyt_ic_lbfgs[:, 0], xyt_ic_lbfgs[:, 1])
                            loss_ic_u_test_lbfgs = torch.mean((u_ic_lbfgs - u_ic_true_lbfgs) ** 2)
                        
                        xyt_ic_grad_lbfgs = xyt_ic_lbfgs.clone().requires_grad_(True)
                        u_ic_grad_lbfgs = self.forward(u0_test_lbfgs, v0_test_lbfgs, src_test_lbfgs, xyt_ic_grad_lbfgs)
                        u_t_ic_lbfgs = torch.autograd.grad(u_ic_grad_lbfgs, xyt_ic_grad_lbfgs, torch.ones_like(u_ic_grad_lbfgs), create_graph=True)[0][:, 2]
                        v_ic_true_lbfgs = self.generate_ic_sine_series(test_case['b_coeffs'], xyt_ic_lbfgs[:, 0], xyt_ic_lbfgs[:, 1])
                        loss_ic_v_test_lbfgs = torch.mean((u_t_ic_lbfgs - v_ic_true_lbfgs) ** 2)
                        
                        test_metric_lbfgs += np.sqrt(loss_pde_test_lbfgs.item()**2 + 10*loss_ic_u_test_lbfgs.item()**2 + loss_ic_v_test_lbfgs.item()**2)
                    
                    test_metric_lbfgs /= n_test_cases
                    
                    # Track in history
                    history['test_metric'].append(test_metric_lbfgs)
                    history['test_epochs'].append(n_epochs + epoch_lbfgs + 1)  # Epoch number includes main training
                    
                    # Save checkpoint if improved
                    if test_metric_lbfgs < best_test_loss:
                        best_test_loss = test_metric_lbfgs
                        best_model_state = {key: value.cpu().clone() for key, value in self.state_dict().items()}
                        best_epoch = n_epochs + epoch_lbfgs
                        log.info(f"✓ LBFGS: Test metric improved to {test_metric_lbfgs:.6e} at epoch {n_epochs + epoch_lbfgs + 1}")
                    
                    if (epoch_lbfgs + 1) % (val_interval * 5) == 0:
                        log.info(f"  LBFGS Epoch {epoch_lbfgs+1}/{lbfgs_n_epochs} | Test Metric: {test_metric_lbfgs:.6e}")
            
            # Restore best model found during LBFGS if better
            if best_model_state is not None:
                self.load_state_dict(best_model_state)
                log.info(f"\nRestored best model from LBFGS phase (epoch {best_epoch + 1}) with test metric {best_test_loss:.6f}")
        
        log.info(f"Training complete!")
        return history, pretrain_history, {
            'best_epoch': best_epoch, 
            'best_test_metric': best_test_loss, 
            'best_losses': best_test_losses,
        }


class FourierFeatureTransform(nn.Module):
    """
    Fourier Feature Mapping for PINNs to overcome spectral bias.
    
    Transforms input [x, y, t] using:
    \lambda(X) = [cos(B_x*x), sin(B_x*x), cos(B_y*y), sin(B_y*y), cos(B_t*t), sin(B_t*t), ...]
    
    where B matrices are sampled from N(0, σ²)
    
    Args:
        input_dim: Dimension of input (3 for [x, y, t] in 2D)
        m_spatial_x: Number of Fourier features for x dimension
        m_spatial_y: Number of Fourier features for y dimension
        m_temporal: Number of Fourier features for temporal dimension
        sigma_spatial_x: Standard deviation for x spatial features
        sigma_spatial_y: Standard deviation for y spatial features
        sigma_temporal_list: List of standard deviations for temporal features
        seed: Random seed for reproducibility (optional)
    """
    
    def __init__(self, 
                 input_dim=3,
                 m_spatial_x=64,
                 m_spatial_y=64,
                 m_temporal=64,
                 sigma_spatial_x=1.0,
                 sigma_spatial_y=1.0,
                 sigma_temporal_list=[1.0, 10.0],
                 seed=None):
        super().__init__()
        
        self.input_dim = input_dim
        self.m_spatial_x = m_spatial_x
        self.m_spatial_y = m_spatial_y
        self.m_temporal = m_temporal
        self.sigma_spatial_x = sigma_spatial_x
        self.sigma_spatial_y = sigma_spatial_y
        self.sigma_temporal_list = sigma_temporal_list
        
        # Set seed for reproducibility if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Create B matrices for x and y dimensions
        self.register_buffer(
            'B_spatial_x',
            torch.randn(m_spatial_x, 1) * sigma_spatial_x
        )
        self.register_buffer(
            'B_spatial_y',
            torch.randn(m_spatial_y, 1) * sigma_spatial_y
        )
        
        # Temporal features: list of B matrices with different sigmas
        self.B_temporal_list = []
        for idx, sigma_t in enumerate(sigma_temporal_list):
            B_t = torch.randn(m_temporal, 1) * sigma_t
            buffer_name = f'B_temporal_sigma_{idx}'
            self.register_buffer(buffer_name, B_t)
            self.B_temporal_list.append(B_t)
        
        # Calculate output dimension
        # For x: 2*m_spatial_x (cos + sin)
        # For y: 2*m_spatial_y (cos + sin)
        # For t: 2*m_temporal (cos + sin) per sigma
        self.output_dim = 2 * m_spatial_x + 2 * m_spatial_y + len(sigma_temporal_list) * 2 * m_temporal
    
    def forward(self, X):
        """
        Apply Fourier feature mapping to input.
        
        Args:
            X: Input tensor of shape (batch_size, 3) where X[:, 0] is x, X[:, 1] is y, X[:, 2] is t
            
        Returns:
            Transformed features of shape (batch_size, output_dim)
        """
        # Split into spatial (x, y) and temporal (t) components
        x = X[:, 0:1]  # Shape: (batch_size, 1)
        y = X[:, 1:2]  # Shape: (batch_size, 1)
        t = X[:, 2:3]  # Shape: (batch_size, 1)
        
        features = []
        
        # Apply x spatial Fourier features
        x_proj = torch.matmul(x, self.B_spatial_x.T)  # (batch_size, m_spatial_x)
        x_features = torch.cat([torch.cos(2 * np.pi * x_proj), 
                                torch.sin(2 * np.pi * x_proj)], dim=1)
        features.append(x_features)
        
        # Apply y spatial Fourier features
        y_proj = torch.matmul(y, self.B_spatial_y.T)  # (batch_size, m_spatial_y)
        y_features = torch.cat([torch.cos(2 * np.pi * y_proj), 
                                torch.sin(2 * np.pi * y_proj)], dim=1)
        features.append(y_features)
        
        # Apply temporal Fourier features for each sigma
        for i, sigma_t in enumerate(self.sigma_temporal_list):
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