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
            
            # Apply BC factor, squeezing to match prediction shape
            bc_factor_sq = bc_factor_sq.squeeze(-1)  # Remove last dimension if needed
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
    
    def _train_phase1_pretraining(self, cfg, test_cases, output_fold, gt_data, device):
        """
        Phase 1: IC Pre-training with frozen dynamic components.
        Returns pretrain_history dictionary.
        """
        n_epochs_pretrain = cfg.training.n_epochs_pretrain
        n_batches = cfg.training.n_batches
        n_ic = cfg.data.n_ic
        n_ic_u = cfg.data.n_ic_u
        n_ic_v = cfg.data.n_ic_v
        val_interval = cfg.training.val_interval
        lr = cfg.training.lr
        
        a_range = tuple(cfg.data.a_range)
        b_range = tuple(cfg.data.b_range)
        
        log.info(f"=== PHASE 1: Pre-training IC Reconstruction for {n_epochs_pretrain} epochs ===")
        pretrain_history = {'loss_ic_u': [], 'loss_ic_v': [], 'test_metric': [], 'test_epochs': []}
        
        # Freeze all parameters initially
        for param in self.parameters():
            param.requires_grad = False
        
        # Unfreeze only branch_ic and trunk
        trainable_params = []
        if hasattr(self, 'branch_ic'):
            for param in self.branch_ic.parameters():
                param.requires_grad = True
                trainable_params.append(param)
        if hasattr(self, 'trunk'):
            for param in self.trunk.parameters():
                param.requires_grad = True
                trainable_params.append(param)
        if hasattr(self, 'branch_src'):
                        for param in self.branch_src.parameters():
                            param.requires_grad = True
                            trainable_params.append(param)        
        optimizer_pre = torch.optim.Adam(trainable_params, lr=lr)
        best_pretrain_loss = float('inf')
        best_pretrain_model_state = None
        best_pretrain_epoch = 0
        
        for epoch_pre in range(n_epochs_pretrain):
            loss_ic_u_accum = 0.0
            loss_ic_v_accum = 0.0
            
            for _ in range(n_batches):
                a_coeffs_pre = sample_coeffs(n_ic_u, a_range, p=3.0, as_2d=True, device=device)
                b_coeffs_pre = sample_coeffs(n_ic_v, b_range, as_2d=True, device=device)
                
                u0_sensors_pre = self.generate_ic_sine_series(a_coeffs_pre)
                v0_sensors_pre = self.generate_ic_sine_series(b_coeffs_pre)
                
                x_ic_1d = torch.linspace(self.domain[0], self.domain[1], n_ic, device=device)
                y_ic_1d = torch.linspace(self.domain[0], self.domain[1], n_ic, device=device)
                X_ic, Y_ic = torch.meshgrid(x_ic_1d, y_ic_1d, indexing='ij')
                xyt_ic = torch.stack([X_ic.flatten(), Y_ic.flatten(), torch.zeros_like(X_ic.flatten())], dim=1)
                
                u0_true = self.generate_ic_sine_series(a_coeffs_pre, X_ic.flatten(), Y_ic.flatten())
                v0_true = self.generate_ic_sine_series(b_coeffs_pre, X_ic.flatten(), Y_ic.flatten())
                
                u0_pred, v0_pred = self.predict_ic(u0_sensors_pre, v0_sensors_pre, xyt_ic)
                
                loss_ic_u = torch.mean((u0_pred - u0_true)**2)
                loss_ic_v = torch.mean((v0_pred - v0_true)**2)
                loss_pre = loss_ic_u + loss_ic_v
                
                loss_pre.backward()
                loss_ic_u_accum += loss_ic_u.item()
                loss_ic_v_accum += loss_ic_v.item()
            
            optimizer_pre.step()
            optimizer_pre.zero_grad()
            
            loss_ic_u_avg = loss_ic_u_accum / n_batches
            loss_ic_v_avg = loss_ic_v_accum / n_batches
            pretrain_history['loss_ic_u'].append(loss_ic_u_avg)
            pretrain_history['loss_ic_v'].append(loss_ic_v_avg)
            
            # Evaluate on test set every val_interval epochs
            if (epoch_pre + 1) % val_interval == 0:
                pretrain_test_metric = 0.0
                for test_case in test_cases:
                    u0_test_pre = self.generate_ic_sine_series(test_case['a_coeffs'])
                    v0_test_pre = self.generate_ic_sine_series(test_case['b_coeffs'])
                    
                    with torch.no_grad():
                        x_test_ic_1d = torch.linspace(self.domain[0], self.domain[1], n_ic, device=device)
                        y_test_ic_1d = torch.linspace(self.domain[0], self.domain[1], n_ic, device=device)
                        X_test_ic, Y_test_ic = torch.meshgrid(x_test_ic_1d, y_test_ic_1d, indexing='ij')
                        xyt_test_ic = torch.stack([X_test_ic.flatten(), Y_test_ic.flatten(), torch.zeros_like(X_test_ic.flatten())], dim=1)
                        
                        u0_test_pred, v0_test_pred = self.predict_ic(u0_test_pre, v0_test_pre, xyt_test_ic)
                        u0_test_true = self.generate_ic_sine_series(test_case['a_coeffs'], X_test_ic.flatten(), Y_test_ic.flatten())
                        v0_test_true = self.generate_ic_sine_series(test_case['b_coeffs'], X_test_ic.flatten(), Y_test_ic.flatten())
                        
                        loss_test_ic_u = torch.mean((u0_test_pred - u0_test_true)**2)
                        loss_test_ic_v = torch.mean((v0_test_pred - v0_test_true)**2)
                        pretrain_test_metric += (loss_test_ic_u + loss_test_ic_v).item()
                
                pretrain_test_metric /= len(test_cases)
                pretrain_history['test_metric'].append(pretrain_test_metric)
                pretrain_history['test_epochs'].append(epoch_pre + 1)
                
                if pretrain_test_metric < best_pretrain_loss:
                    best_pretrain_loss = pretrain_test_metric
                    best_pretrain_model_state = {key: value.cpu().clone() for key, value in self.state_dict().items()}
                    best_pretrain_epoch = epoch_pre
                    log.info(f"  ✓ Pretrain test metric improved to {pretrain_test_metric:.6e} at epoch {epoch_pre+1}")
                    
                    checkpoint_path = os.path.join(output_fold, 'model_pretrain.pth')
                    torch.save({
                        'epoch': epoch_pre,
                        'model_state_dict': self.state_dict(),
                        'test_metric': pretrain_test_metric,
                        'pretrain_history': pretrain_history,
                        'config': cfg,
                    }, checkpoint_path)
                
                if (epoch_pre + 1) % (val_interval * 5) == 0:
                    log.info(f"  [Pre-train] Epoch {epoch_pre+1}/{n_epochs_pretrain} | IC_u: {loss_ic_u_avg:.6e} | IC_v: {loss_ic_v_avg:.6e} | Test: {pretrain_test_metric:.6e}")
            elif (epoch_pre + 1) % 100 == 0:
                log.info(f"  [Pre-train] Epoch {epoch_pre+1}/{n_epochs_pretrain} | IC_u: {loss_ic_u_avg:.6e} | IC_v: {loss_ic_v_avg:.6e}")
        
        log.info("=== PHASE 1 COMPLETE ===")
        
        if best_pretrain_model_state is not None:
            self.load_state_dict(best_pretrain_model_state)
            log.info(f"Restored best pretraining model from epoch {best_pretrain_epoch+1}")
        
        plot_ic_reconstruction(self, test_cases[0], save_path=os.path.join(output_fold, 'ic_pretrain_diagnostic.png'), gt_data=gt_data)
        
        # Unfreeze everything for Phase 2
        for param in self.parameters():
            param.requires_grad = True
        
        return pretrain_history, best_pretrain_model_state, best_pretrain_loss, {'ic_u': float('inf'), 'ic_v': float('inf'), 'pde': float('inf')}
    
    def _train_phase2_main(self, cfg, test_cases, output_fold, device, adaptive_weights, initial_weights):
        """
        Phase 2: Main training with adaptive/fixed loss weights.
        Returns history dict and best model info.
        """
        n_epochs = cfg.training.n_epochs
        if n_epochs == 0:
            log.info("Phase 2 skipped (n_epochs=0)")
            return {'total': [], 'pde': [], 'ic_u': [], 'ic_v': [], 'test_metric': [], 'test_epochs': []}, None, float('inf'), {}
        
        log.info(f"=== PHASE 2: Main Training for {n_epochs} epochs ===")
        
        optimizer = torch.optim.Adam(self.parameters(), lr=cfg.training.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.training.lr_scheduler_gamma)
        
        history = {'total': [], 'pde': [], 'ic_u': [], 'ic_v': [], 'test_metric': [], 'test_epochs': []}
        best_test_loss = float('inf')
        best_model_state = None
        best_epoch = 0
        best_test_losses = {'pde': float('inf'), 'ic_u': float('inf'), 'ic_v': float('inf')}
        epochs_without_improvement = 0
        
        if cfg.model.strategy == 'fixed':
            log.info(f"Initial Weights -> PDE: {initial_weights['PDE']:.2e}, IC_u: {initial_weights['IC_u']:.2e}, IC_v: {initial_weights['IC_v']:.2e}")
        
        for epoch in range(n_epochs):
            # Get weights for this epoch
            if cfg.model.strategy != 'fixed':
                weights = self._compute_adaptive_weights(epoch, cfg, adaptive_weights)
            else:
                weights = initial_weights
            
            # Train one epoch
            loss_pde_avg, loss_ic_u_avg, loss_ic_v_avg = self._train_epoch_batched(
                n_batches=cfg.training.n_batches,
                n_colloc=cfg.data.n_colloc,
                n_ic=cfg.data.n_ic,
                T_max=cfg.data.T_max,
                a_range=tuple(cfg.data.a_range),
                b_range=tuple(cfg.data.b_range),
                source_type=cfg.data.source_type,
                source_amplitude=cfg.data.source_amplitude,
                center_x=cfg.data.center_x,
                center_y=cfg.data.center_y,
                center_x_range=tuple(cfg.data.center_x_range) if cfg.data.center_x_range else None,
                center_y_range=tuple(cfg.data.center_y_range) if cfg.data.center_y_range else None,
                n_ic_u=cfg.data.n_ic_u,
                n_ic_v=cfg.data.n_ic_v,
                weights=weights,
                optimizer=optimizer,
                max_grad_norm=cfg.training.max_grad_norm,
                device=device
            )
            
            loss_total_avg = weights['PDE'] * loss_pde_avg + weights['IC_u'] * loss_ic_u_avg + weights['IC_v'] * loss_ic_v_avg
            history['total'].append(loss_total_avg)
            history['pde'].append(loss_pde_avg)
            history['ic_u'].append(loss_ic_u_avg)
            history['ic_v'].append(loss_ic_v_avg)
            
            if (epoch + 1) % cfg.training.lr_scheduler_step == 0:
                scheduler.step()
            
            # Evaluate on test set
            if (epoch + 1) % cfg.training.val_interval == 0:
                test_metric, test_losses = self._evaluate_test_set(test_cases, cfg, device)
                history['test_metric'].append(test_metric)
                history['test_epochs'].append(epoch + 1)
                
                if test_metric < best_test_loss:
                    best_test_loss = test_metric
                    best_model_state = {key: value.cpu().clone() for key, value in self.state_dict().items()}
                    best_epoch = epoch
                    best_test_losses = test_losses
                    epochs_without_improvement = 0
                    log.info(f"✓ Test metric improved to {test_metric:.6e} at epoch {epoch}")
                    
                    checkpoint_path = os.path.join(output_fold, 'model.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'test_metric': test_metric,
                        'test_losses': best_test_losses,
                        'history': history,
                    }, checkpoint_path)
                else:
                    epochs_without_improvement += cfg.training.val_interval
                    if epochs_without_improvement >= cfg.training.es_patience:
                        log.info(f"Early stopping at epoch {epoch}")
                        break
            
            if epoch % cfg.training.log_interval == 0 or epoch == n_epochs - 1:
                log.info(f"Epoch {epoch:5d} | Loss: {loss_total_avg:.6f} | PDE: {loss_pde_avg:.6f} | IC_u: {loss_ic_u_avg:.6f} | IC_v: {loss_ic_v_avg:.6f}")
        
        if cfg.model.strategy not in ['fixed', 'equal_init']:
            adaptive_weights.plot_weights_evolution(output_fold)
            if cfg.model.strategy == 'ntk':
                adaptive_weights.plot_ntk_traces(output_fold)
        
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
            log.info(f"Restored best model from epoch {best_epoch} with test metric {best_test_loss:.6f}")
        
        return history, best_model_state, best_test_loss, best_test_losses
    
    def _train_phase3_lbfgs(self, cfg, test_cases, output_fold, device, initial_weights, adaptive_weights, n_epochs_main):
        """
        Phase 3: LBFGS fine-tuning (optional).
        Returns updated history and best model info.
        """
        if not cfg.training.get('use_lbfgs_finetune', False):
            return None, None, float('inf')
        
        log.info(f"\n{'='*70}")
        log.info(f"PHASE 3: LBFGS Fine-tuning")
        log.info(f"{'='*70}")
        
        optimizer_lbfgs = torch.optim.LBFGS(self.parameters(), lr=cfg.training.get('lbfgs_lr', cfg.training.lr),
                                            max_iter=cfg.training.get('lbfgs_max_iter', 20),
                                            line_search_fn='strong_wolfe')
        
        history_lbfgs = {'test_metric': [], 'test_epochs': []}
        best_test_loss_lbfgs = float('inf')
        best_model_state_lbfgs = None
        best_epoch_lbfgs = 0
        
        for epoch_lbfgs in range(cfg.training.get('lbfgs_n_epochs', 100)):
            weights_lbfgs = initial_weights if cfg.model.strategy == 'fixed' else adaptive_weights.get_weights()
            
            def closure_lbfgs():
                optimizer_lbfgs.zero_grad()
                a_lbfgs = sample_coeffs(cfg.data.n_ic_u, tuple(cfg.data.a_range), p=3.0, as_2d=True, device=device)
                b_lbfgs = sample_coeffs(cfg.data.n_ic_v, tuple(cfg.data.b_range), p=3.0, as_2d=True, device=device)
                
                u0_lbfgs = self.generate_ic_sine_series(a_lbfgs)
                v0_lbfgs = self.generate_ic_sine_series(b_lbfgs)
                src_lbfgs = self.generate_source(cfg.data.source_type, cfg.data.source_amplitude, cfg.data.center_x, cfg.data.center_y)
                
                loss_pde, loss_ic_u, loss_ic_v = self._compute_batch_losses(
                    u0_lbfgs, v0_lbfgs, src_lbfgs,
                    cfg.data.n_colloc, cfg.data.n_ic, cfg.data.T_max,
                    cfg.data.source_type, cfg.data.source_amplitude,
                    cfg.data.center_x, cfg.data.center_y,
                    a_lbfgs, b_lbfgs
                )
                
                loss_total = weights_lbfgs['PDE'] * loss_pde + weights_lbfgs['IC_u'] * loss_ic_u + weights_lbfgs['IC_v'] * loss_ic_v
                loss_total.backward()
                return loss_total
            
            optimizer_lbfgs.step(closure_lbfgs)
            
            if (epoch_lbfgs + 1) % cfg.training.val_interval == 0:
                test_metric_lbfgs, _ = self._evaluate_test_set(test_cases, cfg, device)
                history_lbfgs['test_metric'].append(test_metric_lbfgs)
                history_lbfgs['test_epochs'].append(n_epochs_main + epoch_lbfgs + 1)
                
                if test_metric_lbfgs < best_test_loss_lbfgs:
                    best_test_loss_lbfgs = test_metric_lbfgs
                    best_model_state_lbfgs = {key: value.cpu().clone() for key, value in self.state_dict().items()}
                    best_epoch_lbfgs = n_epochs_main + epoch_lbfgs
                    log.info(f"✓ LBFGS: Test metric improved to {test_metric_lbfgs:.6e}")
                
                if (epoch_lbfgs + 1) % (cfg.training.val_interval * 5) == 0:
                    log.info(f"  LBFGS Epoch {epoch_lbfgs+1} | Test Metric: {test_metric_lbfgs:.6e}")
        
        if best_model_state_lbfgs is not None:
            self.load_state_dict(best_model_state_lbfgs)
            log.info(f"Restored best LBFGS model from epoch {best_epoch_lbfgs + 1}")
        
        return history_lbfgs, best_model_state_lbfgs, best_test_loss_lbfgs
    
    def _compute_adaptive_weights(self, epoch, cfg, adaptive_weights):
        """Compute adaptive weights for the current epoch."""
        if cfg.model.strategy == 'fixed':
            return {'PDE': cfg.model.w_pde, 'IC_u': cfg.model.w_ic_u, 'IC_v': cfg.model.w_ic_v}
        
        # For adaptive strategies, compute context-based weights
        device = next(self.parameters()).device
        a_temp = sample_coeffs(cfg.data.n_ic_u, tuple(cfg.data.a_range), p=3.0, as_2d=True, device=device)
        b_temp = sample_coeffs(cfg.data.n_ic_v, tuple(cfg.data.b_range), p=3.0, as_2d=True, device=device)
        
        u0_temp = self.generate_ic_sine_series(a_temp)
        v0_temp = self.generate_ic_sine_series(b_temp)
        src_temp = self.generate_source(cfg.data.source_type, cfg.data.source_amplitude, cfg.data.center_x, cfg.data.center_y)
        
        loss_pde, loss_ic_u, loss_ic_v = self._compute_batch_losses(
            u0_temp, v0_temp, src_temp,
            cfg.data.n_colloc, cfg.data.n_ic, cfg.data.T_max,
            cfg.data.source_type, cfg.data.source_amplitude,
            cfg.data.center_x, cfg.data.center_y,
            a_temp, b_temp
        )
        
        context = {'loss_values': {'PDE': loss_pde.item(), 'IC_u': loss_ic_u.item(), 'IC_v': loss_ic_v.item()}, 'model': self}
        return adaptive_weights.update(epoch, context)
    
    def _evaluate_test_set(self, test_cases, cfg, device):
        """Evaluate model on test set. Returns test_metric and individual losses."""
        test_metric = 0.0
        test_pde_loss_sum = 0.0
        test_ic_u_loss_sum = 0.0
        test_ic_v_loss_sum = 0.0
        
        for test_case in test_cases:
            u0_test = self.generate_ic_sine_series(test_case['a_coeffs'])
            v0_test = self.generate_ic_sine_series(test_case['b_coeffs'])
            src_test = self.generate_source(cfg.data.source_type, cfg.data.source_amplitude, test_case['center_x'], test_case['center_y'])
            
            x_colloc = self.domain[0] + (self.domain[1] - self.domain[0]) * torch.rand(cfg.data.n_colloc, device=device)
            y_colloc = self.domain[0] + (self.domain[1] - self.domain[0]) * torch.rand(cfg.data.n_colloc, device=device)
            t_colloc = cfg.data.T_max * torch.rand(cfg.data.n_colloc, device=device)
            xyt_colloc = torch.stack([x_colloc, y_colloc, t_colloc], dim=1)
            src_colloc = self.source_function(x_colloc, y_colloc, cfg.data.source_type, cfg.data.source_amplitude, test_case['center_x'], test_case['center_y'])
            
            residual_test = self.compute_pde_residual(u0_test, v0_test, src_test, xyt_colloc, src_colloc)
            loss_pde_test = torch.mean(residual_test ** 2)
            
            with torch.no_grad():
                x_ic = torch.linspace(self.domain[0], self.domain[1], cfg.data.n_ic, device=device)
                y_ic = torch.linspace(self.domain[0], self.domain[1], cfg.data.n_ic, device=device)
                X_ic, Y_ic = torch.meshgrid(x_ic, y_ic, indexing='ij')
                xyt_ic = torch.stack([X_ic.flatten(), Y_ic.flatten(), torch.zeros_like(X_ic.flatten())], dim=1)
                
                u_ic = self.forward(u0_test, v0_test, src_test, xyt_ic)
                u_ic_true = self.generate_ic_sine_series(test_case['a_coeffs'], X_ic.flatten(), Y_ic.flatten())
                loss_ic_u_test = torch.mean((u_ic - u_ic_true) ** 2)
            
            xyt_ic_grad = xyt_ic.clone().requires_grad_(True)
            u_ic_grad = self.forward(u0_test, v0_test, src_test, xyt_ic_grad)
            u_t_ic = torch.autograd.grad(u_ic_grad, xyt_ic_grad, torch.ones_like(u_ic_grad), create_graph=True)[0][:, 2]
            v_ic_true = self.generate_ic_sine_series(test_case['b_coeffs'], X_ic.flatten(), Y_ic.flatten())
            loss_ic_v_test = torch.mean((u_t_ic - v_ic_true) ** 2)
            
            test_pde_loss_sum += loss_pde_test.item()
            test_ic_u_loss_sum += loss_ic_u_test.item()
            test_ic_v_loss_sum += loss_ic_v_test.item()
            test_metric += np.sqrt(loss_pde_test.item()**2 + 10*loss_ic_u_test.item()**2 + loss_ic_v_test.item()**2)
        
        test_metric /= len(test_cases)
        test_losses = {
            'pde': test_pde_loss_sum / len(test_cases),
            'ic_u': test_ic_u_loss_sum / len(test_cases),
            'ic_v': test_ic_v_loss_sum / len(test_cases)
        }
        return test_metric, test_losses

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
        Train PINN-DeepONet in 3 phases: IC Pre-training, Main Training, LBFGS Fine-tuning.
        
        Phase 1 (Optional): Pre-train IC reconstruction with frozen dynamic components
        Phase 2 (Main): Train full model with adaptive or fixed loss weights
        Phase 3 (Optional): Fine-tune with LBFGS optimizer
        """
        # Set device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self = self.to(device)
        
        # Log configuration
        log.info(f"Epochs (Phase 2): {cfg.training.n_epochs}")
        log.info(f"Collocation points: {cfg.data.n_colloc}")
        log.info(f"a range: {tuple(cfg.data.a_range)}, b range: {tuple(cfg.data.b_range)}")
        log.info(f"Wave speed: {self.c}, Damping: {self.k}")
        
        # Generate test set
        test_cases = self._generate_test_set(cfg, device)
        
        # ==== PHASE 1: IC Pre-training ====
        pretrain_history = None
        best_pretrain_model_state = None
        best_pretrain_test_loss = None
        best_pretrain_test_losses = None
        if cfg.training.pretrain_ic:
            pretrain_history, best_pretrain_model_state, best_pretrain_test_loss, best_pretrain_test_losses = self._train_phase1_pretraining(cfg, test_cases, output_fold, gt_data, device)
        
        # ==== PHASE 2: Main Training ====
        if cfg.training.train_adam:
            initial_weights = {'PDE': cfg.model.w_pde, 'IC_u': cfg.model.w_ic_u, 'IC_v': cfg.model.w_ic_v}
            adaptive_weights = AdaptiveLossWeights(initial_weights=initial_weights, strategy=cfg.model.strategy)
            
            history, best_model_state, best_test_loss, best_test_losses = self._train_phase2_main(
                cfg, test_cases, output_fold, device, adaptive_weights, initial_weights
            )
        else:
            history, best_test_loss, best_test_losses = pretrain_history, best_pretrain_test_loss, best_pretrain_test_losses
        
        # ==== PHASE 3: LBFGS Fine-tuning ====
        if cfg.training.train_lbfgs:
            history_lbfgs, _, best_test_loss_lbfgs = self._train_phase3_lbfgs(
                cfg, test_cases, output_fold, device, initial_weights, adaptive_weights, cfg.training.n_epochs
            )
        else:
            history_lbfgs=None
        
        # Merge histories if LBFGS was run
        if history_lbfgs is not None:
            history['test_metric'].extend(history_lbfgs['test_metric'])
            history['test_epochs'].extend(history_lbfgs['test_epochs'])
            if best_test_loss_lbfgs < best_test_loss:
                best_test_loss = best_test_loss_lbfgs
        
        log.info(f"Training complete!")
        return history, pretrain_history, {
            'best_epoch': history['test_epochs'][-1] if history['test_epochs'] else 0,
            'best_test_metric': best_test_loss,
            'best_losses': best_test_losses,
        }
    
    def _generate_test_set(self, cfg, device):
        """Generate deterministic test set."""
        test_rng = np.random.RandomState(cfg.training.test_seed)
        test_cases = []
        a_range = tuple(cfg.data.a_range)
        b_range = tuple(cfg.data.b_range)
        
        for _ in range(cfg.training.n_test_cases):
            a_test = sample_coeffs(cfg.data.n_ic_u, a_range, p=3.0, as_2d=True, device=device)
            b_test = sample_coeffs(cfg.data.n_ic_v, b_range, p=3.0, as_2d=True, device=device)
            
            center_x_range = tuple(cfg.data.center_x_range) if cfg.data.center_x_range else None
            center_y_range = tuple(cfg.data.center_y_range) if cfg.data.center_y_range else None
            
            if center_x_range is not None and center_y_range is not None and cfg.data.source_type == 'gaussian':
                cx_test = test_rng.uniform(center_x_range[0], center_x_range[1])
                cy_test = test_rng.uniform(center_y_range[0], center_y_range[1])
            else:
                cx_test, cy_test = cfg.data.center_x, cfg.data.center_y
            
            test_cases.append({
                'a_coeffs': a_test,
                'b_coeffs': b_test,
                'center_x': cx_test,
                'center_y': cy_test
            })
        
        return test_cases


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