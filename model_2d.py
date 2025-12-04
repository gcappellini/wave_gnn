import numpy as np
import torch
import torch.nn as nn
import warnings
from adaptive_weights import AdaptiveLossWeights
import os
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def sample_coeffs(n, base_range, p=1.5, as_2d=False):
    """
    Sample coefficients with decreasing range per mode.
    
    Args:
        n: Number of modes (for 1D) or modes per dimension (for 2D)
        base_range: (min, max) range for first mode
        p: Power law decay exponent
        as_2d: If True, returns (n, n) 2D tensor with different x,y modes
    
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
    
    coeffs_1d = torch.tensor(coeffs)
    
    if as_2d:
        # Create 2D coefficient matrix: different values for each (k, l) mode
        coeffs_2d = torch.zeros(n, n)
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
    
    def __init__(self, n_sensors_ic=20, n_sensors_src=20, 
                 branch_width=100, trunk_width=100, branch_depth=4, trunk_depth=4, p=100, wave_speed=1.0, damping_coeff=1.0, branch_activation=nn.LeakyReLU(), trunk_activation=nn.Tanh(), use_fft_trunk=False, fft_trunk_args=None):
        super().__init__()
        
        self.n_sensors_ic = n_sensors_ic
        self.n_sensors_src = n_sensors_src
        self.p = p
        self.c = wave_speed
        self.k = damping_coeff
        self.domain = [0.0, 1.0]  # Square domain [0,1] × [0,1]

        fft_trunk = FourierFeatureTransform(**(fft_trunk_args or {})) if use_fft_trunk else None
        
        # IC branch: takes BOTH u0 and v0 sensors (2D grids flattened)
        self.branch_ic = BranchNet(2 * n_sensors_ic * n_sensors_ic, branch_width, 2 * p, n_hidden_layers=branch_depth, activation=branch_activation)
        
        # Source branch: 2D grid flattened
        self.branch_source = BranchNet(n_sensors_src * n_sensors_src, branch_width, p, n_hidden_layers=branch_depth, activation=branch_activation)
        
        # Trunk network for (x, y, t)
        self.trunk = TrunkNet(trunk_width, p, n_hidden_layers=trunk_depth, fft_transform=fft_trunk, activation=trunk_activation)
        
        # Bias term
        self.bias = nn.Parameter(torch.zeros(1))
        
        # Fixed sensor locations (2D grid)
        x_1d = torch.linspace(self.domain[0], self.domain[1], n_sensors_ic)
        y_1d = torch.linspace(self.domain[0], self.domain[1], n_sensors_ic)
        X_grid, Y_grid = torch.meshgrid(x_1d, y_1d, indexing='ij')
        self.register_buffer('sensor_x_ic', X_grid.flatten())
        self.register_buffer('sensor_y_ic', Y_grid.flatten())
        
        x_1d_src = torch.linspace(self.domain[0], self.domain[1], n_sensors_src)
        y_1d_src = torch.linspace(self.domain[0], self.domain[1], n_sensors_src)
        X_grid_src, Y_grid_src = torch.meshgrid(x_1d_src, y_1d_src, indexing='ij')
        self.register_buffer('sensor_x_src', X_grid_src.flatten())
        self.register_buffer('sensor_y_src', Y_grid_src.flatten())
    
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
        
        # Concatenate ICs and encode
        ic_concat = torch.cat([u0_sensors, v0_sensors], dim=1)
        ic_encoded = self.branch_ic(ic_concat)
        
        # Split into displacement and velocity contributions
        b_u = ic_encoded[:, :self.p]
        b_v = ic_encoded[:, self.p:]
        
        # Encode source
        b_src = self.branch_source(src_sensors)
        
        # Combine contributions
        b_combined = b_u + b_v + b_src
        
        # Encode spatiotemporal locations
        tau = self.trunk(xyt)
        
        # DeepONet operation
        u_net = torch.matmul(b_combined, tau.T) + self.bias
        
        # Extract x, y coordinates
        x = xyt[:, 0]
        y = xyt[:, 1]
        
        # Hard BC enforcement: u = x*(1-x) * y*(1-y) * u_net
        # bc_factor = x * (1.0 - x) * y * (1.0 - y)
        bc_factor = torch.sin(np.pi * x) * torch.sin(np.pi * y)  # max = 1.0
        
        if single_sample:
            u = bc_factor * u_net
        else:
            u = bc_factor.unsqueeze(0) * u_net
        
        return u.squeeze(0) if single_sample else u
    
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
    
    def train_pinn(self, n_epochs=5000, n_colloc=500, n_ic=200, lr=1e-3, 
                   a_range=(-1.5, 1.5), b_range=(0.0, 0.0), 
                   source_type='zero', source_amplitude=15.0, w_pde=1.0, w_ic_u=10.0, w_ic_v=10.0, strategy='fixed',
                   center_x=0.5, center_y=0.5, center_y_range=(0.3, 0.7), center_x_range=(0.3, 0.7), n_ic_u=2, n_ic_v=2, T_max=1.0, output_fold=os.path.join(SCRIPT_DIR, 'logs_multibranch_wave'), eval_freq=1, n_test_cases=20, early_stopping_patience=100, test_seed=42):
        """
        Train PINN-DeepONet for 2D wave equation with early stopping based on test loss
        
        Args:
            eval_freq: Evaluate test loss every N epochs
            n_test_cases: Number of random test cases (fixed seed)
            early_stopping_patience: Stop if no improvement for N epochs
            test_seed: Random seed for test set generation (deterministic)
        """
        print(f"Epochs: {n_epochs}")
        print(f"Collocation points: {n_colloc}")
        print(f"a range: {a_range}, b range: {b_range}")
        if center_x_range is not None and center_y_range is not None:
            print(f"Source: {source_type}, amplitude: {source_amplitude}, center x range: {center_x_range}, center y range: {center_y_range}")
        else:
            print(f"Source: {source_type}, amplitude: {source_amplitude}, center: ({center_x}, {center_y})")
        print(f"Wave speed: {self.c}, Damping: {self.k}")
        print(f"Early stopping: patience={early_stopping_patience}, eval_freq={eval_freq}, n_test_cases={n_test_cases}\n")
        
        # Generate deterministic test set with fixed seed
        test_rng = np.random.RandomState(test_seed)
        test_cases = []
        for _ in range(n_test_cases):
            a_test = sample_coeffs(n_ic_u, a_range, p=3.0, as_2d=True)
            b_test = sample_coeffs(n_ic_v, b_range, p=3.0, as_2d=True)
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
        
        initial_weights = {'PDE': w_pde, 'IC_u': w_ic_u, 'IC_v': w_ic_v}
        adaptive_weights = AdaptiveLossWeights(
            initial_weights=initial_weights,
            strategy=strategy,  
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=500, verbose=True
        )
        
        history = {'total': [], 'pde': [], 'ic_u': [], 'ic_v': [], 'test_metric': []}
        
        # Track best model based on TEST loss
        best_test_loss = float('inf')
        best_model_state = None
        best_epoch = 0
        best_test_losses = {'pde': float('inf'), 'ic_u': float('inf'), 'ic_v': float('inf')}
        epochs_without_improvement = 0
        
        # Print initial weights only for fixed strategy
        if strategy == 'fixed':
            print(f"Initial Weights -> PDE: {initial_weights['PDE']:.2e}, IC_u: {initial_weights['IC_u']:.2e}, IC_v: {initial_weights['IC_v']:.2e}\n")
        
        for epoch in range(n_epochs):
            # Sample random IC amplitudes with independent x and y modes
            a_coeffs = sample_coeffs(n_ic_u, a_range, p=3.0, as_2d=True)
            b_coeffs = sample_coeffs(n_ic_v, b_range, as_2d=True)
            
            # Sample random source center
            if center_x_range is not None and center_y_range is not None and source_type == 'gaussian':
                cx = center_x_range[0] + (center_x_range[1] - center_x_range[0]) * torch.rand(1).item()
                cy = center_y_range[0] + (center_y_range[1] - center_y_range[0]) * torch.rand(1).item()
            else:
                cx, cy = center_x, center_y
            
            # Generate sensors
            u0_sensors = self.generate_ic_sine_series(a_coeffs)
            v0_sensors = self.generate_ic_sine_series(b_coeffs)
            src_sensors = self.generate_source(source_type, source_amplitude, cx, cy)
            
            # Sample 3D collocation points
            x_colloc = self.domain[0] + (self.domain[1] - self.domain[0]) * torch.rand(n_colloc)
            y_colloc = self.domain[0] + (self.domain[1] - self.domain[0]) * torch.rand(n_colloc)
            t_colloc = T_max * torch.rand(n_colloc)
            xyt_colloc = torch.stack([x_colloc, y_colloc, t_colloc], dim=1)
            
            # Source values at collocation points
            src_colloc = self.source_function(x_colloc, y_colloc, source_type, source_amplitude, cx, cy)
            
            # PDE Loss
            residual = self.compute_pde_residual(u0_sensors, v0_sensors, src_sensors, 
                                                 xyt_colloc, src_colloc)
            loss_pde = torch.mean(residual ** 2)
            
            # Displacement IC Loss
            x_ic_1d = torch.linspace(self.domain[0], self.domain[1], n_ic)
            y_ic_1d = torch.linspace(self.domain[0], self.domain[1], n_ic)
            X_ic, Y_ic = torch.meshgrid(x_ic_1d, y_ic_1d, indexing='ij')
            x_ic = X_ic.flatten()
            y_ic = Y_ic.flatten()
            t_ic = torch.zeros_like(x_ic)
            xyt_ic = torch.stack([x_ic, y_ic, t_ic], dim=1)
            
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

            # === Total Loss ===
            context = {
                'loss_values': {
                    'PDE': loss_pde.item(),
                    'IC_u': loss_ic_u.item(),
                    'IC_v': loss_ic_v.item(),
                },
                'model': self,
                'xyt_colloc': xyt_colloc,
                'xyt_ic': xyt_ic,
                'u0_sensors': u0_sensors,
                'v0_sensors': v0_sensors,
                'src_sensors': src_sensors,
                'src_colloc': src_colloc,
                'a_coeffs': a_coeffs,
                'b_coeffs': b_coeffs,
                'output_fold': output_fold
            }
            
            # Subsample for NTK computation (reduce from 800/2500 to ~100/200 points)
            if strategy == 'ntk':
                n_pde_subsample = min(100, xyt_colloc.shape[0])
                n_ic_subsample = min(200, xyt_ic.shape[0])
                
                idx_pde = torch.randperm(xyt_colloc.shape[0])[:n_pde_subsample]
                idx_ic = torch.randperm(xyt_ic.shape[0])[:n_ic_subsample]
                
                context['xyt_colloc'] = xyt_colloc[idx_pde]
                context['src_colloc'] = src_colloc[idx_pde]
                context['xyt_ic'] = xyt_ic[idx_ic]

            weights = adaptive_weights.update(epoch, context) if strategy != 'fixed' else initial_weights
            loss_total = weights['PDE'] * loss_pde + weights['IC_u'] * loss_ic_u + weights['IC_v'] * loss_ic_v

            
            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            
            history['total'].append(loss_total.item())
            history['pde'].append(loss_pde.item())
            history['ic_u'].append(loss_ic_u.item())
            history['ic_v'].append(loss_ic_v.item())
            
            # Evaluate on test set every eval_freq epochs
            if (epoch + 1) % eval_freq == 0:
                test_metric = 0.0
                test_pde_loss_sum = 0.0
                test_ic_u_loss_sum = 0.0
                test_ic_v_loss_sum = 0.0
                for test_case in test_cases:
                    u0_test = self.generate_ic_sine_series(test_case['a_coeffs'])
                    v0_test = self.generate_ic_sine_series(test_case['b_coeffs'])
                    src_test = self.generate_source(source_type, source_amplitude, test_case['center_x'], test_case['center_y'])
                    
                    # Sample test collocation and IC points
                    x_test_colloc = self.domain[0] + (self.domain[1] - self.domain[0]) * torch.rand(n_colloc)
                    y_test_colloc = self.domain[0] + (self.domain[1] - self.domain[0]) * torch.rand(n_colloc)
                    t_test_colloc = T_max * torch.rand(n_colloc)
                    xyt_test_colloc = torch.stack([x_test_colloc, y_test_colloc, t_test_colloc], dim=1)
                    src_test_colloc = self.source_function(x_test_colloc, y_test_colloc, source_type, source_amplitude, test_case['center_x'], test_case['center_y'])
                    
                    # Test PDE loss (requires gradients)
                    xyt_test_grad = xyt_test_colloc.clone().requires_grad_(True)
                    u_test = self.forward(u0_test, v0_test, src_test, xyt_test_grad)
                    grad_u_test = torch.autograd.grad(u_test, xyt_test_grad, torch.ones_like(u_test), create_graph=True)[0]
                    u_x_test = grad_u_test[:, 0]
                    u_y_test = grad_u_test[:, 1]
                    u_t_test = grad_u_test[:, 2]
                    u_xx_test = torch.autograd.grad(u_x_test, xyt_test_grad, torch.ones_like(u_x_test), create_graph=True)[0][:, 0]
                    u_yy_test = torch.autograd.grad(u_y_test, xyt_test_grad, torch.ones_like(u_y_test), create_graph=True)[0][:, 1]
                    u_tt_test = torch.autograd.grad(u_t_test, xyt_test_grad, torch.ones_like(u_t_test), create_graph=True)[0][:, 2]
                    residual_test = u_tt_test + self.k * u_t_test - self.c**2 * (u_xx_test + u_yy_test) - src_test_colloc
                    loss_pde_test = torch.mean(residual_test ** 2)
                    
                    # Test IC losses (no gradients needed for prediction, but needed for velocity)
                    with torch.no_grad():
                        x_test_ic_1d = torch.linspace(self.domain[0], self.domain[1], n_ic)
                        y_test_ic_1d = torch.linspace(self.domain[0], self.domain[1], n_ic)
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
                    test_metric += np.sqrt(loss_pde_test.item()**2 + loss_ic_u_test.item()**2 + loss_ic_v_test.item()**2)
                
                
                # Average test metric and losses over all test cases
                test_metric /= n_test_cases
                test_pde_loss_avg = test_pde_loss_sum / n_test_cases
                test_ic_u_loss_avg = test_ic_u_loss_sum / n_test_cases
                test_ic_v_loss_avg = test_ic_v_loss_sum / n_test_cases
                history['test_metric'].append(test_metric)
                
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
                    print(f"  ✓ Test metric improved to {test_metric:.6e} at epoch {epoch}")
                else:
                    epochs_without_improvement += eval_freq
                    if epochs_without_improvement % (eval_freq * 5) == 0:  # Print every 5 evals
                        print(f"  No improvement for {epochs_without_improvement} epochs (best: {best_test_loss:.6e})")
                
                # Early stopping
                if epochs_without_improvement >= early_stopping_patience:
                    print(f"\n✓ Early stopping triggered at epoch {epoch} (patience={early_stopping_patience})")
                    break
            
            scheduler.step(loss_total)

            if strategy == 'equal_init' and epoch == 2:
                print(f"\nInitial Weights -> PDE: {weights['PDE']:.2e}, IC_u: {weights['IC_u']:.2e}, IC_v: {weights['IC_v']:.2e}\n")
                
            if epoch % 500 == 0 or epoch == n_epochs - 1:
                current_lr = optimizer.param_groups[0]['lr']
                # Diagnostic: check prediction scale
                with torch.no_grad():
                    u_sample = self.forward(u0_sensors, v0_sensors, src_sensors, xyt_ic[:100])
                    u_min, u_max = u_sample.min().item(), u_sample.max().item()
                print(f"Epoch {epoch:5d} | Loss: {loss_total.item():.6f} | "
                      f"PDE: {loss_pde.item():.6f} | IC_u: {loss_ic_u.item():.6f} | "
                      f"IC_v: {loss_ic_v.item():.6f} | LR: {current_lr:.2e} | "
                      f"u_range: [{u_min:.3f}, {u_max:.3f}]")
                if strategy not in ['fixed', 'equal_init']:
                    print(f"          Weights -> PDE: {weights['PDE']:.2e}, IC_u: {weights['IC_u']:.2e}, IC_v: {weights['IC_v']:.2e}")
        
        if strategy not in ['fixed', 'equal_init']:
            adaptive_weights.plot_weights_evolution(output_fold)
            if strategy == 'ntk':
                adaptive_weights.plot_ntk_traces(output_fold)
        
        # Restore best model
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
            print(f"\n✓ Restored best model from epoch {best_epoch} with test metric {best_test_loss:.6f}")
            print(f"  Best model loss components -> PDE: {best_test_losses['pde']:.6e}, IC_u: {best_test_losses['ic_u']:.6e}, IC_v: {best_test_losses['ic_v']:.6e}")
        
        print("\n✓ Training complete!\n")
        return history, {
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