import numpy as np
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')


class BranchNet(nn.Module):
    """Branch network: encodes function inputs from sensor measurements"""
    def __init__(self, n_sensors, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_sensors, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, sensors):
        return self.net(sensors)


class TrunkNet(nn.Module):
    """Trunk network: encodes spatiotemporal locations (x, t)"""
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),  # Input: [x, t]
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, xt):
        """
        Args:
            xt: (n_points, 2) with columns [x, t]
        """
        return self.net(xt)


class PINNDeepONet_Wave(nn.Module):
    """
    Physics-Informed DeepONet for the damped wave equation:
        u_tt + k*u_t = c² * u_xx + f(x)
        
    with boundary conditions:
        u(0, t) = u(1, t) = 0
        
    and initial conditions:
        u(x, 0) = u0(x)
        u_t(x, 0) = v0(x)
        
    This implementation uses:
    - Branch_IC: encodes BOTH initial displacement u0 and initial velocity v0
      (input: 2*n_sensors_ic, output: 2*p, split into b_u and b_v)
    - Branch_Source: encodes the source function f(x)
    - Trunk: encodes spatiotemporal coordinates (x, t)
    - Hard BC enforcement via transformation: u = x*(1-x)*u_net
    
    Args:
        n_sensors_ic: number of sensor points for IC sampling
        n_sensors_src: number of sensor points for source sampling
        branch_hidden: hidden layer size in branch networks
        trunk_hidden: hidden layer size in trunk network
        p: embedding dimension
        wave_speed: wave propagation speed c
        damping_coeff: damping coefficient k (k*u_t term)
    """
    
    def __init__(self, n_sensors_ic=20, n_sensors_src=20, 
                 branch_hidden=50, trunk_hidden=50, p=50, wave_speed=1.0, damping_coeff=1.0):
        super().__init__()
        
        self.n_sensors_ic = n_sensors_ic
        self.n_sensors_src = n_sensors_src
        self.p = p
        self.c = wave_speed  # Wave speed
        self.k = damping_coeff  # Damping coefficient
        self.domain = [0.0, 1.0]  # Spatial domain
        
        # IC branch takes BOTH u0 and v0 sensors, outputs 2*p
        # Input: concatenated [u0_sensors, v0_sensors] → 2*n_sensors_ic
        self.branch_ic = BranchNet(2 * n_sensors_ic, branch_hidden, 2 * p)
        
        # Source branch
        self.branch_source = BranchNet(n_sensors_src, branch_hidden, p)
        
        # Single trunk network
        self.trunk = TrunkNet(trunk_hidden, p)
        
        # Bias term
        self.bias = nn.Parameter(torch.zeros(1))
        
        # Fixed sensor locations
        self.register_buffer('sensor_x_ic', 
                            torch.linspace(self.domain[0], self.domain[1], n_sensors_ic))
        self.register_buffer('sensor_x_src', 
                            torch.linspace(self.domain[0], self.domain[1], n_sensors_src))
    
    def forward(self, u0_sensors, v0_sensors, src_sensors, xt):
        """
        Args:
            u0_sensors: (batch, n_sensors_ic) or (n_sensors_ic,) - displacement IC
            v0_sensors: (batch, n_sensors_ic) or (n_sensors_ic,) - velocity IC
            src_sensors: (batch, n_sensors_src) or (n_sensors_src,) - forcing
            xt: (n_points, 2) spatiotemporal coordinates [x, t]
        
        Returns:
            u: (batch, n_points) or (n_points,) - displacement
        """
        # Handle single sample
        single_sample = u0_sensors.dim() == 1
        if single_sample:
            u0_sensors = u0_sensors.unsqueeze(0)
            v0_sensors = v0_sensors.unsqueeze(0)
            src_sensors = src_sensors.unsqueeze(0)
        
        # Concatenate ICs and encode together
        ic_concat = torch.cat([u0_sensors, v0_sensors], dim=1)  # (batch, 2*n_sensors_ic)
        ic_encoded = self.branch_ic(ic_concat)  # (batch, 2*p)
        
        # Split into displacement and velocity contributions
        b_u = ic_encoded[:, :self.p]    # (batch, p) - displacement contribution
        b_v = ic_encoded[:, self.p:]    # (batch, p) - velocity contribution
        
        # Encode source
        b_src = self.branch_source(src_sensors)  # (batch, p)
        
        # Combine all contributions
        b_combined = b_u + b_v + b_src  # (batch, p)
        
        # Encode spatiotemporal locations
        tau = self.trunk(xt)  # (n_points, p)
        
        # DeepONet operation: (batch, p) @ (p, n_points) = (batch, n_points)
        u_net = torch.matmul(b_combined, tau.T) + self.bias
        
        # Extract x coordinates
        x = xt[:, 0]  # (n_points,)
        
        # Hard BC enforcement: u = x * (x - 1) * u_net
        # At x=0: u=0, at x=1: u=0
        bc_factor = x * (x - 1.0)  # (n_points,)
        
        # Apply BC factor (broadcast over batch dimension)
        if single_sample:
            u = bc_factor * u_net
        else:
            u = bc_factor.unsqueeze(0) * u_net  # (batch, n_points)
        
        return u.squeeze(0) if single_sample else u
    
    def compute_pde_residual(self, u0_sensors, v0_sensors, src_sensors, xt, src_values):
        """
        Compute PDE residual: R = u_tt - c^2 * u_xx - f(x)
        
        Args:
            u0_sensors: (n_sensors_ic,) displacement IC measurements
            v0_sensors: (n_sensors_ic,) velocity IC measurements
            src_sensors: (n_sensors_src,) source measurements
            xt: (n_points, 2) collocation points [x, t]
            src_values: (n_points,) source values f(x) at collocation points
        
        Returns:
            residual: (n_points,)
        """
        xt_requires_grad = xt.clone().requires_grad_(True)
        
        # Forward pass
        u = self.forward(u0_sensors, v0_sensors, src_sensors, xt_requires_grad)
        
        # Compute u_t (first time derivative)
        u_t = torch.autograd.grad(u, xt_requires_grad, 
                                  torch.ones_like(u),
                                  create_graph=True)[0][:, 1]  # ∂u/∂t
        
        # Compute u_tt (second time derivative)
        u_tt = torch.autograd.grad(u_t, xt_requires_grad,
                                   torch.ones_like(u_t),
                                   create_graph=True)[0][:, 1]  # ∂²u/∂t²
        
        # Compute u_x (first space derivative)
        u_x = torch.autograd.grad(u, xt_requires_grad,
                                  torch.ones_like(u),
                                  create_graph=True)[0][:, 0]  # ∂u/∂x
        
        # Compute u_xx (second space derivative)
        u_xx = torch.autograd.grad(u_x, xt_requires_grad,
                                   torch.ones_like(u_x),
                                   create_graph=True)[0][:, 0]  # ∂²u/∂x²
        
        # PDE residual: u_tt - c^2 * u_xx - f(x)
        residual = u_tt - self.c**2 * u_xx - src_values + self.k * u_t
        
        return residual
    
    def generate_ic_displacement(self, a):
        """
        Generate displacement IC: u(x, 0) = a * sin(pi * x)
        
        Args:
            a: amplitude parameter
        
        Returns:
            u0_sensors: displacement IC values at sensor locations
        """
        return a * torch.sin(np.pi * self.sensor_x_ic)
    
    def generate_ic_velocity(self, b):
        """
        Generate velocity IC: u_t(x, 0) = b * sin(pi * x)
        
        Args:
            b: amplitude parameter (can be 0 for zero initial velocity)
        
        Returns:
            v0_sensors: velocity IC values at sensor locations
        """
        return b * torch.sin(np.pi * self.sensor_x_ic)
    
    def generate_source(self, source_type='gaussian', amplitude=0.5, center=0.5):
        """
        Generate forcing f(x)
        
        Args:
            source_type: 'gaussian', 'sine', 'constant', 'zero'
            amplitude: source strength
            center: center location for Gaussian source (default: 0.5)
        
        Returns:
            src_sensors: source values at sensor locations
        """
        x = self.sensor_x_src
        
        if source_type == 'gaussian':
            # Gaussian centered at specified location
            width = 0.3
            src = amplitude * torch.exp(-((x - center) / width) ** 2)
        elif source_type == 'sine':
            # Sine wave
            src = amplitude * torch.sin(2 * np.pi * x)
        elif source_type == 'constant':
            # Constant source
            src = amplitude * torch.ones_like(x)
        elif source_type == 'zero':
            # No forcing
            src = torch.zeros_like(x)
        else:
            raise ValueError(f"Unknown source type: {source_type}")
        
        return src
    
    def source_function(self, x, source_type='gaussian', amplitude=0.5, center=0.5):
        """
        Evaluate source function at arbitrary locations
        
        Args:
            x: (n_points,) or scalar
            source_type: type of source
            amplitude: source strength
            center: center location for Gaussian source (default: 0.5)
        
        Returns:
            f: source values at x
        """
        if source_type == 'gaussian':
            width = 0.3
            f = amplitude * torch.exp(-((x - center) / width) ** 2)
        elif source_type == 'sine':
            f = amplitude * torch.sin(2 * np.pi * x)
        elif source_type == 'constant':
            f = amplitude * torch.ones_like(x)
        elif source_type == 'zero':
            f = torch.zeros_like(x)
        else:
            raise ValueError(f"Unknown source type: {source_type}")
        
        return f
    
    def train_pinn(self, n_epochs=5000, n_colloc=200, lr=1e-3, 
                   a_range=(0.0, 1.0), b_range=(0.0, 4.0), 
                   source_type='zero', source_amplitude=7.5, source_center=0.5, 
                   center_range=None, T_max=1.0):
        """
        Train PINN-DeepONet with physics-informed loss
        
        Loss = w_pde * L_pde + w_ic_u * L_ic_u + w_ic_v * L_ic_v
        Note: BC is enforced as hard constraint, no soft BC loss needed
        
        Args:
            n_epochs: number of training epochs
            n_colloc: number of collocation points
            lr: learning rate
            a_range: range for displacement IC amplitude
            b_range: range for velocity IC amplitude (0,0) = zero velocity
            source_type: type of forcing
            source_amplitude: forcing strength
            source_center: center location for Gaussian (used if center_range is None)
            center_range: if provided, randomly sample center from this range (e.g., (0.1, 0.9))
            T_max: maximum time for training
        """
        print("="*70)
        print("TRAINING PHYSICS-INFORMED DEEPONET FOR WAVE EQUATION")
        print("="*70)
        print(f"Epochs: {n_epochs}")
        print(f"Collocation points: {n_colloc}")
        print(f"Displacement IC amplitude range: {a_range}")
        print(f"Velocity IC amplitude range: {b_range}")
        if center_range is not None:
            print(f"Source type: {source_type}, amplitude: {source_amplitude}, center range: {center_range}")
        else:
            print(f"Source type: {source_type}, amplitude: {source_amplitude}, center: {source_center}")
        print(f"Wave speed c: {self.c}")
        print(f"Time domain: [0, {T_max}]")
        print("Boundary conditions: HARD constraints (no soft loss)")
        print("="*70)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=500, verbose=True
        )
        
        # Loss weights
        w_pde = 1.0
        w_ic_u = 10.0   # Displacement IC
        w_ic_v = 10.0   # Velocity IC
        
        history = {
            'total': [], 'pde': [], 'ic_u': [], 'ic_v': []
        }
        
        for epoch in range(n_epochs):
            # Sample random IC amplitudes
            a = a_range[0] + (a_range[1] - a_range[0]) * torch.rand(1).item()
            b = b_range[0] + (b_range[1] - b_range[0]) * torch.rand(1).item()
            
            # Sample random source center (for varying forcing location during training)
            if center_range is not None and source_type == 'gaussian':
                # Vary center in [center_range[0], center_range[1]] for generalization
                center_sample = center_range[0] + (center_range[1] - center_range[0]) * torch.rand(1).item()
            else:
                center_sample = source_center
            
            # Generate ICs and source sensors
            u0_sensors = self.generate_ic_displacement(a)
            v0_sensors = self.generate_ic_velocity(b)
            src_sensors = self.generate_source(source_type, source_amplitude, center_sample)
            
            # Sample collocation points (x, t)
            x_colloc = self.domain[0] + (self.domain[1] - self.domain[0]) * torch.rand(n_colloc)
            t_colloc = T_max * torch.rand(n_colloc)  # t ∈ [0, T_max]
            xt_colloc = torch.stack([x_colloc, t_colloc], dim=1)
            
            # Source values at collocation points
            src_colloc = self.source_function(x_colloc, source_type, source_amplitude, center_sample)
            
            # === PDE Loss ===
            residual = self.compute_pde_residual(u0_sensors, v0_sensors, src_sensors, 
                                                 xt_colloc, src_colloc)
            loss_pde = torch.mean(residual ** 2)
            
            # === Displacement IC Loss: u(x, 0) = a * sin(pi * x) ===
            n_ic = 50
            x_ic = torch.linspace(self.domain[0], self.domain[1], n_ic)
            t_ic = torch.zeros(n_ic)
            xt_ic = torch.stack([x_ic, t_ic], dim=1)
            
            u_ic_pred = self.forward(u0_sensors, v0_sensors, src_sensors, xt_ic)
            u_ic_true = a * torch.sin(np.pi * x_ic)
            loss_ic_u = torch.mean((u_ic_pred - u_ic_true) ** 2)
            
            # === Velocity IC Loss: u_t(x, 0) = b * sin(pi * x) ===
            xt_ic_grad = xt_ic.clone().requires_grad_(True)
            u_ic_grad = self.forward(u0_sensors, v0_sensors, src_sensors, xt_ic_grad)
            
            u_t_ic_pred = torch.autograd.grad(u_ic_grad, xt_ic_grad,
                                              torch.ones_like(u_ic_grad),
                                              create_graph=True)[0][:, 1]  # ∂u/∂t at t=0
            
            v_ic_true = b * torch.sin(np.pi * x_ic)
            loss_ic_v = torch.mean((u_t_ic_pred - v_ic_true) ** 2)
            
            # === Total Loss ===
            loss_total = w_pde * loss_pde + w_ic_u * loss_ic_u + w_ic_v * loss_ic_v
            
            # Optimization step
            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Record history
            history['total'].append(loss_total.item())
            history['pde'].append(loss_pde.item())
            history['ic_u'].append(loss_ic_u.item())
            history['ic_v'].append(loss_ic_v.item())
            
            scheduler.step(loss_total)
            
            # Print progress
            if epoch % 500 == 0 or epoch == n_epochs - 1:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:5d} | Loss: {loss_total.item():.6f} | "
                      f"PDE: {loss_pde.item():.6f} | IC_u: {loss_ic_u.item():.6f} | "
                      f"IC_v: {loss_ic_v.item():.6f} | LR: {current_lr:.2e}")
        
        print("\n✓ Training complete!\n")
        return history