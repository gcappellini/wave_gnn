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
    """Trunk network: encodes spatiotemporal locations (x, y, t)"""
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),  # Input: [x, y, t]
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, xyt):
        """
        Args:
            xyt: (n_points, 3) with columns [x, y, t]
        """
        return self.net(xyt)


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
                 branch_hidden=100, trunk_hidden=100, p=100, wave_speed=1.0, damping_coeff=1.0):
        super().__init__()
        
        self.n_sensors_ic = n_sensors_ic
        self.n_sensors_src = n_sensors_src
        self.p = p
        self.c = wave_speed
        self.k = damping_coeff
        self.domain = [0.0, 1.0]  # Square domain [0,1] × [0,1]
        
        # IC branch: takes BOTH u0 and v0 sensors (2D grids flattened)
        self.branch_ic = BranchNet(2 * n_sensors_ic * n_sensors_ic, branch_hidden, 2 * p)
        
        # Source branch: 2D grid flattened
        self.branch_source = BranchNet(n_sensors_src * n_sensors_src, branch_hidden, p)
        
        # Trunk network for (x, y, t)
        self.trunk = TrunkNet(trunk_hidden, p)
        
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
    
    def generate_ic_displacement(self, a):
        """u(x,y,0) = a * sin(π*x) * sin(π*y)"""
        return a * torch.sin(np.pi * self.sensor_x_ic) * torch.sin(np.pi * self.sensor_y_ic)
    
    def generate_ic_velocity(self, b):
        """u_t(x,y,0) = b * sin(π*x) * sin(π*y)"""
        return b * torch.sin(np.pi * self.sensor_x_ic) * torch.sin(np.pi * self.sensor_y_ic)
    
    def generate_source(self, source_type='gaussian', amplitude=7.5, center_x=0.5, center_y=0.5):
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
    
    def source_function(self, x, y, source_type='gaussian', amplitude=7.5, center_x=0.5, center_y=0.5):
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
    
    def train_pinn(self, n_epochs=5000, n_colloc=500, lr=1e-3, 
                   a_range=(0.0, 1.5), b_range=(0.0, 0.0), 
                   source_type='zero', source_amplitude=7.5, 
                   center_x=0.5, center_y=0.5, center_range=None, T_max=2.0):
        """
        Train PINN-DeepONet for 2D wave equation
        
        Args:
            center_range: if provided, tuple (min, max) for random center sampling
        """
        print(f"Epochs: {n_epochs}")
        print(f"Collocation points: {n_colloc}")
        print(f"a range: {a_range}, b range: {b_range}")
        if center_range is not None:
            print(f"Source: {source_type}, amplitude: {source_amplitude}, center range: {center_range}")
        else:
            print(f"Source: {source_type}, amplitude: {source_amplitude}, center: ({center_x}, {center_y})")
        print(f"Wave speed: {self.c}, Damping: {self.k}")
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=500, verbose=True
        )
        
        w_pde = 1.0
        w_ic_u = 200.0
        w_ic_v = 200.0
        
        history = {'total': [], 'pde': [], 'ic_u': [], 'ic_v': []}
        
        # Track best model
        best_loss = float('inf')
        best_model_state = None
        best_epoch = 0
        
        for epoch in range(n_epochs):
            # Sample random IC amplitudes
            a = a_range[0] + (a_range[1] - a_range[0]) * torch.rand(1).item()
            b = b_range[0] + (b_range[1] - b_range[0]) * torch.rand(1).item()
            
            # Sample random source center
            if center_range is not None and source_type == 'gaussian':
                cx = center_range[0] + (center_range[1] - center_range[0]) * torch.rand(1).item()
                cy = center_range[0] + (center_range[1] - center_range[0]) * torch.rand(1).item()
            else:
                cx, cy = center_x, center_y
            
            # Generate sensors
            u0_sensors = self.generate_ic_displacement(a)
            v0_sensors = self.generate_ic_velocity(b)
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
            n_ic = 50
            x_ic_1d = torch.linspace(self.domain[0], self.domain[1], n_ic)
            y_ic_1d = torch.linspace(self.domain[0], self.domain[1], n_ic)
            X_ic, Y_ic = torch.meshgrid(x_ic_1d, y_ic_1d, indexing='ij')
            x_ic = X_ic.flatten()
            y_ic = Y_ic.flatten()
            t_ic = torch.zeros_like(x_ic)
            xyt_ic = torch.stack([x_ic, y_ic, t_ic], dim=1)
            
            u_ic_pred = self.forward(u0_sensors, v0_sensors, src_sensors, xyt_ic)
            u_ic_true = a * torch.sin(np.pi * x_ic) * torch.sin(np.pi * y_ic)
            loss_ic_u = torch.mean((u_ic_pred - u_ic_true) ** 2)
            
            # Velocity IC Loss
            xyt_ic_grad = xyt_ic.clone().requires_grad_(True)
            u_ic_grad = self.forward(u0_sensors, v0_sensors, src_sensors, xyt_ic_grad)
            u_t_ic_pred = torch.autograd.grad(u_ic_grad, xyt_ic_grad,
                                              torch.ones_like(u_ic_grad),
                                              create_graph=True)[0][:, 2]
            v_ic_true = b * torch.sin(np.pi * x_ic) * torch.sin(np.pi * y_ic)
            loss_ic_v = torch.mean((u_t_ic_pred - v_ic_true) ** 2)
            
            # Total Loss
            loss_total = w_pde * loss_pde + w_ic_u * loss_ic_u + w_ic_v * loss_ic_v
            
            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            
            history['total'].append(loss_total.item())
            history['pde'].append(loss_pde.item())
            history['ic_u'].append(loss_ic_u.item())
            history['ic_v'].append(loss_ic_v.item())
            
            # Save best model
            if loss_total.item() < best_loss:
                best_loss = loss_total.item()
                best_model_state = {key: value.cpu().clone() for key, value in self.state_dict().items()}
                best_epoch = epoch
            
            scheduler.step(loss_total)
            
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
        
        # Restore best model
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
            print(f"\n✓ Restored best model from epoch {best_epoch} with loss {best_loss:.6f}")
        
        print("\n✓ Training complete!\n")
        return history
