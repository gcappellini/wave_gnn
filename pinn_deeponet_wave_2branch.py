import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
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


def plot_solution(model, a_test=1.5, b_test=0.0, source_type='zero', 
                  source_amplitude=0.0, source_center=0.5, T_max=1.0, gt_data=None):
    """Visualize the trained solution"""
    
    # Generate test case
    u0_sensors = model.generate_ic_displacement(a_test)
    v0_sensors = model.generate_ic_velocity(b_test)
    src_sensors = model.generate_source(source_type, source_amplitude, source_center)
    
    # Create spatiotemporal grid
    nx, nt = 100, 50
    x_plot = torch.linspace(model.domain[0], model.domain[1], nx)
    t_plot = torch.linspace(0, T_max, nt)
    
    X, T = torch.meshgrid(x_plot, t_plot, indexing='ij')
    xt_grid = torch.stack([X.flatten(), T.flatten()], dim=1)
    
    # Predict solution
    with torch.no_grad():
        u_pred = model.forward(u0_sensors, v0_sensors, src_sensors, xt_grid)
        u_pred = u_pred.reshape(nx, nt).numpy()
    
    # Source and ICs for plotting
    src_plot = model.source_function(x_plot, source_type, source_amplitude, source_center).numpy()
    u0_plot = (a_test * torch.sin(np.pi * x_plot)).numpy()
    v0_plot = (b_test * torch.sin(np.pi * x_plot)).numpy()
    
    # Determine figure layout based on whether gt_data is provided
    if gt_data is not None:
        fig = plt.figure(figsize=(18, 10))
        nrows = 2
        
        # Reshape ground truth data to match prediction grid
        # gt_data format: [x, t, f, u, v] - reshape to (nt, nx) for u and v
        x_gt = gt_data[:, 0]
        t_gt = gt_data[:, 1]
        f_gt = gt_data[:, 2]
        u_gt = gt_data[:, 3]
        v_gt = gt_data[:, 4]
        
        # Determine grid size from gt_data
        nt_gt = len(np.unique(t_gt))
        nx_gt = len(np.unique(x_gt))
        u_gt_grid = u_gt.reshape(nt_gt, nx_gt)  # Shape: (nt, nx)
        
        # Interpolate gt to match prediction grid if needed
        if (nt_gt != nt) or (nx_gt != nx):
            from scipy.interpolate import RegularGridInterpolator
            t_gt_unique = np.unique(t_gt)
            x_gt_unique = np.unique(x_gt)
            interp = RegularGridInterpolator((t_gt_unique, x_gt_unique), u_gt_grid)
            points = np.column_stack([T.numpy().ravel(), X.numpy().ravel()])
            u_gt_interp = interp(points).reshape(nx, nt)
        else:
            u_gt_interp = u_gt_grid.T  # Transpose to (nx, nt) to match u_pred
        
        # Compute absolute error
        abs_error = np.abs(u_pred - u_gt_interp)
        
    else:
        fig = plt.figure(figsize=(18, 5))
        nrows = 1
    
    # Plot 1: Solution evolution (snapshots)
    ax1 = plt.subplot(nrows, 3, 1)
    time_indices = [0, nt//4, nt//2, 3*nt//4, -1]
    for idx in time_indices:
        t_val = t_plot[idx].item()
        ax1.plot(x_plot.numpy(), u_pred[:, idx], label=f't={t_val:.3f}')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('u(x,t)', fontsize=12)
    ax1.set_title(f'PINN Solution (a={a_test}, b={b_test})', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: PINN Spatiotemporal heatmap
    ax2 = plt.subplot(nrows, 3, 2)
    im = ax2.contourf(T.numpy(), X.numpy(), u_pred, levels=20, cmap='RdBu_r')
    ax2.set_xlabel('t', fontsize=12)
    ax2.set_ylabel('x', fontsize=12)
    ax2.set_title('PINN: u(x,t)', fontsize=12)
    plt.colorbar(im, ax=ax2)
    
    # Plot 3: ICs and Source
    ax3 = plt.subplot(nrows, 3, 3)
    ax3.plot(x_plot.numpy(), u0_plot, 'b-', linewidth=2, label='IC: u(x,0)')
    ax3.plot(x_plot.numpy(), v0_plot, 'g--', linewidth=2, label='IC: u_t(x,0)')
    ax3.plot(x_plot.numpy(), src_plot, 'r-.', linewidth=2, label='Source f(x)')
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('Value', fontsize=12)
    ax3.set_title('Initial Conditions & Forcing', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Additional plots if ground truth is provided
    if gt_data is not None:
        # Plot 4: MATLAB Ground Truth
        ax4 = plt.subplot(2, 3, 4)
        im4 = ax4.contourf(T.numpy(), X.numpy(), u_gt_interp, levels=20, cmap='RdBu_r')
        ax4.set_xlabel('t', fontsize=12)
        ax4.set_ylabel('x', fontsize=12)
        ax4.set_title('MATLAB: u(x,t)', fontsize=12)
        plt.colorbar(im4, ax=ax4)
        
        # Plot 5: Absolute Error
        ax5 = plt.subplot(2, 3, 5)
        im5 = ax5.contourf(T.numpy(), X.numpy(), abs_error, levels=20, cmap='hot')
        ax5.set_xlabel('t', fontsize=12)
        ax5.set_ylabel('x', fontsize=12)
        max_err = np.max(abs_error)
        mean_err = np.mean(abs_error)
        ax5.set_title(f'Absolute Error\nmax={max_err:.2e}, mean={mean_err:.2e}', fontsize=12)
        plt.colorbar(im5, ax=ax5)
        
        # Plot 6: Error statistics
        ax6 = plt.subplot(2, 3, 6)
        # Plot error over time (spatial average)
        error_vs_time = np.mean(abs_error, axis=0)
        ax6.plot(t_plot.numpy(), error_vs_time, 'r-', linewidth=2)
        ax6.set_xlabel('t', fontsize=12)
        ax6.set_ylabel('Mean Absolute Error', fontsize=12)
        ax6.set_title('Error Evolution', fontsize=12)
        ax6.grid(True, alpha=0.3)
        
        # Print error statistics
        print(f"\n=== Error Statistics vs MATLAB ===")
        print(f"Max absolute error: {max_err:.6e}")
        print(f"Mean absolute error: {mean_err:.6e}")
        print(f"RMS error: {np.sqrt(np.mean(abs_error**2)):.6e}")
        print(f"Relative L2 error: {np.linalg.norm(abs_error)/np.linalg.norm(u_gt_interp):.6e}")
    
    plt.tight_layout()
    return fig


def plot_training_history(history):
    """Plot training loss history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].semilogy(history['total'], 'k-', linewidth=1.5)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].semilogy(history['pde'], 'b-', linewidth=1.5)
    axes[0, 1].set_title('PDE Residual Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].semilogy(history['ic_u'], 'r-', linewidth=1.5)
    axes[1, 0].set_title('Displacement IC Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].semilogy(history['ic_v'], 'g-', linewidth=1.5)
    axes[1, 1].set_title('Velocity IC Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def rollout_test(model, gt_data, T_total=10.0, dt_interval=1.0, 
                 nx=100, nt_per_interval=50):
    """
    Rollout test with changing source center every dt_interval seconds
    Uses ground truth data to extract sensor values at each interval
    
    Args:
        model: trained PINN-DeepONet model
        gt_data: ground truth data [x, t, f, u, v] (REQUIRED)
        T_total: total simulation time (default: 10s)
        dt_interval: time interval for source center change (default: 1s)
        nx: number of spatial points
        nt_per_interval: number of time points per interval
    
    Returns:
        u_rollout: complete solution (nx, nt_total)
        t_rollout: time vector
        x_rollout: spatial vector
        fig: matplotlib figure
    """
    
    if gt_data is None:
        raise ValueError("Ground truth data is required for rollout test!")
    
    # Extract ground truth - format: [x, t, f, u, v]
    x_gt = gt_data[:, 0]
    t_gt = gt_data[:, 1]
    f_gt = gt_data[:, 2]
    u_gt = gt_data[:, 3]
    v_gt = gt_data[:, 4]
    
    # Get unique coordinates
    x_unique = np.unique(x_gt)
    t_unique = np.unique(t_gt)
    nt_gt = len(t_unique)
    nx_gt = len(x_unique)
    
    # Reshape to grids
    u_gt_grid = u_gt.reshape(nt_gt, nx_gt).T  # (nx_gt, nt_gt)
    v_gt_grid = v_gt.reshape(nt_gt, nx_gt).T
    f_gt_grid = f_gt.reshape(nt_gt, nx_gt).T
    
    # Setup
    n_intervals = int(T_total / dt_interval)
    x_plot = torch.linspace(model.domain[0], model.domain[1], nx)
    t_interval = torch.linspace(0, dt_interval, nt_per_interval)
    
    # Storage for complete rollout
    u_rollout = []
    
    # Create fixed xt_grid for each interval
    X, T = torch.meshgrid(x_plot, t_interval, indexing='ij')
    xt_grid = torch.stack([X.flatten(), T.flatten()], dim=1)
    
    model.eval()
    with torch.no_grad():
        for interval_idx in range(n_intervals):
            # Find the time index in ground truth for start of this interval
            t_start = interval_idx * dt_interval
            t_idx = np.argmin(np.abs(t_unique - t_start))
            
            # Extract u, v, f from ground truth at this time step
            u_current_gt = u_gt_grid[:, t_idx]  # (nx_gt,)
            v_current_gt = v_gt_grid[:, t_idx]
            f_current_gt = f_gt_grid[:, t_idx]
            
            # Interpolate to sensor locations if needed
            u0_sensors = torch.tensor(np.interp(model.sensor_x_ic.numpy(), 
                                                 x_unique, u_current_gt), dtype=torch.float32)
            v0_sensors = torch.tensor(np.interp(model.sensor_x_ic.numpy(), 
                                                 x_unique, v_current_gt), dtype=torch.float32)
            src_sensors = torch.tensor(np.interp(model.sensor_x_src.numpy(), 
                                                  x_unique, f_current_gt), dtype=torch.float32)
            
            # Predict for this interval
            u_pred = model.forward(u0_sensors, v0_sensors, src_sensors, xt_grid)
            u_pred_grid = u_pred.reshape(nx, nt_per_interval)
            
            # Store results
            u_rollout.append(u_pred_grid.numpy())
    
    # Concatenate all intervals
    u_rollout = np.concatenate(u_rollout, axis=1)
    
    # Create full time and space arrays
    t_rollout = np.linspace(0, T_total, n_intervals * nt_per_interval)
    x_rollout = x_plot.numpy()
    
    # Interpolate ground truth to match rollout grid
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((x_unique, t_unique), u_gt_grid)
    X_new, T_new = np.meshgrid(x_rollout, t_rollout, indexing='ij')
    points = np.column_stack([X_new.ravel(), T_new.ravel()])
    u_gt_interp = interp(points).reshape(u_rollout.shape)
    
    # Plot results
    fig = plt.figure(figsize=(18, 10))
    
    # 3 subplots: PINN, Ground Truth, Error
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.contourf(t_rollout, x_rollout, u_rollout, levels=50, cmap='viridis')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('x')
    ax1.set_title('PINN Rollout Prediction')
    plt.colorbar(im1, ax=ax1)
    
    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.contourf(t_rollout, x_rollout, u_gt_interp, levels=50, cmap='viridis')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('x')
    ax2.set_title('MATLAB Ground Truth')
    plt.colorbar(im2, ax=ax2)
    
    ax3 = plt.subplot(1, 3, 3)
    error = np.abs(u_rollout - u_gt_interp)
    rel_l2_error = np.linalg.norm(error)/np.linalg.norm(u_gt_interp)
    im3 = ax3.contourf(t_rollout, x_rollout, error, levels=50, cmap='hot')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('x')
    ax3.set_title(f'Absolute Error\n(Rel L2: {rel_l2_error:.2e})')
    plt.colorbar(im3, ax=ax3)
    
    print(f"\n=== Rollout Error Statistics ===")
    print(f"Max absolute error: {np.max(error):.6e}")
    print(f"Mean absolute error: {np.mean(error):.6e}")
    print(f"RMS error: {np.sqrt(np.mean(error**2)):.6e}")
    print(f"Relative L2 error: {rel_l2_error:.6e}")
    
    plt.tight_layout()
    return u_rollout, t_rollout, x_rollout, fig


if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # ==========================================================================
    # TRAINING CASE SELECTION
    # ==========================================================================
    # Choose training scenario:
    # 'no_source': Free wave with zero forcing
    # 'with_source': Wave with Gaussian source (varying amplitude)
    
    TRAINING_CASE = 'with_source'  # Change to 'no_source' or 'with_source'
    
    print("\n" + "="*70)
    print("PHYSICS-INFORMED DEEPONET FOR DAMPED WAVE EQUATION")
    print("="*70)
    print("PDE: u_tt + k*u_t = c² * u_xx + f(x)")
    print("IC:  u(x, 0) = a * sin(π*x)")
    print("     u_t(x, 0) = b * sin(π*x)")
    print("BC:  u(0, t) = u(1, t) = 0 (hard constraint)")
    print("="*70)
    
    # Create model
    model = PINNDeepONet_Wave(
        n_sensors_ic=20,
        n_sensors_src=20,
        branch_hidden=50,
        trunk_hidden=50,
        p=50,
        wave_speed=1.0,  # c=1, so c²=1
        damping_coeff=1.0
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training case: {TRAINING_CASE}")
    print()
    
    # ==========================================================================
    # TRAINING
    # ==========================================================================
    
    if TRAINING_CASE == 'no_source':
        # Case 1: Free wave (no forcing)
        print("Training: Free damped wave (no source)")
        history = model.train_pinn(
            n_epochs=5000,
            n_colloc=200,
            lr=1e-3,
            a_range=(-1.0, 1.0),      # Displacement IC amplitude range
            b_range=(-4.0, 4.0),      # Initial velocity range
            source_type='zero',      # No forcing
            source_amplitude=7.5,    # Default amplitude (not used for zero source)
            T_max=2.0                # Time domain
        )
        
        # Test case
        a_test = 0.5
        b_test = 2.0
        source_test_type = 'zero'
        source_test_amp = 7.5  # Default (not used for zero source)
        source_test_center = 0.5  # Not used for zero source
        T_test = 2.0
        model_filename = 'pinn_deeponet_wave_nosource.pth'
        gt_filename = 'gt_wave1D_nosource.csv'
        
    elif TRAINING_CASE == 'with_source':
        # Case 2: Wave with Gaussian source (varying center location during training)
        print("Training: Damped wave with Gaussian source (varying center)")
        print("  Source amplitude: 7.5 (fixed)")
        print("  Source width: 0.3")
        print("  Training center range: [0.1, 0.9]")
        
        history = model.train_pinn(
            n_epochs=5000,
            n_colloc=200,
            lr=1e-3,
            a_range=(-1.0, 1.0),        # Displacement IC amplitude range
            b_range=(-4.0, 4.0),        # Initial velocity range
            source_type='gaussian',    # Gaussian forcing
            source_amplitude=7.5,      # Fixed amplitude
            source_center=0.5,         # Default center (not used when center_range is provided)
            center_range=(0.1, 0.9),   # Vary center location during training
            T_max=2.0                  # Time domain
        )
        
        # Test case: test at center x=0.5 (middle of training range)
        a_test = 0.5
        b_test = 2.0
        source_test_type = 'gaussian'
        source_test_amp = 7.5
        source_test_center = 0.17  # Test at x=0.17
        T_test = 2.0
        model_filename = 'pinn_deeponet_wave_withsource.pth'
        gt_filename = 'gt_wave1D_withsource.csv'
        
    else:
        raise ValueError(f"Unknown training case: {TRAINING_CASE}")
    
    # ==========================================================================
    # SAVE MODEL
    # ==========================================================================
    torch.save(model.state_dict(), model_filename)
    print(f"\n✓ Model saved: {model_filename}\n")
    
    # ==========================================================================
    # PLOT RESULTS
    # ==========================================================================
    print("Generating plots...")
    
    # Try to load ground truth if available
    try:
        gt_data = np.loadtxt(gt_filename, delimiter=',')
        print(f"✓ Loaded ground truth: {gt_filename}")
    except FileNotFoundError:
        print(f"⚠ Ground truth file not found: {gt_filename}")
        print("  Run MATLAB script first to generate ground truth.")
        gt_data = None
    
    # Plot solution
    fig1 = plot_solution(model, 
                         a_test=a_test, 
                         b_test=b_test, 
                         source_type=source_test_type, 
                         source_amplitude=source_test_amp,
                         source_center=source_test_center,
                         T_max=T_test, 
                         gt_data=gt_data)
    
    solution_plot_filename = f'pinn_wave_solution_{TRAINING_CASE}.png'
    fig1.savefig(solution_plot_filename, dpi=150, bbox_inches='tight')
    print(f"✓ Solution plot saved: {solution_plot_filename}")
    
    # Plot training history
    fig2 = plot_training_history(history)
    training_plot_filename = f'pinn_wave_training_{TRAINING_CASE}.png'
    fig2.savefig(training_plot_filename, dpi=150, bbox_inches='tight')
    print(f"✓ Training history saved: {training_plot_filename}")
    
    # ==========================================================================
    # ROLLOUT TEST (only for with_source case)
    # ==========================================================================
    if TRAINING_CASE == 'with_source':
        print("\n" + "="*70)
        print("ROLLOUT TEST: 10-second simulation with changing source")
        print("="*70)
        
        # Load rollout ground truth (REQUIRED)
        try:
            gt_rollout = np.loadtxt('gt_wave1D_withsource_rollout.csv', delimiter=',')
            print(f"✓ Loaded rollout ground truth: gt_wave1D_withsource_rollout.csv")
        except FileNotFoundError:
            print(f"✗ Rollout ground truth not found: gt_wave1D_withsource_rollout.csv")
            print("  Rollout test requires ground truth data. Skipping...")
            gt_rollout = None
        
        # Run rollout test only if ground truth is available
        if gt_rollout is not None:
            u_roll, t_roll, x_roll, fig3 = rollout_test(
                model, 
                gt_data=gt_rollout,
                T_total=10.0,
                dt_interval=1.0
            )
            
            rollout_plot_filename = 'pinn_wave_rollout_withsource.png'
            fig3.savefig(rollout_plot_filename, dpi=150, bbox_inches='tight')
            print(f"✓ Rollout plot saved: {rollout_plot_filename}")
            print(f"  Rollout shape: {u_roll.shape}")
    
    plt.show()
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
