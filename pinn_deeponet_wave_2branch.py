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
    Physics-Informed DeepONet for wave equation with source:
    u_tt = c^2 * u_xx + f(x)
    
    Two-branch architecture:
    - Branch_IC: encodes BOTH initial conditions [u(x,0), u_t(x,0)] with 2p output
    - Branch_Source: encodes forcing f(x)
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
    
    def generate_source(self, source_type='gaussian', amplitude=0.5):
        """
        Generate forcing f(x)
        
        Args:
            source_type: 'gaussian', 'sine', 'constant', 'zero'
            amplitude: source strength
        
        Returns:
            src_sensors: source values at sensor locations
        """
        x = self.sensor_x_src
        
        if source_type == 'gaussian':
            # Gaussian centered at x=0.50
            center = 0.50
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
    
    def source_function(self, x, source_type='gaussian', amplitude=0.5):
        """
        Evaluate source function at arbitrary locations
        
        Args:
            x: (n_points,) or scalar
            source_type: type of source
            amplitude: source strength
        
        Returns:
            f: source values at x
        """
        if source_type == 'gaussian':
            center = 0.50
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
                   a_range=(1.0, 2.0), b_range=(0.0, 0.0), 
                   source_type='zero', source_amplitude=0.0, T_max=1.0):
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
            T_max: maximum time for training
        """
        print("="*70)
        print("TRAINING PHYSICS-INFORMED DEEPONET FOR WAVE EQUATION")
        print("="*70)
        print(f"Epochs: {n_epochs}")
        print(f"Collocation points: {n_colloc}")
        print(f"Displacement IC amplitude range: {a_range}")
        print(f"Velocity IC amplitude range: {b_range}")
        print(f"Source type: {source_type}, amplitude: {source_amplitude}")
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
            
            # Generate ICs and source sensors
            u0_sensors = self.generate_ic_displacement(a)
            v0_sensors = self.generate_ic_velocity(b)
            src_sensors = self.generate_source(source_type, source_amplitude)
            
            # Sample collocation points (x, t)
            x_colloc = self.domain[0] + (self.domain[1] - self.domain[0]) * torch.rand(n_colloc)
            t_colloc = T_max * torch.rand(n_colloc)  # t ∈ [0, T_max]
            xt_colloc = torch.stack([x_colloc, t_colloc], dim=1)
            
            # Source values at collocation points
            src_colloc = self.source_function(x_colloc, source_type, source_amplitude)
            
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
                  source_amplitude=0.0, T_max=1.0, gt_data=None):
    """Visualize the trained solution"""
    
    # Generate test case
    u0_sensors = model.generate_ic_displacement(a_test)
    v0_sensors = model.generate_ic_velocity(b_test)
    src_sensors = model.generate_source(source_type, source_amplitude)
    
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
    src_plot = model.source_function(x_plot, source_type, source_amplitude).numpy()
    u0_plot = (a_test * torch.sin(np.pi * x_plot)).numpy()
    v0_plot = (b_test * torch.sin(np.pi * x_plot)).numpy()
    
    # Determine figure layout based on whether gt_data is provided
    if gt_data is not None:
        fig = plt.figure(figsize=(18, 10))
        nrows = 2
        
        # Reshape ground truth data to match prediction grid
        # gt_data format: [x, t, f, u] - reshape to (nt, nx) for u
        x_gt = gt_data[:, 0]
        t_gt = gt_data[:, 1]
        f_gt = gt_data[:, 2]
        u_gt = gt_data[:, 3]
        
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


if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("\n" + "="*70)
    print("PHYSICS-INFORMED DEEPONET FOR WAVE EQUATION")
    print("="*70)
    print("PDE: u_tt = c² * u_xx + f(x)")
    print("IC:  u(x, 0) = a * sin(π*x)")
    print("     u_t(x, 0) = b * sin(π*x)")
    print("BC:  u(0, t) = u(1, t) = 0")
    print("="*70 + "\n")
    
    # Create model
    model = PINNDeepONet_Wave(
        n_sensors_ic=20,
        n_sensors_src=20,
        branch_hidden=50,
        trunk_hidden=50,
        p=50,
        wave_speed=1.0
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Train (start with zero forcing and zero initial velocity)
    history = model.train_pinn(
        n_epochs=5000,
        n_colloc=200,
        lr=1e-3,
        a_range=(1.0, 2.0),     # Displacement IC
        b_range=(0.0, 0.0),     # Zero initial velocity
        source_type='zero',      # No forcing
        source_amplitude=0.0,
        T_max=2.0               # Longer time for wave propagation
    )
    
    # Save model
    torch.save(model.state_dict(), 'pinn_deeponet_wave.pth')
    print("✓ Model saved: pinn_deeponet_wave.pth\n")
    
    # Plot results
    print("Generating plots...")
    gt_data = np.loadtxt('gt_wave1D_2branch.csv', delimiter=',')
    
    fig1 = plot_solution(model, a_test=1.5, b_test=0.0, 
                         source_type='zero', source_amplitude=0.0, T_max=2.0, gt_data=gt_data)
    fig1.savefig('pinn_deeponet_wave_solution.png', dpi=150, bbox_inches='tight')
    print("✓ Solution plot saved: pinn_deeponet_wave_solution.png")
    
    fig2 = plot_training_history(history)
    fig2.savefig('pinn_deeponet_wave_training.png', dpi=150, bbox_inches='tight')
    print("✓ Training history saved: pinn_deeponet_wave_training.png")
    
    plt.show()
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
