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


class PINNDeepONet(nn.Module):
    """
    Physics-Informed DeepONet for heat equation with source:
    u_t = u_xx + f(x)
    
    Two-branch architecture:
    - Branch_IC: encodes initial condition u(x, 0)
    - Branch_Source: encodes heat source f(x)
    """
    
    def __init__(self, n_sensors_ic=20, n_sensors_src=20, 
                 branch_hidden=50, trunk_hidden=50, p=50):
        super().__init__()
        
        self.n_sensors_ic = n_sensors_ic
        self.n_sensors_src = n_sensors_src
        self.p = p
        self.domain = [0.0, 1.0]  # Spatial domain
        
        # Two branch networks
        self.branch_ic = BranchNet(n_sensors_ic, branch_hidden, p)
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
    
    def forward(self, ic_sensors, src_sensors, xt):
        """
        Args:
            ic_sensors: (batch, n_sensors_ic) or (n_sensors_ic,)
            src_sensors: (batch, n_sensors_src) or (n_sensors_src,)
            xt: (n_points, 2) spatiotemporal coordinates [x, t]
        
        Returns:
            u: (batch, n_points) or (n_points,)
        """
        # Handle single sample
        single_sample = ic_sensors.dim() == 1
        if single_sample:
            ic_sensors = ic_sensors.unsqueeze(0)
            src_sensors = src_sensors.unsqueeze(0)
        
        # Encode IC and source
        b_ic = self.branch_ic(ic_sensors)       # (batch, p)
        b_src = self.branch_source(src_sensors) # (batch, p)
        
        # Combine branch outputs
        b_combined = b_ic + b_src               # (batch, p)
        
        # Encode spatiotemporal locations
        tau = self.trunk(xt)                    # (n_points, p)
        
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
    
    def compute_pde_residual(self, ic_sensors, src_sensors, xt, src_values):
        """
        Compute PDE residual: R = u_t - u_xx - f(x)
        
        Args:
            ic_sensors: (n_sensors_ic,) IC measurements
            src_sensors: (n_sensors_src,) source measurements
            xt: (n_points, 2) collocation points [x, t]
            src_values: (n_points,) source values f(x) at collocation points
        
        Returns:
            residual: (n_points,)
        """
        xt_requires_grad = xt.clone().requires_grad_(True)
        
        # Forward pass
        u = self.forward(ic_sensors, src_sensors, xt_requires_grad)
        
        # Compute u_t using autograd
        u_t = torch.autograd.grad(u, xt_requires_grad, 
                                  torch.ones_like(u),
                                  create_graph=True)[0][:, 1]  # ∂u/∂t
        
        # Compute u_x
        u_x = torch.autograd.grad(u, xt_requires_grad,
                                  torch.ones_like(u),
                                  create_graph=True)[0][:, 0]  # ∂u/∂x
        
        # Compute u_xx
        u_xx = torch.autograd.grad(u_x, xt_requires_grad,
                                   torch.ones_like(u_x),
                                   create_graph=True)[0][:, 0]  # ∂²u/∂x²
        
        # PDE residual: u_t - u_xx - f(x)
        residual = u_t - u_xx - src_values
        
        return residual
    
    def generate_ic(self, a):
        """
        Generate IC: u(x, 0) = a * sin(pi * x)
        
        Args:
            a: amplitude parameter
        
        Returns:
            ic_sensors: IC values at sensor locations
        """
        return a * torch.sin(np.pi * self.sensor_x_ic)
    
    def generate_source(self, source_type='gaussian', amplitude=0.5):
        """
        Generate heat source f(x)
        
        Args:
            source_type: 'gaussian', 'sine', 'constant'
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
        else:
            raise ValueError(f"Unknown source type: {source_type}")
        
        return f
    
    def train_pinn(self, n_epochs=5000, n_colloc=200, lr=1e-3, 
                   a_range=(1.0, 2.0), source_type='gaussian', source_amplitude=0.5):
        """
        Train PINN-DeepONet with physics-informed loss
        
        Loss = w_pde * L_pde + w_ic * L_ic
        Note: BC is enforced as hard constraint, no soft BC loss needed
        """
        print("="*70)
        print("TRAINING PHYSICS-INFORMED DEEPONET")
        print("="*70)
        print(f"Epochs: {n_epochs}")
        print(f"Collocation points: {n_colloc}")
        print(f"IC amplitude range: {a_range}")
        print(f"Source type: {source_type}, amplitude: {source_amplitude}")
        print("Boundary conditions: HARD constraints (no soft loss)")
        print("="*70)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=500, verbose=True
        )
        
        # Loss weights (no BC weight needed - hard constraint!)
        w_pde = 1.0
        w_ic = 10.0
        
        history = {
            'total': [], 'pde': [], 'ic': []
        }
        
        for epoch in range(n_epochs):
            # Sample random IC amplitude
            a = a_range[0] + (a_range[1] - a_range[0]) * torch.rand(1).item()
            
            # Generate IC and source sensors
            ic_sensors = self.generate_ic(a)
            src_sensors = self.generate_source(source_type, source_amplitude)
            
            # Sample collocation points (x, t)
            x_colloc = self.domain[0] + (self.domain[1] - self.domain[0]) * torch.rand(n_colloc)
            t_colloc = 0.1 * torch.rand(n_colloc)  # t ∈ [0, 0.1]
            xt_colloc = torch.stack([x_colloc, t_colloc], dim=1)
            
            # Source values at collocation points
            src_colloc = self.source_function(x_colloc, source_type, source_amplitude)
            
            # === PDE Loss ===
            residual = self.compute_pde_residual(ic_sensors, src_sensors, xt_colloc, src_colloc)
            loss_pde = torch.mean(residual ** 2)
            
            # === IC Loss ===
            n_ic = 50
            x_ic = torch.linspace(self.domain[0], self.domain[1], n_ic)
            t_ic = torch.zeros(n_ic)
            xt_ic = torch.stack([x_ic, t_ic], dim=1)
            
            u_ic_pred = self.forward(ic_sensors, src_sensors, xt_ic)
            u_ic_true = a * torch.sin(np.pi * x_ic)
            loss_ic = torch.mean((u_ic_pred - u_ic_true) ** 2)
            
            # === Total Loss (no BC loss - hard constraint!) ===
            loss_total = w_pde * loss_pde + w_ic * loss_ic
            
            # Optimization step
            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Record history
            history['total'].append(loss_total.item())
            history['pde'].append(loss_pde.item())
            history['ic'].append(loss_ic.item())
            
            scheduler.step(loss_total)
            
            # Print progress
            if epoch % 500 == 0 or epoch == n_epochs - 1:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:5d} | Loss: {loss_total.item():.6f} | "
                      f"PDE: {loss_pde.item():.6f} | IC: {loss_ic.item():.6f} | "
                      f"LR: {current_lr:.2e}")
        
        print("\n✓ Training complete!\n")
        return history


def plot_solution(model, a_test=1.5, source_type='gaussian', source_amplitude=0.5):
    """Visualize the trained solution"""
    
    # Generate test case
    ic_sensors = model.generate_ic(a_test)
    src_sensors = model.generate_source(source_type, source_amplitude)
    
    # Create spatiotemporal grid
    nx, nt = 100, 50
    x_plot = torch.linspace(model.domain[0], model.domain[1], nx)
    t_plot = torch.linspace(0, 0.1, nt)
    
    X, T = torch.meshgrid(x_plot, t_plot, indexing='ij')
    xt_grid = torch.stack([X.flatten(), T.flatten()], dim=1)
    
    # Predict solution
    with torch.no_grad():
        u_pred = model.forward(ic_sensors, src_sensors, xt_grid)
        u_pred = u_pred.reshape(nx, nt).numpy()
    
    # Source function for plotting
    src_plot = model.source_function(x_plot, source_type, source_amplitude).numpy()
    
    # Plotting
    fig = plt.figure(figsize=(18, 5))
    
    # Plot 1: Solution evolution (snapshots)
    ax1 = plt.subplot(1, 3, 1)
    time_indices = [0, nt//4, nt//2, 3*nt//4, -1]
    for idx in time_indices:
        t_val = t_plot[idx].item()
        ax1.plot(x_plot.numpy(), u_pred[:, idx], label=f't={t_val:.3f}')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('u(x,t)', fontsize=12)
    ax1.set_title(f'Solution Evolution (a={a_test})', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Spatiotemporal heatmap
    ax2 = plt.subplot(1, 3, 2)
    im = ax2.contourf(T.numpy(), X.numpy(), u_pred, levels=20, cmap='RdBu_r')
    ax2.set_xlabel('t', fontsize=12)
    ax2.set_ylabel('x', fontsize=12)
    ax2.set_title('Solution u(x,t)', fontsize=12)
    plt.colorbar(im, ax=ax2)
    
    # Plot 3: Source function
    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(x_plot.numpy(), src_plot, 'r-', linewidth=2, label='f(x)')
    ic_plot = (a_test * torch.sin(np.pi * x_plot)).numpy()
    ax3.plot(x_plot.numpy(), ic_plot, 'b--', linewidth=2, label='IC: u(x,0)')
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('Value', fontsize=12)
    ax3.set_title('Heat Source & Initial Condition', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_training_history(history):
    """Plot training loss history"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].semilogy(history['total'], 'k-', linewidth=1.5)
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].semilogy(history['pde'], 'b-', linewidth=1.5)
    axes[1].set_title('PDE Residual Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].semilogy(history['ic'], 'r-', linewidth=1.5)
    axes[2].set_title('IC Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    s_ampl = 5.0
    
    print("\n" + "="*70)
    print("PHYSICS-INFORMED DEEPONET FOR HEAT EQUATION WITH SOURCE")
    print("="*70)
    print("PDE: u_t = u_xx + f(x)")
    print("IC:  u(x, 0) = a * sin(π*x)")
    print("BC:  u(0, t) = u(1, t) = 0")
    print("="*70 + "\n")
    
    # Create model
    model = PINNDeepONet(
        n_sensors_ic=20,
        n_sensors_src=20,
        branch_hidden=50,
        trunk_hidden=50,
        p=50
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Train
    history = model.train_pinn(
        n_epochs=5000,
        n_colloc=200,
        lr=1e-3,
        a_range=(1.0, 2.0),
        source_type='gaussian',
        source_amplitude=s_ampl
    )
    
    # Save model
    torch.save(model.state_dict(), 'pinn_deeponet_heat.pth')
    print("✓ Model saved: pinn_deeponet_heat.pth\n")
    
    # Plot results
    print("Generating plots...")
    
    fig1 = plot_solution(model, a_test=1.5, source_type='gaussian', source_amplitude=s_ampl)
    fig1.savefig('pinn_deeponet_solution.png', dpi=150, bbox_inches='tight')
    print("✓ Solution plot saved: pinn_deeponet_solution.png")
    
    fig2 = plot_training_history(history)
    fig2.savefig('pinn_deeponet_training.png', dpi=150, bbox_inches='tight')
    print("✓ Training history saved: pinn_deeponet_training.png")
    
    plt.show()
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
