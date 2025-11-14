import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

class BranchNet(nn.Module):
    """Branch network: encodes IC from sensor measurements"""
    def __init__(self, n_sensors, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_sensors, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, u_sensors):
        return self.net(u_sensors)

class TrunkNet(nn.Module):
    """Trunk network: encodes spatial locations"""
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class EDEDeepONet(nn.Module):
    """
    Complete Energy-Dissipative Evolutionary Deep Operator Neural Network
    Solves: u_t = u_xx, u(x,0) = a*sin(pi*x), u(0,t) = u(2,t) = 0
    """
    
    def __init__(self, n_sensors=20, branch_hidden=50, trunk_hidden=50, p=50):
        super().__init__()
        self.n_sensors = n_sensors
        self.p = p
        self.domain = [0, 2]
        
        # Networks
        self.branch = BranchNet(n_sensors, branch_hidden, p)
        self.trunk = TrunkNet(trunk_hidden, p)
        self.bias = nn.Parameter(torch.zeros(1))
        
        # Fixed sensor locations
        self.register_buffer('sensor_x', 
                            torch.linspace(self.domain[0], self.domain[1], n_sensors))
        
    def forward(self, u_sensors, x):
        """
        DeepONet forward: u(x) = sum_k b_k * g_k(x) + bias
        Args:
            u_sensors: (batch, n_sensors) or (n_sensors,)
            x: (n_points, 1) spatial locations
        Returns:
            u: (batch, n_points) or (n_points,)
        """
        # Handle single sample
        single_sample = u_sensors.dim() == 1
        if single_sample:
            u_sensors = u_sensors.unsqueeze(0)
        
        b = self.branch(u_sensors)  # (batch, p)
        g = self.trunk(x)  # (n_points, p)
        
        # Compute inner product: (batch, p) @ (p, n_points) = (batch, n_points)
        u = torch.matmul(b, g.T) + self.bias
        
        return u.squeeze(0) if single_sample else u
    
    def generate_ic(self, a):
        """Generate IC: u(x,0) = a*sin(pi*x) at sensor locations"""
        return a * torch.sin(np.pi * self.sensor_x)
    
    def exact_solution(self, x, t, a):
        """Exact solution: u(x,t) = a*sin(pi*x)*exp(-pi^2*t)"""
        # Handle both tensors and arrays for x, t, and a
        is_tensor = isinstance(x, torch.Tensor) or isinstance(t, torch.Tensor) or isinstance(a, torch.Tensor)
        
        if is_tensor:
            # Convert all to tensors if any is a tensor
            x_t = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
            t_t = t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32)
            a_t = a if isinstance(a, torch.Tensor) else torch.tensor(a, dtype=torch.float32)
            return a_t * torch.sin(np.pi * x_t) * torch.exp(-np.pi**2 * t_t)
        else:
            # All are arrays/scalars, use numpy
            return a * np.sin(np.pi * x) * np.exp(-np.pi**2 * t)
    
    def train_initial_operator(self, n_train=50, epochs=3000, lr=0.001):
        """
        Phase 1: Train to learn initial condition operator
        This learns to reconstruct IC from sensor measurements
        """
        print("="*60)
        print("PHASE 1: Training Initial Condition Operator")
        print("="*60)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=200, factor=0.5)
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            # Mini-batch of different IC parameters
            for _ in range(10):
                # Random a in [1, 2]
                a = 1.0 + torch.rand(1).item()
                
                # IC at sensors
                u_sensors = self.generate_ic(a)
                
                # Random query points (more points for better training)
                n_query = 100
                x_query = torch.rand(n_query, 1) * 2.0
                u_target = a * torch.sin(np.pi * x_query.squeeze())
                
                # Forward
                u_pred = self.forward(u_sensors, x_query)
                
                # Loss with regularization
                mse_loss = torch.mean((u_pred - u_target)**2)
                
                # Add small weight regularization to prevent overfitting
                reg_loss = 1e-5 * sum(torch.sum(p**2) for p in self.parameters())
                loss = mse_loss + reg_loss
                
                # Backprop
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += mse_loss.item()
            
            epoch_loss /= 10
            losses.append(epoch_loss)
            scheduler.step(epoch_loss)
            
            if epoch % 500 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.6f}, LR: {current_lr:.2e}")
        
        print("✓ Initial operator training complete!\n")
        return losses
    
    def compute_spatial_derivative(self, u, dx):
        """Compute spatial derivatives using finite differences"""
        # u: (n_points,)
        u_x = torch.gradient(u, spacing=(dx,))[0]
        u_xx = torch.gradient(u_x, spacing=(dx,))[0]
        return u_x, u_xx
    
    def compute_energy(self, u, dx):
        """Compute energy: E = 0.5 * integral(|u_x|^2 dx)"""
        u_x, _ = self.compute_spatial_derivative(u, dx)
        energy = 0.5 * torch.sum(u_x**2) * dx
        return energy
    
    def compute_Nx(self, u, dx):
        """Compute N_x(u) = -u_xx for heat equation"""
        _, u_xx = self.compute_spatial_derivative(u, dx)
        return -u_xx
    
    def compute_jacobians(self, u_sensors, x_grid):
        """
        Compute J1 and J2 for the least squares system (Equations 38-39)
        J1[i,j] = ∂u[i]/∂W_trunk[j]
        J2[i,j] = ∂u[i]/∂W_branch[j]
        
        Returns:
            J1: (n_points, n_trunk_params)
            J2: (n_points, n_branch_params)
        """
        n_points = len(x_grid)
        
        # Collect all parameters
        branch_params = list(self.branch.parameters())
        trunk_params = list(self.trunk.parameters())
        
        n_branch_params = sum(p.numel() for p in branch_params)
        n_trunk_params = sum(p.numel() for p in trunk_params)
        
        J1 = torch.zeros(n_points, n_trunk_params)
        J2 = torch.zeros(n_points, n_branch_params)
        
        # Compute full forward pass
        u_full = self.forward(u_sensors, x_grid)  # (n_points,)
        
        # Compute J1: ∂u/∂W_trunk
        for i in range(n_points):
            if u_full[i].requires_grad or any(p.requires_grad for p in trunk_params):
                grads = torch.autograd.grad(u_full[i], trunk_params, retain_graph=True, 
                                           create_graph=False, allow_unused=True)
                grad_vec = torch.cat([g.flatten() if g is not None else torch.zeros(p.numel()) 
                                     for g, p in zip(grads, trunk_params)])
                J1[i] = grad_vec
        
        # Compute J2: ∂u/∂W_branch  
        for i in range(n_points):
            if u_full[i].requires_grad or any(p.requires_grad for p in branch_params):
                grads = torch.autograd.grad(u_full[i], branch_params, retain_graph=True,
                                           create_graph=False, allow_unused=True)
                grad_vec = torch.cat([gr.flatten() if gr is not None else torch.zeros(p.numel()) 
                                     for gr, p in zip(grads, branch_params)])
                J2[i] = grad_vec
        
        return J1, J2
    
    def evolve_weights_sav(self, u_sensors, dt, n_spatial=51, restart_tol=2e-2):
        """
        Phase 2: Evolve weights using SAV method (Equations 36-37)
        
        Returns:
            gamma1, gamma2: Weight updates (∂W1/∂t, ∂W2/∂t)
            r_new: Updated SAV variable
            energy: Current energy
        """
        # Spatial grid
        x_grid = torch.linspace(self.domain[0], self.domain[1], n_spatial).unsqueeze(1)
        dx = (x_grid[1, 0] - x_grid[0, 0]).item()
        
        # Current solution
        with torch.no_grad():
            u_n = self.forward(u_sensors, x_grid)
        
        # Compute energy and N_x
        E_n = self.compute_energy(u_n, dx)
        N_x = self.compute_Nx(u_n, dx)
        
        # SAV update for r (Equation 31)
        norm_Nx_sq = torch.sum(N_x**2) * dx
        r_n = torch.sqrt(E_n + 1e-10)  # Add small epsilon for stability
        
        # More stable SAV update
        denominator = 1 + dt * norm_Nx_sq / (2 * (E_n + 1e-10))
        r_new = r_n / denominator
        
        # Check restart condition
        xi = r_new / r_n
        if torch.abs(1 - xi) > restart_tol:
            print(f"  Restart triggered: |1-ξ|={torch.abs(1-xi).item():.4f}")
            r_new = r_n
        
        # Modified RHS: (r^{n+1}/√E_n) * N_x
        rhs = (r_new / r_n) * N_x  # (n_spatial,)
        
        # Compute Jacobians J1, J2
        J1, J2 = self.compute_jacobians(u_sensors, x_grid)
        
        # Stack into single system
        J_stacked = torch.cat([J1, J2], dim=1)  # (n_spatial, n_trunk + n_branch)
        
        # Solve least squares with Tikhonov regularization
        # min ||J @ gamma - rhs||^2 + lambda * ||gamma||^2
        lambda_reg = 1e-4
        
        # Add regularization directly in lstsq
        n_params = J_stacked.shape[1]
        J_reg = torch.cat([J_stacked, np.sqrt(lambda_reg) * torch.eye(n_params)], dim=0)
        rhs_reg = torch.cat([rhs, torch.zeros(n_params)])
        
        try:
            gamma = torch.linalg.lstsq(J_reg, rhs_reg, rcond=1e-8).solution
        except RuntimeError:
            # Fallback: use normal equations with stronger regularization
            A = J_stacked.T @ J_stacked
            b_vec = J_stacked.T @ rhs
            lambda_reg_strong = 1e-2
            A_reg = A + lambda_reg_strong * torch.eye(A.shape[0])
            gamma = torch.linalg.solve(A_reg, b_vec)
        
        # Clip gradients for stability
        gamma_norm = torch.norm(gamma)
        max_norm = 10.0
        if gamma_norm > max_norm:
            gamma = gamma * (max_norm / gamma_norm)
        
        # Split into gamma1 (trunk) and gamma2 (branch)
        n_trunk_params = J1.shape[1]
        gamma1 = gamma[:n_trunk_params]
        gamma2 = gamma[n_trunk_params:]
        
        return gamma1, gamma2, r_new, E_n
    
    def update_weights(self, gamma1, gamma2, dt):
        """Update network weights using computed gradients"""
        # Update trunk parameters
        trunk_params = list(self.trunk.parameters())
        offset = 0
        for p in trunk_params:
            numel = p.numel()
            p.data += dt * gamma1[offset:offset+numel].reshape(p.shape)
            offset += numel
        
        # Update branch parameters
        branch_params = list(self.branch.parameters())
        offset = 0
        for p in branch_params:
            numel = p.numel()
            p.data += dt * gamma2[offset:offset+numel].reshape(p.shape)
            offset += numel
    
    def evolve_solution(self, a, T_final, dt=2.5e-4, n_spatial=51):
        """
        Complete time evolution from t=0 to T_final
        
        Args:
            a: IC parameter (amplitude)
            T_final: Final time
            dt: Time step
            n_spatial: Number of spatial points
        
        Returns:
            solution_history: List of (t, u(x,t)) tuples
            energy_history: List of (t, E(t), r(t)) tuples
        """
        print("="*60)
        print(f"PHASE 2: Time Evolution for a={a}")
        print(f"T_final={T_final}, dt={dt}, n_steps={int(T_final/dt)}")
        print("="*60)
        
        # IC at sensors
        u_sensors = self.generate_ic(a)
        
        # Spatial grid for evaluation
        x_grid = torch.linspace(self.domain[0], self.domain[1], n_spatial).unsqueeze(1)
        dx = (x_grid[1, 0] - x_grid[0, 0]).item()
        
        # Initialize
        t = 0
        n_steps = int(T_final / dt)
        
        solution_history = []
        energy_history = []
        
        # Initial state
        with torch.no_grad():
            u_init = self.forward(u_sensors, x_grid)
            E_init = self.compute_energy(u_init, dx)
            r = torch.sqrt(E_init)
        
        solution_history.append((t, x_grid.squeeze().numpy(), u_init.numpy()))
        energy_history.append((t, E_init.item(), r.item()))
        
        print(f"t={t:.4f}, E={E_init.item():.6f}, r={r.item():.6f}")
        
        # Time stepping
        save_every = max(1, n_steps // 20)  # Save ~20 snapshots
        E_prev = E_init.item()
        
        for step in range(n_steps):
            # Evolve weights using SAV
            gamma1, gamma2, r_new, E_n = self.evolve_weights_sav(u_sensors, dt, n_spatial)
            
            # Check for numerical issues
            if torch.isnan(gamma1).any() or torch.isnan(gamma2).any():
                print(f"WARNING: NaN detected in gradients at step {step}")
                break
            
            # Update weights
            self.update_weights(gamma1, gamma2, dt)
            
            # Update time and SAV variable
            t += dt
            r = r_new
            
            # Check energy dissipation
            energy_change = E_n.item() - E_prev
            if energy_change > 1e-10:  # Energy should not increase significantly
                print(f"  WARNING: Energy increased by {energy_change:.2e} at t={t:.4f}")
            E_prev = E_n.item()
            
            # Save solution periodically
            if step % save_every == 0 or step == n_steps - 1:
                with torch.no_grad():
                    u_current = self.forward(u_sensors, x_grid)
                    
                    # Compute error vs exact solution
                    u_exact = self.exact_solution(x_grid.squeeze(), t, a)
                    rel_error = torch.norm(u_current - u_exact) / torch.norm(u_exact)
                    
                solution_history.append((t, x_grid.squeeze().numpy(), u_current.numpy()))
                energy_history.append((t, E_n.item(), r.item()))
                print(f"t={t:.4f}, E={E_n.item():.6f}, r={r.item():.6f}, "
                      f"|γ1|={torch.norm(gamma1).item():.2e}, |γ2|={torch.norm(gamma2).item():.2e}, "
                      f"rel_err={rel_error.item():.2e}")
        
        print("✓ Time evolution complete!\n")
        return solution_history, energy_history


def plot_results(model, a_test, solution_history, energy_history):
    """Visualize results"""
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Solution evolution
    ax1 = plt.subplot(2, 3, 1)
    for t, x, u in solution_history[::2]:  # Plot every other snapshot
        u_exact = model.exact_solution(x, t, a_test).numpy()
        ax1.plot(x, u, '-', alpha=0.7, label=f't={t:.3f}')
        ax1.plot(x, u_exact, 'o', markersize=3, alpha=0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('u(x,t)')
    ax1.set_title(f'Solution Evolution (a={a_test})')
    ax1.legend(fontsize=8)
    ax1.grid(True)
    
    # Plot 2: Energy evolution
    ax2 = plt.subplot(2, 3, 2)
    times = [h[0] for h in energy_history]
    energies = [h[1] for h in energy_history]
    rs = [h[2] for h in energy_history]
    ax2.plot(times, energies, 'b-', label='E(t)')
    ax2.plot(times, [r**2 for r in rs], 'r--', label='r²(t)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Energy')
    ax2.set_title('Energy Dissipation (SAV)')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Error vs exact solution
    ax3 = plt.subplot(2, 3, 3)
    errors = []
    for t, x, u in solution_history:
        u_exact = model.exact_solution(x, t, a_test).numpy()
        error = np.mean((u - u_exact)**2)
        errors.append(error)
    ax3.plot([h[0] for h in solution_history], errors, 'g-')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('MSE')
    ax3.set_title('Error vs Exact Solution')
    ax3.set_yscale('log')
    ax3.grid(True)
    
    # Plot 4-6: Comparison at specific times
    comparison_indices = [0, len(solution_history)//2, -1]
    for idx, comp_idx in enumerate(comparison_indices):
        ax = plt.subplot(2, 3, 4+idx)
        t, x, u = solution_history[comp_idx]
        u_exact = model.exact_solution(x, t, a_test).numpy()
        ax.plot(x, u, 'b-', linewidth=2, label='EDE-DeepONet')
        ax.plot(x, u_exact, 'ro', markersize=4, label='Exact')
        ax.set_xlabel('x')
        ax.set_ylabel('u(x,t)')
        ax.set_title(f't = {t:.4f}')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create model
    print("Creating EDE-DeepONet model...")
    model = EDEDeepONet(n_sensors=20, branch_hidden=50, trunk_hidden=50, p=50)
    
    # Phase 1: Train initial condition operator (increased epochs for better convergence)
    losses = model.train_initial_operator(n_train=50, epochs=5000, lr=0.001)
    
    # Phase 2: Evolve solution for specific IC
    a_test = 1.5  # Test with a=1.5
    T_final = 0.01  # Start with shorter time to test
    dt = 1e-4  # Smaller time step for better stability
    
    solution_history, energy_history = model.evolve_solution(
        a=a_test, 
        T_final=T_final, 
        dt=dt, 
        n_spatial=51
    )
    
    # Visualize
    fig = plot_results(model, a_test, solution_history, energy_history)
    plt.savefig('ede_deeponet_complete.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("="*60)
    print("SUMMARY: How IC is Enforced and Evolved")
    print("="*60)
    print("1. TRAINING: Learn IC operator G: u → f")
    print("   - Input: IC values at sensors [f(x₁),...,f(xₘ)]")
    print("   - Output: IC at any point f(y)")
    print("   - Result: Weights W₁⁰, W₂⁰ encode the IC")
    print()
    print("2. EVOLUTION: Solve heat equation u_t = u_xx")
    print("   - Given: NEW IC via sensor measurements")
    print("   - Compute: ∂W₁/∂t, ∂W₂/∂t from Eqs 36-37")
    print("   - Update: W₁ⁿ⁺¹ = W₁ⁿ + dt·γ₁")
    print("   - Update: W₂ⁿ⁺¹ = W₂ⁿ + dt·γ₂")
    print("   - Result: u(x,t) = Σ bₖ(W₁ⁿ) · gₖ(x; W₂ⁿ)")
    print("="*60)