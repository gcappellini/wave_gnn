import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

class EDEDeepONet:
    """
    Energy-Dissipative Evolutionary Deep Operator Neural Network
    for solving: u_t = u_xx, u(x,0) = a*sin(pi*x), u(0,t) = u(2,t) = 0
    """
    
    def __init__(self, n_sensors=20, branch_hidden=40, trunk_hidden=40, p=40):
        """
        Args:
            n_sensors: Number of sensor points for Branch net
            branch_hidden: Hidden layer size in Branch net
            trunk_hidden: Hidden layer size in Trunk net
            p: Output dimension (number of basis functions)
        """
        self.n_sensors = n_sensors
        self.branch_hidden = branch_hidden
        self.trunk_hidden = trunk_hidden
        self.p = p
        self.domain = [0, 2]
        
        # Initialize sensor locations (fixed)
        self.sensor_x = np.linspace(self.domain[0], self.domain[1], n_sensors)
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize neural network weights using He initialization"""
        # Branch net: sensors -> hidden -> p outputs
        self.W1_branch = np.random.randn(self.branch_hidden, self.n_sensors) * np.sqrt(2.0/self.n_sensors)
        self.b1_branch = np.zeros(self.branch_hidden)
        self.W2_branch = np.random.randn(self.p, self.branch_hidden) * np.sqrt(2.0/self.branch_hidden)
        self.b2_branch = np.zeros(self.p)
        
        # Trunk net: 1D (x) -> hidden -> p outputs
        self.W1_trunk = np.random.randn(self.trunk_hidden, 1) * np.sqrt(2.0/1)
        self.b1_trunk = np.zeros(self.trunk_hidden)
        self.W2_trunk = np.random.randn(self.p, self.trunk_hidden) * np.sqrt(2.0/self.trunk_hidden)
        self.b2_trunk = np.zeros(self.p)
        
        # Bias term
        self.bias = 0.0
        
    def activation(self, x):
        """Tanh activation"""
        return np.tanh(x)
    
    def activation_derivative(self, x):
        """Derivative of tanh"""
        return 1 - np.tanh(x)**2
    
    def branch_forward(self, u_sensors):
        """
        Forward pass through Branch net
        Args:
            u_sensors: (n_sensors,) array of IC values at sensor locations
        Returns:
            b: (p,) array - Branch net output
        """
        # Layer 1
        z1 = self.W1_branch @ u_sensors + self.b1_branch
        h1 = self.activation(z1)
        
        # Layer 2
        z2 = self.W2_branch @ h1 + self.b2_branch
        b = z2  # No activation on output layer
        
        return b
    
    def trunk_forward(self, x):
        """
        Forward pass through Trunk net
        Args:
            x: scalar or (n_points,) array of spatial locations
        Returns:
            g: (p, n_points) array - Trunk net outputs
        """
        x = np.atleast_1d(x).reshape(-1, 1)  # (n_points, 1)
        
        # Layer 1
        z1 = x @ self.W1_trunk.T + self.b1_trunk  # (n_points, trunk_hidden)
        h1 = self.activation(z1)
        
        # Layer 2
        z2 = h1 @ self.W2_trunk.T + self.b2_trunk  # (n_points, p)
        g = z2.T  # (p, n_points)
        
        return g
    
    def forward(self, u_sensors, x):
        """
        Full DeepONet forward pass: u(x) ≈ sum_k b_k * g_k(x) + bias
        Args:
            u_sensors: (n_sensors,) IC values at sensors
            x: scalar or array of query points
        Returns:
            u: predicted solution at x
        """
        b = self.branch_forward(u_sensors)  # (p,)
        g = self.trunk_forward(x)  # (p, n_points)
        
        # Compute inner product
        u = b @ g + self.bias  # (n_points,)
        
        return u
    
    def generate_ic(self, a):
        """
        Generate initial condition: u(x,0) = a*sin(pi*x)
        Args:
            a: amplitude parameter
        Returns:
            u_sensors: IC values at sensor locations
        """
        return a * np.sin(np.pi * self.sensor_x)
    
    def exact_solution(self, x, t, a):
        """
        Exact solution: u(x,t) = a*sin(pi*x)*exp(-pi^2*t)
        """
        return a * np.sin(np.pi * x) * np.exp(-np.pi**2 * t)
    
    def train_initial_operator(self, n_train=50, epochs=2000, lr=0.001):
        """
        Train DeepONet to learn initial condition operator G: u -> f
        This learns to reconstruct IC from sensor measurements
        """
        print("Training initial condition operator...")
        
        # Generate training data: different values of a in [1, 2]
        a_train = np.random.uniform(1.0, 2.0, n_train)
        
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for a in a_train:
                # IC at sensors
                u_sensors = self.generate_ic(a)
                
                # Random query points
                n_query = 30
                x_query = np.random.uniform(self.domain[0], self.domain[1], n_query)
                u_target = a * np.sin(np.pi * x_query)
                
                # Forward pass
                u_pred = self.forward(u_sensors, x_query)
                
                # Loss
                loss = np.mean((u_pred - u_target)**2)
                epoch_loss += loss
                
                # Backprop (simplified - using finite differences)
                eps = 1e-7
                
                # Gradient w.r.t. bias
                grad_bias = 2 * np.mean(u_pred - u_target)
                self.bias -= lr * grad_bias
                
                # Gradient w.r.t. Branch net output layer
                b = self.branch_forward(u_sensors)
                g = self.trunk_forward(x_query)
                error = u_pred - u_target
                
                for i in range(self.p):
                    grad_b2 = 2 * np.mean(error * g[i, :])
                    self.b2_branch[i] -= lr * grad_b2
                    
                    grad_g2 = 2 * np.mean(error * b[i])
                    self.b2_trunk[i] -= lr * grad_g2
            
            epoch_loss /= n_train
            losses.append(epoch_loss)
            
            if epoch % 200 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.6f}")
        
        print("Training complete!")
        return losses
    
    def compute_energy(self, u, dx):
        """
        Compute energy: E = 0.5 * integral(|u_x|^2 dx)
        Using finite differences for derivative
        """
        u_x = np.gradient(u, dx)
        energy = 0.5 * np.sum(u_x**2) * dx
        return energy
    
    def compute_Nx(self, u, dx):
        """
        Compute N_x(u) = -u_xx (for heat equation)
        Using finite differences
        """
        u_xx = np.gradient(np.gradient(u, dx), dx)
        return -u_xx
    
    def evolve_step_sav(self, a, dt, n_x=51):
        """
        Single evolution step using SAV method
        This is the core EDE-DeepONet evolution
        
        Args:
            a: initial condition parameter
            dt: time step
            n_x: number of spatial points
        """
        # Spatial grid
        x = np.linspace(self.domain[0], self.domain[1], n_x)
        dx = x[1] - x[0]
        
        # Get initial condition at sensors
        u_sensors = self.generate_ic(a)
        
        # Evaluate current solution at all spatial points
        u_n = self.forward(u_sensors, x)
        
        # Compute energy and N_x
        E_n = self.compute_energy(u_n, dx)
        N_x = self.compute_Nx(u_n, dx)
        
        # SAV: compute r^{n+1}
        norm_Nx_sq = np.sum(N_x**2) * dx
        r_n = np.sqrt(E_n)
        r_np1 = r_n / (1 + dt * norm_Nx_sq / (2 * E_n))
        
        # Modified right-hand side
        rhs = (r_np1 / np.sqrt(E_n)) * N_x
        
        # Solve least squares for dW/dt using Equation (36)-(37)
        # J1 = ∂g_k/∂W1 * b_k, J2 = g_k * ∂b_k/∂W2
        # This is simplified - full implementation needs automatic differentiation
        
        # For now, use a simple update (this is a simplified version)
        # In full implementation, you'd solve the linear system from Eq 36-37
        
        print(f"Energy: {E_n:.6f}, r: {r_n:.6f}, r_new: {r_np1:.6f}")
        
        return u_n, E_n, r_np1

# Example usage
if __name__ == "__main__":
    # Create model
    model = EDEDeepONet(n_sensors=20, branch_hidden=40, trunk_hidden=40, p=40)
    
    # Train initial operator
    losses = model.train_initial_operator(n_train=50, epochs=1000, lr=0.001)
    
    # Plot training loss
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.yscale('log')
    plt.grid(True)
    
    # Test on different a values
    plt.subplot(1, 2, 2)
    x_test = np.linspace(0, 2, 100)
    
    for a in [1.0, 1.5, 2.0, 2.5]:  # Note: 2.5 is outside training range [1,2]
        u_sensors = model.generate_ic(a)
        u_pred = model.forward(u_sensors, x_test)
        u_exact = model.exact_solution(x_test, 0, a)
        
        plt.plot(x_test, u_pred, '-', label=f'Pred a={a}')
        plt.plot(x_test, u_exact, 'o', markevery=10, label=f'Exact a={a}')
    
    plt.xlabel('x')
    plt.ylabel('u(x, 0)')
    plt.title('Initial Condition Reconstruction')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ede_deeponet_results.png', dpi=150)
    plt.show()
    
    print("\n" + "="*60)
    print("KEY POINTS ABOUT IC ENFORCEMENT:")
    print("="*60)
    print("1. Branch net gets: IC values at sensors [u(x1), u(x2), ..., u(xm)]")
    print("   - For heat equation: [a*sin(πx1), a*sin(πx2), ..., a*sin(πxm)]")
    print("   - This encodes 'which IC' (the parameter 'a')")
    print()
    print("2. Trunk net gets: spatial query point 'y'")
    print("   - This encodes 'where to evaluate'")
    print()
    print("3. Output: u(y,0) = sum_k b_k(IC_sensors) * g_k(y)")
    print("   - Branch extracts features from IC")
    print("   - Trunk evaluates spatially")
    print("   - Together: reconstruct IC at any point y")
    print()
    print("4. Evolution: Both W1 and W2 become functions of time")
    print("   - At t=0: weights encode the IC operator")
    print("   - At t>0: weights evolve via Equations 36-37")
    print("="*60)