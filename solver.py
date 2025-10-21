import numpy as np
import sys
from scipy.io import savemat
from datetime import datetime
from plot import plot_features_2d
from dataset import build_laplacian_matrix, WaveGNN1D

def stringforce(x, t, t_f=10, f_min=-3, x_f_1=0.2, x_f_2=0.8):
    """
    Time-varying force on the string - Gaussian pulses in space and time.

    Parameters
    ----------
    x : float or np.ndarray
        Spatial coordinate(s).
    t : float
        Time value.
    t_f : float
        Final time constant.
    f_min : float
        Force amplitude.
    x_f_1 : float
        First force center position.
    x_f_2 : float
        Second force center position.

    Returns
    -------
    f : np.ndarray
        Computed force field values.
    """
    t_1 = 0.2 * t_f
    t_2 = 0.8 * t_f
    h = f_min

    z1 = h * np.exp(-400 * (x - x_f_1)**2) * np.exp(-((t - t_1)**2) / (2 * 0.5**2))
    z2 = h * np.exp(-400 * (x - x_f_2)**2) * np.exp(-((t - t_2)**2) / (2 * 0.5**2))

    f = z1 + z2
    return f


def simulate_wave(gnn, u0, v0, f0, x, T, t_f):
    """
    Simulate wave equation using GNN with integrated time stepping.
    
    Args:
        gnn: WaveGNN1D instance (contains dt)
        u0: Initial positions, shape (N,)
        v0: Initial velocities, shape (N,)
        f0: Initial forces, shape (N,)
        x: Spatial coordinates
        T: Total simulation time
        t_f: Final time for force function
        
    Returns:
        t_history: Time points
        u_history: Position history, shape (num_steps, N)
        v_history: Velocity history, shape (num_steps, N)
        f_history: Force history, shape (num_steps, N)
    """
    # Initialize node features
    features = np.stack([u0, v0, f0], axis=1)  # Shape (N, 3)
    t = 0.0
    
    # Storage
    num_steps = int(T / gnn.dt) + 1
    t_history = np.zeros(num_steps)
    u_history = np.zeros((num_steps, gnn.N))
    v_history = np.zeros((num_steps, gnn.N))
    f_history = np.zeros((num_steps, gnn.N))
    
    # Store initial condition
    t_history[0] = t
    u_history[0] = features[:, 0]
    v_history[0] = features[:, 1]
    f_history[0] = stringforce(x, t, t_f)
    
    # Time integration loop
    for step in range(1, num_steps):
        # Single GNN call advances time by dt
        features = gnn.forward(features)
        
        # Update time
        t += gnn.dt
        t_history[step] = t
        
        # Compute new force based on current time
        f_history[step] = stringforce(x, t, t_f)
        
        # Update features with new force
        features[:, 2] = f_history[step]
        
        # Store results
        u_history[step] = features[:, 0]
        v_history[step] = features[:, 1]
    
    return t_history, u_history, v_history, f_history


# Redirect stdout to capture terminal output
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

if __name__ == "__main__":

    # Start logging
    logger = Logger("./figures/simulation_output_1d.txt")
    sys.stdout = logger

    # ============================================================================
    # Setup and Simulation
    # ============================================================================

    print("="*60)
    print("1D WAVE EQUATION GNN - STRING WITH FORCING")
    print("="*60)
    print("\nArchitecture:")
    print("  - Node features: [u, v, f] (position, velocity, force)")
    print("  - Message passing: Computes spatial Laplacian")
    print("  - Node update: a = c²·Laplacian(u) - k·v + f")
    print("  - Time integration: RK4 method")
    print("  - One forward pass: t -> t+dt")
    print()

    # Parameters
    N = 100             # Number of nodes
    c = 1.0             # Wave speed
    k = 1.0             # Damping coefficient
    T = 10.0            # Total time
    dt = 0.01          # Time step

    # Spatial discretization
    dx = 1.0 / (N - 1)
    x = np.linspace(0, 1.0, N)  # Node positions

    # Initial conditions
    u0 = np.zeros(N)        # Position: all zeros
    v0 = np.zeros(N)        # Velocity: all zeros
    f0 = np.zeros(N)        # Initial force

    print("Parameters:")
    print(f"  N = {N} nodes")
    print(f"  c = {c} (wave speed)")
    print(f"  k = {k} (damping coefficient)")
    print(f"  T = {T} s (total time)")
    print(f"  dt = {dt} s (time step)")
    print(f"  dx = {dx:.6f} (spatial step)")
    print()

    # Create GNN
    L = build_laplacian_matrix(N, dx)
    gnn = WaveGNN1D(L, c, k, dt)

    # Run simulation
    print("Running simulation...")
    start_time = datetime.now()
    t_history, u_history, v_history, f_history = simulate_wave(gnn, u0, v0, f0, x, T, T)
    end_time = datetime.now()
    print(f"Simulation complete: {len(t_history)} time steps in {end_time - start_time}")
    print(f"Each step = one GNN forward pass")
    print()

    # Save data in MATLAB format
    matlab_times = np.linspace(0, 10, num=100)
    indices = [np.argmin(np.abs(t_history - t)) for t in matlab_times]
    pinn_data = u_history[indices]
    savemat('./matlab/data_gn.mat', {
        'pinn_data': pinn_data,
        'x': x,
        't': matlab_times
    })
    print("Saved: data_gn.mat")
    print()

    histories = np.array([u_history, v_history, f_history])

    plot_features_2d(
    x,  
    histories,
    output_file='./figures/gn_string_sol.png'
    )


    # ============================================================================
    # Visualization: Animation
    # ============================================================================

    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # # Position plot
    # line_u, = ax1.plot([], [], 'b-o', label='Position u(x,t)', markersize=3, linewidth=2)
    # ax1.set_xlim(0, 1)
    # ax1.set_ylim(-1.5, 1.5)
    # ax1.set_xlabel('Position x')
    # ax1.set_ylabel('Displacement u(x,t)')
    # ax1.set_title('1D Wave Equation: Position')
    # ax1.legend()
    # ax1.grid(True, alpha=0.3)

    # # Velocity plot
    # line_v, = ax2.plot([], [], 'g-o', label='Velocity v(x,t)', markersize=3, linewidth=2)
    # ax2.set_xlim(0, 1)
    # ax2.set_ylim(-5, 5)
    # ax2.set_xlabel('Position x')
    # ax2.set_ylabel('Velocity v(x,t)')
    # ax2.set_title('1D Wave Equation: Velocity')
    # ax2.legend()
    # ax2.grid(True, alpha=0.3)

    # # Force plot
    # line_f, = ax3.plot([], [], 'r-o', label='Force f(x,t)', markersize=3, linewidth=2)
    # ax3.set_xlim(0, 1)
    # ax3.set_ylim(-4, 1)
    # ax3.set_xlabel('Position x')
    # ax3.set_ylabel('Force f(x,t)')
    # ax3.set_title('External Force')
    # ax3.legend()
    # ax3.grid(True, alpha=0.3)

    # time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, 
    #                     verticalalignment='top', fontsize=12,
    #                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # def init():
    #     line_u.set_data([], [])
    #     line_v.set_data([], [])
    #     line_f.set_data([], [])
    #     time_text.set_text('')
    #     return line_u, line_v, line_f, time_text

    # def animate(frame):
    #     t = t_history[frame]
        
    #     # Update plots
    #     line_u.set_data(x, u_history[frame])
    #     line_v.set_data(x, v_history[frame])
    #     line_f.set_data(x, f_history[frame])
    #     time_text.set_text(f't = {t:.3f} s')
        
    #     return line_u, line_v, line_f, time_text

    # # Create animation
    # print("Creating animation...")
    # skip = 10
    # frames = range(0, len(t_history), skip)
    # anim = FuncAnimation(fig, animate, init_func=init, frames=frames,
    #                     interval=50, blit=True, repeat=True)

    # plt.tight_layout()
    # print("Saving animation as wave_1d.gif...")
    # anim.save('./figures/wave_1d.gif', writer='pillow', fps=20, dpi=100)
    # print("Saved: wave_1d.gif")
    # print()

    # plt.close()

    # ============================================================================
    # Energy evolution
    # ============================================================================

    # def compute_energy(u, v, c, dx):
    #     """Compute kinetic and potential energy"""
    #     kinetic = 0.5 * np.sum(v**2) * dx
    #     u_with_bc = np.concatenate([[0], u, [0]])
    #     gradient = np.diff(u_with_bc) / dx
    #     potential = 0.5 * c**2 * np.sum(gradient**2) * dx
    #     return kinetic, potential, kinetic + potential

    # energies_kinetic = []
    # energies_potential = []
    # energies_total = []

    # print("Computing energy evolution...")
    # for i in range(len(t_history)):
    #     ke, pe, te = compute_energy(u_history[i], v_history[i], c, dx)
    #     energies_kinetic.append(ke)
    #     energies_potential.append(pe)
    #     energies_total.append(te)

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # ax1.plot(t_history, energies_kinetic, 'b-', label='Kinetic', linewidth=2)
    # ax1.plot(t_history, energies_potential, 'r-', label='Potential', linewidth=2)
    # ax1.plot(t_history, energies_total, 'k-', label='Total', linewidth=2)
    # ax1.set_xlabel('Time t (s)')
    # ax1.set_ylabel('Energy')
    # ax1.set_title('Energy Evolution - 1D String')
    # ax1.legend()
    # ax1.grid(True, alpha=0.3)

    # # Force magnitude over time (integrated over space)
    # force_magnitude = [np.sum(np.abs(f_history[i])) * dx for i in range(len(t_history))]
    # ax2.plot(t_history, force_magnitude, 'm-', linewidth=2)
    # ax2.set_xlabel('Time t (s)')
    # ax2.set_ylabel('Total Force Magnitude')
    # ax2.set_title('Applied Force vs Time')
    # ax2.grid(True, alpha=0.3)

    # plt.tight_layout()
    # plt.savefig('./figures/energy_1d.png', dpi=150, bbox_inches='tight')
    # print("Saved: energy_1d.png")
    # plt.close()

    # # ============================================================================
    # # Validation Results
    # # ============================================================================

    # print("\n" + "="*60)
    # print("SIMULATION RESULTS - 1D STRING")
    # print("="*60)

    # print(f"\nFinal time t = {T} s")
    # print(f"Final displacement range: [{np.min(u_history[-1]):.6f}, {np.max(u_history[-1]):.6f}]")
    # print(f"Final velocity range: [{np.min(v_history[-1]):.6f}, {np.max(v_history[-1]):.6f}]")

    # # Max displacement over time
    # max_displacement = np.max(np.abs(u_history))
    # print(f"\nMaximum displacement (entire simulation): {max_displacement:.6f}")

    # # Energy
    # initial_energy = energies_total[0]
    # final_energy = energies_total[-1]
    # max_energy = np.max(energies_total)

    # print(f"\nEnergy:")
    # print(f"  Initial energy: {initial_energy:.6f}")
    # print(f"  Maximum energy: {max_energy:.6f}")
    # print(f"  Final energy: {final_energy:.6f}")
    # print(f"  Energy injected by force: {max_energy - initial_energy:.6f}")
    # print(f"  Energy dissipated by damping: {max_energy - final_energy:.6f}")

    # print("\n" + "="*60)
    # print("All outputs saved successfully!")
    # print("  - wave_1d.gif (animation)")
    # print("  - energy_1d.png (energy evolution)")
    # print("  - data_1d.mat (MATLAB data)")
    # print("  - simulation_output_1d.txt (this text)")
    # print("="*60)

    # Close logger
    sys.stdout = logger.terminal
    logger.close()