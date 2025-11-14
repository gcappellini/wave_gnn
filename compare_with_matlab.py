"""
Compare PINN-DeepONet results with MATLAB reference solution
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from pinn_deeponet_heat_2branch import PINNDeepONet

# Load MATLAB solution
print("Loading MATLAB reference solution...")
matlab_data = np.loadtxt('heat_equation_matlab_solution.txt')
x_matlab = matlab_data[:, 0]
t_matlab = matlab_data[:, 1]
u_matlab = matlab_data[:, 2]

# Reshape to grid (assumes MATLAB saved with meshgrid format)
# Determine grid size
nx = len(np.unique(x_matlab))
nt = len(np.unique(t_matlab))
print(f"MATLAB solution grid: nx={nx}, nt={nt}")

X_matlab = x_matlab.reshape(nt, nx)
T_matlab = t_matlab.reshape(nt, nx)
U_matlab = u_matlab.reshape(nt, nx)

# Test case parameters (must match MATLAB and training)
a_test = 1.5
source_amplitude = 5.0
source_type = 'gaussian'

# Load trained PINN model
print("\nLoading trained PINN-DeepONet model...")
model = PINNDeepONet(
    n_sensors_ic=20,
    n_sensors_src=20,
    branch_hidden=50,
    trunk_hidden=50,
    p=50
)

# Try to load saved model weights (if available)
try:
    model.load_state_dict(torch.load('pinn_deeponet_heat.pth'))
    print("✓ Model weights loaded from file")
except FileNotFoundError:
    print("⚠ No saved model found. Please train the model first!")
    print("Run the training script with model.state_dict() saved.")
    exit(1)

model.eval()

# Generate predictions on MATLAB grid
print("\nGenerating PINN predictions on MATLAB grid...")
ic_sensors = model.generate_ic(a_test)
src_sensors = model.generate_source(source_type, source_amplitude)

# Create input for PINN (use same grid as MATLAB)
x_pinn = torch.tensor(X_matlab[0, :], dtype=torch.float32)  # First row (all x values)
t_pinn = torch.tensor(T_matlab[:, 0], dtype=torch.float32)  # First column (all t values)

X_pinn_mesh, T_pinn_mesh = torch.meshgrid(x_pinn, t_pinn, indexing='ij')
xt_grid = torch.stack([X_pinn_mesh.flatten(), T_pinn_mesh.flatten()], dim=1)

with torch.no_grad():
    u_pinn = model.forward(ic_sensors, src_sensors, xt_grid)
    u_pinn = u_pinn.reshape(nx, nt).numpy()

# Note: MATLAB solution is (nt, nx), PINN is (nx, nt), so transpose PINN
U_pinn = u_pinn.T  # Now both are (nt, nx)

print("✓ PINN predictions generated")

# Compute errors
print("\n" + "="*70)
print("ERROR ANALYSIS")
print("="*70)

# Point-wise absolute error
abs_error = np.abs(U_pinn - U_matlab)
rel_error = abs_error / (np.abs(U_matlab) + 1e-10)

# Statistics
print(f"Max absolute error: {np.max(abs_error):.6e}")
print(f"Mean absolute error: {np.mean(abs_error):.6e}")
print(f"RMS error: {np.sqrt(np.mean(abs_error**2)):.6e}")
print(f"Max relative error: {np.max(rel_error):.6e}")
print(f"Mean relative error: {np.mean(rel_error):.6e}")

# L2 norm error
l2_error = np.sqrt(np.sum(abs_error**2)) / np.sqrt(np.sum(U_matlab**2))
print(f"Relative L2 error: {l2_error:.6e}")

# Visualization
fig = plt.figure(figsize=(18, 10))

# Plot 1: MATLAB solution
ax1 = plt.subplot(2, 3, 1)
im1 = ax1.contourf(T_matlab, X_matlab, U_matlab, levels=20, cmap='RdBu_r')
ax1.set_xlabel('t')
ax1.set_ylabel('x')
ax1.set_title('MATLAB Reference Solution')
plt.colorbar(im1, ax=ax1)

# Plot 2: PINN solution
ax2 = plt.subplot(2, 3, 2)
im2 = ax2.contourf(T_matlab, X_matlab, U_pinn, levels=20, cmap='RdBu_r')
ax2.set_xlabel('t')
ax2.set_ylabel('x')
ax2.set_title('PINN-DeepONet Solution')
plt.colorbar(im2, ax=ax2)

# Plot 3: Absolute error
ax3 = plt.subplot(2, 3, 3)
im3 = ax3.contourf(T_matlab, X_matlab, abs_error, levels=20, cmap='hot')
ax3.set_xlabel('t')
ax3.set_ylabel('x')
ax3.set_title(f'Absolute Error (max={np.max(abs_error):.2e})')
plt.colorbar(im3, ax=ax3)

# Plot 4: Snapshots comparison at t=0
ax4 = plt.subplot(2, 3, 4)
ax4.plot(X_matlab[0, :], U_matlab[0, :], 'b-', linewidth=2, label='MATLAB')
ax4.plot(X_matlab[0, :], U_pinn[0, :], 'r--', linewidth=2, label='PINN')
ax4.set_xlabel('x')
ax4.set_ylabel('u(x,t)')
ax4.set_title('t = 0.000')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Snapshots comparison at t=T/2
mid_idx = nt // 2
ax5 = plt.subplot(2, 3, 5)
ax5.plot(X_matlab[mid_idx, :], U_matlab[mid_idx, :], 'b-', linewidth=2, label='MATLAB')
ax5.plot(X_matlab[mid_idx, :], U_pinn[mid_idx, :], 'r--', linewidth=2, label='PINN')
ax5.set_xlabel('x')
ax5.set_ylabel('u(x,t)')
ax5.set_title(f't = {T_matlab[mid_idx, 0]:.3f}')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Snapshots comparison at t=T_final
ax6 = plt.subplot(2, 3, 6)
ax6.plot(X_matlab[-1, :], U_matlab[-1, :], 'b-', linewidth=2, label='MATLAB')
ax6.plot(X_matlab[-1, :], U_pinn[-1, :], 'r--', linewidth=2, label='PINN')
ax6.set_xlabel('x')
ax6.set_ylabel('u(x,t)')
ax6.set_title(f't = {T_matlab[-1, 0]:.3f}')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_matlab_vs_pinn.png', dpi=150, bbox_inches='tight')
print("\n✓ Comparison plot saved: comparison_matlab_vs_pinn.png")

plt.show()

print("\n" + "="*70)
print("DONE!")
print("="*70)
