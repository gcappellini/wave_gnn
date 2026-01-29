"""
Visualization of DeepONet Predictions
- Load trained DeepONet checkpoint (frozen trunk + trained branch)
- Generate predictions on test set
- Plot: GT vs Predicted SVD coefficients + errors
- Visualize reconstructed solution at 5 time instants
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_MODES = 18

print(f"Device: {DEVICE}")

# Find most recent DeepONet checkpoint
checkpoint_dir = os.path.join(SCRIPT_DIR, 'data')
deeponet_checkpoints = sorted(Path(checkpoint_dir).glob('deeponet_supervised_*.pth'))
if not deeponet_checkpoints:
    raise FileNotFoundError(f"No DeepONet checkpoint found in {checkpoint_dir}")

checkpoint_path = str(deeponet_checkpoints[-1])
print(f"Loading checkpoint: {checkpoint_path}")

# ============================================================
# 1. LOAD SVD DATA AND GRID INFO
# ============================================================
print("\n1. Loading SVD basis data...")
svd_data = np.load(os.path.join(SCRIPT_DIR, 'data/svd_basis_data.npy'), allow_pickle=True).item()
U_basis = svd_data['basis']
grid_info = svd_data['grid_info']
Nx, Ny, Nt = grid_info.astype(int)

U_basis_truncated = U_basis[:, :N_MODES]
print(f"Grid: {Nx} x {Ny} x {Nt}")
print(f"SVD basis shape: {U_basis_truncated.shape}")

# ============================================================
# 2. GENERATE COORDINATES
# ============================================================
print("\n2. Generating coordinates...")
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
t = np.linspace(0, 1, Nt)

X, Y, T = np.meshgrid(x, y, t, indexing='ij')
coords = np.stack([X.flatten('F'), Y.flatten('F'), T.flatten('F')], axis=1)

print(f"Coordinates shape: {coords.shape}")

# ============================================================
# 3. LOAD CHECKPOINT AND RECONSTRUCT MODELS
# ============================================================
print("\n3. Reconstructing models from checkpoint...")

checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

# Get configurations
branch_config = checkpoint['branch_config']
trunk_config = checkpoint['trunk_config']

# Build Trunk (with frozen weights)
class SimpleTrunk(nn.Module):
    def __init__(self, hidden_dim, output_dim, n_layers):
        super().__init__()
        layers = []
        layers.append(nn.Linear(3, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

trunk_model = SimpleTrunk(
    hidden_dim=trunk_config['hidden_dim'],
    output_dim=trunk_config['output_dim'],
    n_layers=trunk_config['n_layers']
).to(DEVICE)
trunk_model.load_state_dict(checkpoint['trunk_state_dict'])
trunk_model.eval()

# Build Branch
class BranchNetwork(nn.Module):
    def __init__(self, ic_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        layers = []
        layers.append(nn.Linear(ic_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, ic):
        return self.net(ic)

branch_model = BranchNetwork(
    ic_dim=branch_config['ic_dim'],
    hidden_dim=branch_config['hidden_dim'],
    output_dim=branch_config['output_dim'],
    n_layers=branch_config['n_layers']
).to(DEVICE)
branch_model.load_state_dict(checkpoint['branch_state_dict'])
branch_model.eval()

# Build DeepONet
class DeepONet(nn.Module):
    def __init__(self, branch, trunk):
        super().__init__()
        self.branch = branch
        self.trunk = trunk
    
    def forward(self, ic, coords):
        branch_out = self.branch(ic)
        trunk_out = self.trunk(coords)
        predictions = branch_out * trunk_out
        return predictions

deeponet = DeepONet(branch_model, trunk_model).to(DEVICE)
print("✓ DeepONet model reconstructed")

# ============================================================
# 4. PREPARE INITIAL CONDITIONS (extract from t=0)
# ============================================================
print("\n4. Preparing initial conditions from t=0...")
n_spatial = Nx * Ny
N_SENSORS = 8  # Match training configuration

# Sensor locations: uniform grid
sensor_x_indices = np.linspace(0, Nx-1, N_SENSORS, dtype=int)
sensor_y_indices = np.linspace(0, Ny-1, N_SENSORS, dtype=int)

# Reconstruct full field at t=0 from SVD
ic_full_field = np.zeros((Nx, Ny))
for mode_idx in range(N_MODES):
    mode_spatial_field = U_basis_truncated[:n_spatial, mode_idx].reshape((Nx, Ny), order='F')
    ic_full_field += mode_spatial_field

# Extract values at sensor locations
ic_at_sensors = []
for si in sensor_x_indices:
    row = []
    for sj in sensor_y_indices:
        row.append(ic_full_field[si, sj])
    ic_at_sensors.append(row)

ic_field_spatial = np.array(ic_at_sensors)  # (N_SENSORS, N_SENSORS)
ic_field_flattened = ic_field_spatial.flatten()  # (N_SENSORS*N_SENSORS,)

print(f"IC sensor measurements shape: {ic_field_flattened.shape}")
print(f"IC range: [{ic_field_flattened.min():.6e}, {ic_field_flattened.max():.6e}]")

# ============================================================
# 5. SELECT TIME INSTANTS AND MAKE PREDICTIONS
# ============================================================
print("\n5. Selecting time instants and making predictions...")

time_indices = [0, Nt//4, Nt//2, 3*Nt//4, Nt-1]
time_values = [t[i] for i in time_indices]

print(f"Time indices: {time_indices}")
print(f"Time values: {time_values}")

# Prepare IC tensor (same for all predictions)
ic_tensor = torch.from_numpy(ic_field_flattened).float().to(DEVICE).unsqueeze(0)  # (1, N_SENSORS*N_SENSORS)

# Generate predictions only for selected time instants
pred_all_times = []

for t_idx in time_indices:
    print(f"  Predicting t={t[t_idx]:.3f}...")
    
    # Generate coordinates for all spatial points at this time instant
    coords_at_t = []
    for i in range(Nx):
        for j in range(Ny):
            coords_at_t.append([x[i], y[j], t[t_idx]])
    
    coords_at_t = np.array(coords_at_t)  # (Nx*Ny, 3)
    coords_tensor = torch.from_numpy(coords_at_t).float().to(DEVICE)
    
    # Replicate IC for all spatial points
    ic_batch = ic_tensor.repeat(Nx * Ny, 1)  # (Nx*Ny, N_SENSORS*N_SENSORS)
    
    # Predict
    with torch.no_grad():
        pred_t = deeponet(ic_batch, coords_tensor)  # (Nx*Ny, N_MODES)
    
    pred_all_times.append(pred_t.cpu().numpy())

print(f"✓ Predictions completed for {len(time_indices)} time instants")

# ============================================================
# 6. DENORMALIZE PREDICTIONS
# ============================================================
print("\n6. Denormalizing predictions...")

norm = checkpoint['normalization']
targets_min = norm['targets_min']
targets_max = norm['targets_max']
targets_range = norm['targets_range']

# Denormalize predictions
pred_all_denorm = []
for pred_t in pred_all_times:
    pred_t_denorm = (pred_t + 1) * targets_range / 2 + targets_min
    pred_all_denorm.append(pred_t_denorm)

print(f"✓ Denormalized {len(pred_all_denorm)} time instants")

# ============================================================
# 7. COMPUTE ERRORS
# ============================================================
print("\n7. Computing errors...")

abs_errors = []
for idx, t_idx in enumerate(time_indices):
    start_idx = t_idx * Nx * Ny
    end_idx = (t_idx + 1) * Nx * Ny
    
    true_t = U_basis_truncated[start_idx:end_idx, :]
    pred_t = pred_all_denorm[idx]
    
    abs_error_t = np.abs(true_t - pred_t)
    abs_errors.append(abs_error_t)
    
    print(f"  t={t[t_idx]:.3f}: mean abs error = {abs_error_t.mean():.6e}")

# ============================================================
# 8. PLOT GT vs PREDICTIONS (6x6 grid per time step)
# ============================================================
print("\n8. Creating GT vs Prediction plots...")

for idx, (t_idx, t_val) in enumerate(zip(time_indices, time_values)):
    # Get ground truth for this time instant
    start_idx = t_idx * Nx * Ny
    end_idx = (t_idx + 1) * Nx * Ny
    true_at_t = U_basis_truncated[start_idx:end_idx, :]  # (Nx*Ny, N_MODES)
    
    # Get prediction for this time instant
    pred_at_t = pred_all_denorm[idx]  # (Nx*Ny, N_MODES)
    
    # Reshape to spatial grid (Fortran order to match data layout)
    true_at_t_spatial = true_at_t.reshape(Nx, Ny, N_MODES, order='F')
    pred_at_t_spatial = pred_at_t.reshape(Nx, Ny, N_MODES, order='F')
    
    # Global vmin/vmax for this time step
    vmin = np.min([true_at_t_spatial.min(), pred_at_t_spatial.min()])
    vmax = np.max([true_at_t_spatial.max(), pred_at_t_spatial.max()])
    
    # Create figure with 6x6 grid (GT and Pred side-by-side for 18 modes)
    fig = plt.figure(figsize=(14, 16))
    fig.suptitle(f'GT vs Predictions at t={t_val:.3f}', fontsize=16, fontweight='bold')
    
    for mode_idx in range(N_MODES):
        # GT
        ax_gt = plt.subplot(6, 6, 2*mode_idx + 1)
        im_gt = ax_gt.imshow(true_at_t_spatial[:, :, mode_idx], 
                             cmap='seismic', vmin=vmin, vmax=vmax, origin='lower')
        ax_gt.set_title(f'Mode {mode_idx}: GT', fontsize=9)
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])
        
        # Pred
        ax_pred = plt.subplot(6, 6, 2*mode_idx + 2)
        im_pred = ax_pred.imshow(pred_at_t_spatial[:, :, mode_idx], 
                                 cmap='seismic', vmin=vmin, vmax=vmax, origin='lower')
        ax_pred.set_title(f'Mode {mode_idx}: Pred', fontsize=9)
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
    
    # Add single colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im_gt, cax=cbar_ax)
    cbar.set_label('Value', fontsize=10)
    
    plt.subplots_adjust(left=0.05, right=0.9, top=0.96, bottom=0.02, 
                        wspace=0.15, hspace=0.4)
    
    output_path = os.path.join(SCRIPT_DIR, f'deeponet_predictions_t_{t_val:.2f}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    plt.close()

# ============================================================
# 9. PLOT ABSOLUTE ERRORS (6x3 grid per time step)
# ============================================================
print("\n9. Creating absolute error plots...")

for idx, (t_idx, t_val) in enumerate(zip(time_indices, time_values)):
    error_at_t = abs_errors[idx]  # (Nx*Ny, N_MODES)
    error_at_t_spatial = error_at_t.reshape(Nx, Ny, N_MODES, order='F')
    
    # Global vmin/vmax for absolute error at this time
    vmin = error_at_t_spatial.min()
    vmax = error_at_t_spatial.max()
    
    # Create figure with 6x3 grid
    fig, axes = plt.subplots(6, 3, figsize=(10, 14))
    fig.suptitle(f'Absolute Error at t={t_val:.3f}', fontsize=14, fontweight='bold')
    
    mode_counter = 0
    for i in range(6):
        for j in range(3):
            if mode_counter < N_MODES:
                im = axes[i, j].imshow(error_at_t_spatial[:, :, mode_counter], 
                                       cmap='hot', vmin=vmin, vmax=vmax, origin='lower')
                axes[i, j].set_title(f'Mode {mode_counter}', fontsize=9)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                mode_counter += 1
            else:
                axes[i, j].axis('off')
    
    # Add single colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Absolute Error', fontsize=10)
    
    plt.subplots_adjust(left=0.05, right=0.9, top=0.96, bottom=0.02,
                        wspace=0.15, hspace=0.4)
    
    output_path = os.path.join(SCRIPT_DIR, f'deeponet_abs_errors_t_{t_val:.2f}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    plt.close()

print("\n" + "="*60)
print("DeepONet visualization completed!")
print("="*60)
