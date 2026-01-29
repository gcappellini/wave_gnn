"""
Visualize Trained Trunk Network Predictions
- Load checkpoint from train_trunk_svd_simple.py
- Plot all 18 modes at time instants: 0, 0.25, 0.5, 0.75, 1.0
- Compare predictions across time evolution
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, 'data/trunk_svd_simple.pth')

print("=" * 70)
print("VISUALIZING TRUNK NETWORK PREDICTIONS")
print("=" * 70)

# ============================================================
# 1. LOAD CHECKPOINT
# ============================================================
print("\n1. Loading checkpoint...")
if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
hyperparams = checkpoint['hyperparams']
normalization = checkpoint['normalization']
grid_info = checkpoint['grid_info']

N_MODES = hyperparams['n_modes']
TRUNK_HIDDEN_DIM = hyperparams['hidden_dim']
TRUNK_N_LAYERS = hyperparams['n_layers']
Nx = grid_info['Nx']
Ny = grid_info['Ny']
Nt = grid_info['Nt']
targets_min = normalization['min']
targets_max = normalization['max']
targets_range = targets_max - targets_min

print(f"Loaded checkpoint: {N_MODES} modes, {Nx}x{Ny}x{Nt} grid")
print(f"Network: hidden_dim={TRUNK_HIDDEN_DIM}, n_layers={TRUNK_N_LAYERS}")

# ============================================================
# 2. RECONSTRUCT MODEL
# ============================================================
print("\n2. Reconstructing model...")

class SimpleTrunk(nn.Module):
    """Simple MLP: (x,y,t) -> [18 basis values]"""
    def __init__(self, hidden_dim, output_dim, n_layers):
        super().__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(3, hidden_dim))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

trunk = SimpleTrunk(
    hidden_dim=TRUNK_HIDDEN_DIM,
    output_dim=N_MODES,
    n_layers=TRUNK_N_LAYERS
)
trunk.load_state_dict(checkpoint['model_state_dict'])
trunk.eval()

print("✓ Model reconstructed and loaded")

# ============================================================
# 3. GENERATE COORDINATES
# ============================================================
print("\n3. Generating coordinates...")
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
t = np.linspace(0, 1, Nt)

X, Y, T = np.meshgrid(x, y, t, indexing='ij')
coords = np.stack([X.flatten('F'), Y.flatten('F'), T.flatten('F')], axis=1)

print(f"Coordinates shape: {coords.shape}")

# ============================================================
# 4. LOAD TRUE SVD BASIS
# ============================================================
print("\n4. Loading true SVD basis...")
svd_data_path = os.path.join(SCRIPT_DIR, 'data/svd_basis_data.npy')
svd_data = np.load(svd_data_path, allow_pickle=True).item()
true_basis = svd_data['basis'][:, :N_MODES]  # (n_space_time, N_MODES)
print(f"True basis shape: {true_basis.shape}")

# ============================================================
# 5. PREDICT ON ALL COORDINATES
# ============================================================
print("\n5. Making predictions...")
coords_tensor = torch.from_numpy(coords).float()

with torch.no_grad():
    pred_all = trunk(coords_tensor).numpy()

# Denormalize
pred_all_denorm = (pred_all + 1) / 2 * targets_range + targets_min
print(f"Predictions shape: {pred_all_denorm.shape}")

# Compute relative error (pointwise)
# Avoid division by zero with a small epsilon
epsilon = 1e-12
abs_error = np.abs(true_basis - pred_all_denorm)
denom = np.maximum(np.abs(true_basis), epsilon)
rel_error = abs_error / denom
print(f"Relative error range: [{rel_error.min():.6e}, {rel_error.max():.6e}]")
print(f"Absolute error range: [{abs_error.min():.6e}, {abs_error.max():.6e}]")

# ============================================================
# 6. PLOT MODES AT 5 TIME INSTANTS
# ============================================================
print("\n6. Creating prediction visualizations...")

# Time instants to visualize
time_instants = [0.0, 0.25, 0.5, 0.75, 1.0]
t_indices = [int(t_inst * (Nt - 1)) for t_inst in time_instants]

for t_inst, t_idx in zip(time_instants, t_indices):
    print(f"   Plotting predictions at t={t_inst} (index {t_idx})...")
    
    # Extract spatial slice at this time
    layer_size = Nx * Ny
    start_idx = t_idx * layer_size
    end_idx = start_idx + layer_size
    
    pred_at_t = pred_all_denorm[start_idx:end_idx, :]  # (Nx*Ny, N_MODES)
    true_at_t = true_basis[start_idx:end_idx, :]       # (Nx*Ny, N_MODES)
    
    # Compute global min/max for this time instant across GT and predictions
    vmin_global = min(true_at_t.min(), pred_at_t.min())
    vmax_global = max(true_at_t.max(), pred_at_t.max())
    
    # Create 6x6 grid: GT and Pred side-by-side for each mode
    fig, axes = plt.subplots(6, 6, figsize=(18, 20), constrained_layout=True)
    fig.suptitle(f'GT vs Predicted Modes at t={t_inst}', fontsize=16)
    
    for i in range(N_MODES):
        row = i // 3
        col = i % 3
        
        gt_ax = axes[row, 2 * col]
        pred_ax = axes[row, 2 * col + 1]
        
        gt_slice = true_at_t[:, i].reshape(Nx, Ny, order='F')
        pred_slice = pred_at_t[:, i].reshape(Nx, Ny, order='F')
        
        im = gt_ax.imshow(gt_slice, cmap='seismic', origin='lower',
                          vmin=vmin_global, vmax=vmax_global)
        pred_ax.imshow(pred_slice, cmap='seismic', origin='lower',
                       vmin=vmin_global, vmax=vmax_global)
        
        gt_ax.set_title(f'Mode {i} GT', fontsize=9)
        pred_ax.set_title(f'Mode {i} Pred', fontsize=9)
        gt_ax.set_xticks([])
        gt_ax.set_yticks([])
        pred_ax.set_xticks([])
        pred_ax.set_yticks([])
    
    # Add single colorbar for entire figure
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cbar.set_label('Value', rotation=270, labelpad=15)
    
    # Save with t in filename
    t_str = f"{t_inst:.2f}".replace('.', '_')
    save_path = os.path.join(SCRIPT_DIR, f'data/trunk_predictions_t_{t_str}.png')
    plt.savefig(save_path, dpi=150)
    print(f"   ✓ Saved predictions to {save_path}")
    plt.close()

# ============================================================
# 7. PLOT RELATIVE ERRORS AT 5 TIME INSTANTS
# ============================================================
print("\n7. Creating relative error visualizations...")

for t_inst, t_idx in zip(time_instants, t_indices):
    print(f"   Plotting relative errors at t={t_inst} (index {t_idx})...")
    
    # Extract spatial slice at this time
    layer_size = Nx * Ny
    start_idx = t_idx * layer_size
    end_idx = start_idx + layer_size
    
    error_at_t = rel_error[start_idx:end_idx, :]  # (Nx*Ny, N_MODES)
    
    # Compute global min/max for this time instant across all 18 modes
    vmin_global = 0.0
    vmax_global = error_at_t.max()
    
    # Create 6x3 grid for 18 modes
    fig, axes = plt.subplots(6, 3, figsize=(15, 20), constrained_layout=True)
    fig.suptitle(f'Relative Error at t={t_inst}', fontsize=16)
    
    for i in range(N_MODES):
        row = i // 3
        col = i % 3
        
        # Reshape spatial data to 2D grid
        error_slice = error_at_t[:, i].reshape(Nx, Ny, order='F')
        
        # Plot with global scale (hot colormap for errors)
        im = axes[row, col].imshow(error_slice, cmap='hot', origin='lower', 
                                    vmin=vmin_global, vmax=vmax_global)
        axes[row, col].set_title(f'Mode {i}', fontsize=9)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
    
    # Add single colorbar for entire figure
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cbar.set_label('Relative Error', rotation=270, labelpad=15)
    
    # Save with t in filename
    t_str = f"{t_inst:.2f}".replace('.', '_')
    save_path = os.path.join(SCRIPT_DIR, f'data/trunk_rel_errors_t_{t_str}.png')
    plt.savefig(save_path, dpi=150)
    print(f"   ✓ Saved relative errors to {save_path}")
    plt.close()

print("\n" + "=" * 70)
print("✓ Visualization complete!")
print(f"Generated 10 PNG files (5 predictions + 5 error maps)")
print("=" * 70)
