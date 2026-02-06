"""
Compare trunk network predictions vs SVD U_basis modes for all 18 modes.
Shows side-by-side comparison at a given time instant for sample 0.
"""

import os
import numpy as np
import h5py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration
TIME_INSTANT = 0.5  # Which time to visualize (0.0 to 1.0)

# ============================================================
# DEFINE MLP ARCHITECTURE (MUST MATCH TRAINING)
# ============================================================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DeepONet(nn.Module):
    def __init__(self, trunk, branch):
        super().__init__()
        self.trunk = trunk
        self.branch = branch

    def forward(self, ic, coords):
        trunk_out = self.trunk(coords)  # (B, N_MODES)
        branch_out = self.branch(ic)    # (B, N_MODES)
        u = torch.sum(trunk_out * branch_out, dim=1)  # (B,)
        return u

print("=" * 70)
print("COMPARING TRUNK NETWORK VS SVD MODES FOR ALL 18 MODES")
print("=" * 70)

# ============================================================
# 1. LOAD MATLAB DATA
# ============================================================
print("\n[1/4] Loading MATLAB data...")
matlab_path = os.path.join(SCRIPT_DIR, 'data/test_cases.mat')
with h5py.File(matlab_path, 'r') as f:
    u_fom = np.array(f['U_data']).T  # (Nx, Ny, Nt, N_samples)

Nx, Ny, Nt, N_samples = u_fom.shape
print(f"Data shape: {u_fom.shape}")

# ============================================================
# 2. LOAD SVD DATA
# ============================================================
print("\n[2/4] Loading SVD data...")
svd_path = os.path.join(SCRIPT_DIR, 'data/svd_basis_data.npy')
svd_data = np.load(svd_path, allow_pickle=True).item()

U_basis = svd_data['basis']              # (Nx*Ny*Nt, p)
Sigma = svd_data['singular_values']      # (p,)
VT = svd_data['coefficients']            # (p, N_samples)

# Reshape SVD modes to spatiotemporal: (Nx, Ny, Nt, n_modes)
modes_svd = U_basis.reshape(Nx, Ny, Nt, U_basis.shape[1], order='F')
print(f"SVD modes shape: {modes_svd.shape}")

# ============================================================
# 3. LOAD DEEPONET AND GET TRUNK PREDICTIONS
# ============================================================
print("\n[3/4] Loading DeepONet and getting trunk predictions...")

# Find latest checkpoint
search_dirs = [
    os.path.join(SCRIPT_DIR, 'data'),
    os.path.join(SCRIPT_DIR, 'logs_2601'),
    os.path.join(SCRIPT_DIR, 'logs'),
]

latest_ckpt = None
for search_dir in search_dirs:
    if os.path.exists(search_dir):
        checkpoints = sorted(Path(search_dir).glob('deeponet_joint_supervised_*.pth'))
        if checkpoints:
            latest_ckpt = checkpoints[-1]
            break

if latest_ckpt is None:
    raise FileNotFoundError(f"No checkpoints found in: {search_dirs}")

print(f"Loading checkpoint: {latest_ckpt.name}")

# Load checkpoint
checkpoint = torch.load(latest_ckpt, map_location=DEVICE)
config = checkpoint['config']

N_MODES = config['n_modes']
N_SENSORS = config['n_sensors']
TRUNK_HIDDEN_DIM = config['trunk_hidden_dim']
TRUNK_N_LAYERS = config['trunk_n_layers']
BRANCH_HIDDEN_DIM = config['branch_hidden_dim']
BRANCH_N_LAYERS = config['branch_n_layers']

# Build trunk only
trunk = MLP(3, TRUNK_HIDDEN_DIM, N_MODES, TRUNK_N_LAYERS).to(DEVICE)
branch = MLP(N_SENSORS * N_SENSORS, BRANCH_HIDDEN_DIM, N_MODES, BRANCH_N_LAYERS).to(DEVICE)
model = DeepONet(trunk, branch).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model loaded: N_MODES={N_MODES}")

# Build coordinate grid
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
t = np.linspace(0, 1, Nt)
X, Y, T = np.meshgrid(x, y, t, indexing='ij')

# Select time instant
t_idx = int(TIME_INSTANT * (Nt - 1))
t_actual = t[t_idx]
print(f"\nTime instant: {TIME_INSTANT:.2f} (index {t_idx}, actual value {t_actual:.6f})")

# Get trunk predictions at this time instant
coords_t = np.stack([X[:, :, t_idx].flatten('F'),
                     Y[:, :, t_idx].flatten('F'),
                     np.full((Nx * Ny,), t_actual)], axis=1)
coords_tensor = torch.from_numpy(coords_t).float().to(DEVICE)

with torch.no_grad():
    trunk_out = model.trunk(coords_tensor).cpu().numpy()  # (Nx*Ny, N_MODES)

# Reshape to spatial: (Nx, Ny, N_MODES)
trunk_modes = trunk_out.reshape(Nx, Ny, N_MODES, order='F')
print(f"Trunk predictions shape: {trunk_modes.shape}")

# ============================================================
# 4. CREATE COMPARISON PLOT
# ============================================================
print("\n[4/4] Creating comparison plot...")

# Calculate global vmin/vmax across all modes (both trunk and SVD)
all_trunk_values = trunk_modes.flatten()
all_svd_values = modes_svd[:, :, t_idx, :N_MODES].flatten()
all_values = np.concatenate([all_trunk_values, all_svd_values])
vmin_global = all_values.min()
vmax_global = all_values.max()

print(f"Global value range: [{vmin_global:.6f}, {vmax_global:.6f}]")

# Organize as 9 rows × 4 columns
# Each row shows 2 modes, each mode has [Trunk, SVD]
n_rows = 9
n_cols = 4

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))

mode_idx = 0
for row in range(n_rows):
    for pair in range(2):  # 2 mode pairs per row
        if mode_idx >= N_MODES:
            # Hide unused subplots
            axes[row, 2*pair].axis('off')
            axes[row, 2*pair + 1].axis('off')
            continue
        
        # Trunk mode (left)
        trunk_mode = trunk_modes[:, :, mode_idx]
        ax_trunk = axes[row, 2*pair]
        im = ax_trunk.imshow(trunk_mode, cmap='seismic', origin='lower', 
                             vmin=vmin_global, vmax=vmax_global)
        ax_trunk.set_title(f'Trunk Mode {mode_idx}', fontsize=9)
        ax_trunk.set_xticks([])
        ax_trunk.set_yticks([])
        
        # SVD mode (right)
        svd_mode = modes_svd[:, :, t_idx, mode_idx]
        ax_svd = axes[row, 2*pair + 1]
        ax_svd.imshow(svd_mode, cmap='seismic', origin='lower',
                      vmin=vmin_global, vmax=vmax_global)
        ax_svd.set_title(f'SVD Mode {mode_idx}', fontsize=9)
        ax_svd.set_xticks([])
        ax_svd.set_yticks([])
        
        mode_idx += 1

# Add single colorbar on the right
fig.subplots_adjust(right=0.92, hspace=0.3, wspace=0.2)
cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Mode Value')

plt.suptitle(f'Trunk Network vs SVD Modes Comparison (Sample 0, t={t_actual:.2f})\n'
             f'Left: Trunk Prediction | Right: SVD Ground Truth', 
             fontsize=14, y=0.995)

save_path = os.path.join(SCRIPT_DIR, f'data/trunk_vs_svd_all_modes_t{TIME_INSTANT:.2f}.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Comparison plot saved to {save_path}")
plt.close()

# ============================================================
# 5. COMPUTE ERROR METRICS
# ============================================================
print("\n" + "=" * 70)
print("ERROR METRICS FOR EACH MODE")
print("=" * 70)

total_l2_error = 0.0
total_max_error = 0.0

for mode_idx in range(N_MODES):
    trunk_mode = trunk_modes[:, :, mode_idx]
    svd_mode = modes_svd[:, :, t_idx, mode_idx]
    
    l2_error = np.linalg.norm(trunk_mode - svd_mode) / (np.linalg.norm(svd_mode) + 1e-10)
    max_error = np.abs(trunk_mode - svd_mode).max()
    
    total_l2_error += l2_error
    total_max_error += max_error
    
    print(f"Mode {mode_idx:2d}: Relative L2 = {l2_error:.6e}, Max Abs = {max_error:.6e}")

print(f"\nAverage Relative L2 Error: {total_l2_error / N_MODES:.6e}")
print(f"Average Max Abs Error: {total_max_error / N_MODES:.6e}")

print("\n" + "=" * 70)
print("✓ COMPARISON COMPLETE")
print("=" * 70)
