"""
Compare three reconstruction methods for sample 0:
1. Ground truth (MATLAB test_cases.mat)
2. SVD reconstruction (svd_basis_data.npy)
3. DeepONet prediction (joint supervised)
"""

import os
import numpy as np
import h5py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Create timestamped output directory
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', TIMESTAMP)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
print("COMPARING RECONSTRUCTION METHODS FOR SAMPLE 0")
print("=" * 70)

# ============================================================
# 1. LOAD GROUND TRUTH FROM MATLAB
# ============================================================
print("\n[1/3] Loading ground truth from MATLAB...")
matlab_path = os.path.join(SCRIPT_DIR, 'data/test_cases.mat')
with h5py.File(matlab_path, 'r') as f:
    u_fom = np.array(f['U_data']).T  # (Nx, Ny, Nt, N_samples)

Nx, Ny, Nt, N_samples = u_fom.shape
print(f"Ground truth shape: {u_fom.shape}")

# ============================================================
# 2. LOAD SVD DATA AND RECONSTRUCT SAMPLE 0
# ============================================================
print("\n[2/3] Loading SVD data and reconstructing sample 0...")
svd_path = os.path.join(SCRIPT_DIR, 'data/svd_basis_data.npy')
svd_data = np.load(svd_path, allow_pickle=True).item()

U_basis = svd_data['basis']              # (Nx*Ny*Nt, p)
Sigma = svd_data['singular_values']      # (p,)
VT = svd_data['coefficients']            # (p, N_samples)

print(f"SVD modes: {U_basis.shape}")
print(f"SVD coefficients: {VT.shape}")

# Reconstruct sample 0: u = U_basis @ (Sigma * VT[:, 0])
sample_idx = 0
actual_modes = U_basis.shape[1]
reconstructed_flat = U_basis @ (Sigma[:actual_modes] * VT[:actual_modes, sample_idx])
u_svd = reconstructed_flat.reshape(Nx, Ny, Nt, order='F')
print(f"SVD reconstruction shape: {u_svd.shape}")

# ============================================================
# 3. LOAD DEEPONET AND MAKE PREDICTIONS
# ============================================================
print("\n[3/3] Loading DeepONet and making predictions...")

# Find latest checkpoint
search_dirs = [
    os.path.join(SCRIPT_DIR, 'models'),
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
norm_data = checkpoint['normalization']

N_MODES = config['n_modes']
N_SENSORS = config['n_sensors']
TRUNK_HIDDEN_DIM = config['trunk_hidden_dim']
TRUNK_N_LAYERS = config['trunk_n_layers']
BRANCH_HIDDEN_DIM = config['branch_hidden_dim']
BRANCH_N_LAYERS = config['branch_n_layers']

ic_min = norm_data['ic_min']
ic_max = norm_data['ic_max']
ic_range = norm_data['ic_range']
u_min = norm_data['u_min']
u_max = norm_data['u_max']
u_range = norm_data['u_range']

# Build models
ic_dim = N_SENSORS * N_SENSORS
trunk = MLP(3, TRUNK_HIDDEN_DIM, N_MODES, TRUNK_N_LAYERS).to(DEVICE)
branch = MLP(ic_dim, BRANCH_HIDDEN_DIM, N_MODES, BRANCH_N_LAYERS).to(DEVICE)
model = DeepONet(trunk, branch).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model loaded: N_MODES={N_MODES}, N_SENSORS={N_SENSORS}")

# Extract IC sensors for sample 0
sensor_x_indices = np.linspace(0, Nx - 1, N_SENSORS, dtype=int)
sensor_y_indices = np.linspace(0, Ny - 1, N_SENSORS, dtype=int)

u_ic = u_fom[:, :, 0, 0]
sensors = []
for si in sensor_x_indices:
    for sj in sensor_y_indices:
        sensors.append(u_ic[si, sj])
ic_sensors = np.array(sensors)

# Normalize IC
ic_norm = 2 * (ic_sensors - ic_min.squeeze()) / (ic_range.squeeze() + 1e-10) - 1
ic0_tensor = torch.from_numpy(ic_norm).float().to(DEVICE).unsqueeze(0)

# Build coordinate grid
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
t = np.linspace(0, 1, Nt)
X, Y, T = np.meshgrid(x, y, t, indexing='ij')

# Make predictions at 5 time instants
time_instants = [0.0, 0.25, 0.5, 0.75, 1.0]
time_indices = [int(ti * (Nt - 1)) for ti in time_instants]

print("\nMaking predictions at 5 time instants...")
predictions_deeponet = []

with torch.no_grad():
    for t_val, t_idx in zip(time_instants, time_indices):
        coords_t = np.stack([X[:, :, t_idx].flatten('F'),
                             Y[:, :, t_idx].flatten('F'),
                             np.full((Nx * Ny,), t_val)], axis=1)
        coords_tensor = torch.from_numpy(coords_t).float().to(DEVICE)
        ic_batch = ic0_tensor.repeat(coords_tensor.shape[0], 1)
        
        u_pred_norm = model(ic_batch, coords_tensor).cpu().numpy()
        u_pred = (u_pred_norm + 1) * u_range / 2 + u_min
        predictions_deeponet.append(u_pred.reshape(Nx, Ny, order='F'))

print("✓ Predictions complete")

# ============================================================
# 4. CREATE COMPARISON PLOT
# ============================================================
print("\n[4/4] Creating comparison plot...")

# Calculate global vmin/vmax across all times and methods
all_values = []
for i, t_idx in enumerate(time_indices):
    all_values.append(u_fom[:, :, t_idx, 0].flatten())
    all_values.append(u_svd[:, :, t_idx].flatten())
    all_values.append(predictions_deeponet[i].flatten())

all_values = np.concatenate(all_values)
vmin_global = all_values.min()
vmax_global = all_values.max()

fig, axes = plt.subplots(len(time_instants), 3, figsize=(15, 4 * len(time_instants)))

for i, (t_val, t_idx) in enumerate(zip(time_instants, time_indices)):
    # Ground truth
    gt = u_fom[:, :, t_idx, 0]
    
    # SVD reconstruction
    svd_recon = u_svd[:, :, t_idx]
    
    # DeepONet prediction
    deeponet_pred = predictions_deeponet[i]
    
    # Column 1: Ground Truth
    im0 = axes[i, 0].imshow(gt, cmap='seismic', origin='lower', vmin=vmin_global, vmax=vmax_global)
    axes[i, 0].set_title(f"Ground Truth", fontsize=11)
    axes[i, 0].set_xticks([])
    axes[i, 0].set_yticks([])
    
    # Column 2: SVD Reconstruction
    im1 = axes[i, 1].imshow(svd_recon, cmap='seismic', origin='lower', vmin=vmin_global, vmax=vmax_global)
    axes[i, 1].set_title(f"SVD Reconstruction", fontsize=11)
    axes[i, 1].set_xticks([])
    axes[i, 1].set_yticks([])
    
    # Column 3: DeepONet Prediction
    im2 = axes[i, 2].imshow(deeponet_pred, cmap='seismic', origin='lower', vmin=vmin_global, vmax=vmax_global)
    axes[i, 2].set_title(f"DeepONet Prediction", fontsize=11)
    axes[i, 2].set_xticks([])
    axes[i, 2].set_yticks([])
    
    # Add time label on the left
    axes[i, 0].set_ylabel(f't={t_val:.2f}', fontsize=12, fontweight='bold')

# Add single colorbar on the right
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
fig.colorbar(im2, cax=cbar_ax, label='Solution Value')

plt.suptitle('Comparison of Reconstruction Methods (Sample 0)\nGround Truth | SVD | DeepONet', 
             fontsize=14, y=0.995)
plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, 'comparison_reconstructions_sample0.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Comparison plot saved to {save_path}")
plt.close()

# ============================================================
# 5. COMPUTE ERROR METRICS
# ============================================================
print("\n" + "=" * 70)
print("ERROR METRICS FOR SAMPLE 0")
print("=" * 70)

for i, (t_val, t_idx) in enumerate(zip(time_instants, time_indices)):
    gt = u_fom[:, :, t_idx, 0]
    svd_recon = u_svd[:, :, t_idx]
    deeponet_pred = predictions_deeponet[i]
    
    # SVD errors
    svd_l2_error = np.linalg.norm(svd_recon - gt) / np.linalg.norm(gt)
    svd_max_error = np.abs(svd_recon - gt).max()
    
    # DeepONet errors
    deeponet_l2_error = np.linalg.norm(deeponet_pred - gt) / np.linalg.norm(gt)
    deeponet_max_error = np.abs(deeponet_pred - gt).max()
    
    print(f"\nt={t_val:.2f}:")
    print(f"  SVD Reconstruction:")
    print(f"    Relative L2 error: {svd_l2_error:.6e}")
    print(f"    Max absolute error: {svd_max_error:.6e}")
    print(f"  DeepONet Prediction:")
    print(f"    Relative L2 error: {deeponet_l2_error:.6e}")
    print(f"    Max absolute error: {deeponet_max_error:.6e}")

print("\n" + "=" * 70)
print("✓ COMPARISON COMPLETE")
print("=" * 70)
