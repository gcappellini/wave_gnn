"""
Compare branch network coefficients vs SVD coefficients (Sigma * VT) for sample 0.
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

SAMPLE_IDX = 0

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
print("COMPARING BRANCH COEFFICIENTS VS SVD COEFFICIENTS")
print("=" * 70)

# ============================================================
# 1. LOAD MATLAB DATA (IC SENSORS)
# ============================================================
print("\n[1/4] Loading MATLAB data...")
matlab_path = os.path.join(SCRIPT_DIR, 'data/test_cases.mat')
with h5py.File(matlab_path, 'r') as f:
    u_fom = np.array(f['U_data']).T  # (Nx, Ny, Nt, N_samples)

Nx, Ny, Nt, N_samples = u_fom.shape
print(f"Data shape: {u_fom.shape}")

# ============================================================
# 2. LOAD SVD COEFFICIENTS (Sigma * VT)
# ============================================================
print("\n[2/4] Loading SVD data...")
svd_path = os.path.join(SCRIPT_DIR, 'data/svd_basis_data.npy')
svd_data = np.load(svd_path, allow_pickle=True).item()

Sigma = svd_data['singular_values']      # (p,)
VT = svd_data['coefficients']            # (p, N_samples)

# ============================================================
# 3. LOAD DEEPONET AND GET BRANCH PREDICTION
# ============================================================
print("\n[3/4] Loading DeepONet and getting branch prediction...")

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

u_ic = u_fom[:, :, 0, SAMPLE_IDX]
sensors = []
for si in sensor_x_indices:
    for sj in sensor_y_indices:
        sensors.append(u_ic[si, sj])
ic_sensors = np.array(sensors)

# Normalize IC to [-1, 1]
ic_norm = 2 * (ic_sensors - ic_min.squeeze()) / (ic_range.squeeze() + 1e-10) - 1
ic_tensor = torch.from_numpy(ic_norm).float().to(DEVICE).unsqueeze(0)

with torch.no_grad():
    branch_out = model.branch(ic_tensor).cpu().numpy().squeeze()  # (N_MODES,)

# ============================================================
# 4. COMPARE BRANCH OUTPUT VS SVD COEFFICIENTS
# ============================================================
print("\n[4/4] Creating comparison plot...")

svd_coeffs = Sigma[:N_MODES] * VT[:N_MODES, SAMPLE_IDX]

# Scale branch coefficients to match SVD coefficient scale (least-squares factor)
denom = np.dot(branch_out, branch_out) + 1e-12
scale = np.dot(branch_out, svd_coeffs) / denom
branch_scaled = branch_out * scale

# Sort by magnitude of SVD coefficients for consistent ordering
sort_idx = np.argsort(np.abs(svd_coeffs))[::-1]
svd_sorted = svd_coeffs[sort_idx]
branch_sorted = branch_scaled[sort_idx]

print(f"Branch coeff range (raw):    [{branch_out.min():.6f}, {branch_out.max():.6f}]")
print(f"Branch coeff range (scaled): [{branch_scaled.min():.6f}, {branch_scaled.max():.6f}]")
print(f"SVD coeff range:    [{svd_coeffs.min():.6f}, {svd_coeffs.max():.6f}]")
print(f"Scale factor (branch -> SVD): {scale:.6e}")

x = np.arange(N_MODES)
width = 0.4

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - width / 2, branch_sorted, width=width, label='Branch (scaled to SVD)')
ax.bar(x + width / 2, svd_sorted, width=width, label='SVD (Sigma * VT)')

ax.set_xlabel('Mode index (sorted by |SVD coeff|)')
ax.set_ylabel('Coefficient value')
ax.set_title(f'Branch vs SVD Coefficients (Sample {SAMPLE_IDX})')
ax.set_xticks(x)
ax.set_xticklabels([str(i) for i in x])
ax.grid(True, axis='y', alpha=0.3)
ax.legend()

plt.tight_layout()
save_path = os.path.join(SCRIPT_DIR, f'data/branch_vs_svd_coeffs_sample{SAMPLE_IDX}.png')
plt.savefig(save_path, dpi=150)
print(f"\n✓ Comparison plot saved to {save_path}")
plt.close()

# Correlation metric
corr = np.corrcoef(branch_out, svd_coeffs)[0, 1]
print(f"\nCorrelation (branch vs SVD): {corr:.6f}")

print("\n" + "=" * 70)
print("✓ COMPARISON COMPLETE")
print("=" * 70)
