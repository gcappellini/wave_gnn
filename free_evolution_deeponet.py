"""
Simple debugging script: compare different reconstruction methods
- Original FOM data
- SVD reconstruction
- Trunk * Branch (full DeepONet)
- Trunk * (SVD Sigma * coefficients)
- (SVD basis * Branch coefficients)
"""

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from src.models import DualHeadSensorBranch, DualHeadMLP


# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("=" * 70)
print("DEBUGGING: DeepONet Reconstruction Methods")
print("=" * 70)

data_dir = Path("data")
models_dir = Path("models")

print("\n1. Loading free_evolution.mat...")
with h5py.File(data_dir / "free_evolution.mat", 'r') as f:
    u_fom = np.array(f['U_data']).T  # (Nx, Ny, Nt, N_samples)
    v_fom = np.array(f['V_data']).T

Nx, Ny, Nt, N_samples = u_fom.shape
print(f"   u_fom shape: {u_fom.shape}")
print(f"   v_fom shape: {v_fom.shape}")

# ============================================================================
# 2. COMPUTE SVD DATA
# ============================================================================
print("\n2. Computing SVD decomposition...")

n_samples = u_fom.shape[3]

# Flatten spatial-temporal dimensions for SVD (use C order consistently)
u_flat = u_fom.reshape(Nx * Ny * Nt, n_samples, order='C')  # (Nx*Ny*Nt, N_samples)
v_flat = v_fom.reshape(Nx * Ny * Nt, n_samples, order='C')  # (Nx*Ny*Nt, N_samples)

print(f"   Computing SVD for u... shape={u_flat.shape}")
U_basis_full, Sigma_u_full, VT_u_full = np.linalg.svd(u_flat, full_matrices=False)

print(f"   Computing SVD for v... shape={v_flat.shape}")
U_basis_v_full, Sigma_v_full, VT_v_full = np.linalg.svd(v_flat, full_matrices=False)

print(f"   Full SVD computed. Will determine n_modes from trained models...")
print(f"   Full U_basis shape: {U_basis_full.shape}")
print(f"   Full Sigma_u shape: {Sigma_u_full.shape}")

# ============================================================================
# 3. LOAD TRUNK CHECKPOINT
# ============================================================================
print("\n3. Loading trunk checkpoint...")
trunk_ckpt_path = models_dir / "trunk_svd_free_evolution.pth"
trunk_ckpt = torch.load(str(trunk_ckpt_path), map_location='cpu', weights_only=False)
trunk_state = trunk_ckpt.get('model_state_dict', trunk_ckpt)
print(f"   Trunk loaded from {trunk_ckpt_path}")

# ============================================================================
# 4. LOAD BRANCH CHECKPOINT
# ============================================================================
print("\n4. Loading branch checkpoint...")
branch_ckpt_path = models_dir / "branch_svd_free_evolution.pth"
branch_ckpt = torch.load(str(branch_ckpt_path), map_location='cpu', weights_only=False)
branch_state = branch_ckpt.get('model_state_dict', branch_ckpt)
n_sensors = int(branch_ckpt.get('n_sensors', 0))
input_norm = branch_ckpt.get('input_normalization', {})

print(f"   Branch loaded from {branch_ckpt_path}")
print(f"   n_sensors: {n_sensors}")

raw_u_min = float(input_norm.get('raw_u_min', 0.0))
raw_u_max = float(input_norm.get('raw_u_max', 1.0))
raw_v_min = float(input_norm.get('raw_v_min', 0.0))
raw_v_max = float(input_norm.get('raw_v_max', 1.0))

# Determine n_modes from trained trunk
n_modes = int(trunk_state['head_u.weight'].shape[0])
print(f"\n   ✓ Determined n_modes from trunk: {n_modes}")

# Now truncate full SVD to match trained n_modes
U_basis = U_basis_full[:, :n_modes]
Sigma_u = Sigma_u_full[:n_modes]
VT_u = VT_u_full[:n_modes, :]

U_basis_v = U_basis_v_full[:, :n_modes]
Sigma_v = Sigma_v_full[:n_modes]
VT_v = VT_v_full[:n_modes, :]

print(f"   Truncated SVD basis to n_modes={n_modes}")
print(f"   U_basis shape: {U_basis.shape}")
print(f"   Sigma_u shape: {Sigma_u.shape}")

# ============================================================================
# 5. SELECT SAMPLE AND TIME SNAPSHOT
# ============================================================================
print("\n5. Selecting sample and time snapshot...")
sample_idx = 0
t_idx = 0  # t=0

u_sample = u_fom[:, :, t_idx, sample_idx]  # (Nx, Ny)
v_sample = v_fom[:, :, t_idx, sample_idx]  # (Nx, Ny)

print(f"   Sample: {sample_idx}, t_idx: {t_idx}")
print(f"   u_sample range: [{u_sample.min():.4e}, {u_sample.max():.4e}]")
print(f"   v_sample range: [{v_sample.min():.4e}, {v_sample.max():.4e}]")

# ============================================================================
# 6. SVD RECONSTRUCTION (Ground Truth)
# ============================================================================
print("\n6. Computing SVD reconstruction...")

# Compute spatial indices for time slice t_idx (using C order to match SVD reshape)
spatial_indices = np.arange(Nx * Ny * Nt).reshape(Nx, Ny, Nt, order='C')[:, :, t_idx].flatten('C')

u_coeffs_svd = VT_u[:, sample_idx]  # Coefficients are already in VT
v_coeffs_svd = VT_v[:, sample_idx]

u_svd_recon = U_basis[spatial_indices, :] @ (Sigma_u * u_coeffs_svd)
v_svd_recon = U_basis_v[spatial_indices, :] @ (Sigma_v * v_coeffs_svd)

u_svd_recon = u_svd_recon.reshape(Nx, Ny, order='C')
v_svd_recon = v_svd_recon.reshape(Nx, Ny, order='C')

svd_error_u = np.sqrt(np.mean((u_sample - u_svd_recon)**2))
svd_error_v = np.sqrt(np.mean((v_sample - v_svd_recon)**2))
print(f"   SVD reconstruction error u: {svd_error_u:.4e}")
print(f"   SVD reconstruction error v: {svd_error_v:.4e}")

# Debug: check correlation
corr_u = np.corrcoef(u_sample.flatten(), u_svd_recon.flatten())[0, 1]
corr_v = np.corrcoef(v_sample.flatten(), v_svd_recon.flatten())[0, 1]
print(f"   SVD correlation with original u: {corr_u:.4f}")
print(f"   SVD correlation with original v: {corr_v:.4f}")
print(f"   SVD u range: [{u_svd_recon.min():.4e}, {u_svd_recon.max():.4e}]")
print(f"   SVD v range: [{v_svd_recon.min():.4e}, {v_svd_recon.max():.4e}]")

# ============================================================================
# 7. EXTRACT SENSOR MEASUREMENTS AND NORMALIZE
# ============================================================================
print("\n7. Extracting and normalizing sensor measurements...")
sensor_x = np.linspace(0, Nx - 1, n_sensors, dtype=int)
sensor_y = np.linspace(0, Ny - 1, n_sensors, dtype=int)

u_ic = u_fom[:, :, 0, sample_idx]
u_meas_raw = np.array([u_ic[si, sj] for si in sensor_x for sj in sensor_y], dtype=np.float32)
u_meas_norm = 2 * (u_meas_raw - raw_u_min) / (raw_u_max - raw_u_min + 1e-10) - 1

v_ic = v_fom[:, :, 0, sample_idx]
v_meas_raw = np.array([v_ic[si, sj] for si in sensor_x for sj in sensor_y], dtype=np.float32)
v_meas_norm = 2 * (v_meas_raw - raw_v_min) / (raw_v_max - raw_v_min + 1e-10) - 1

meas = np.concatenate([u_meas_norm, v_meas_norm]).astype(np.float32)
print(f"   meas shape: {meas.shape}")

# ============================================================================
# 8. RUN BRANCH FORWARD PASS
# ============================================================================
print("\n8. Running branch network...")
device = torch.device('cpu')

branch = DualHeadSensorBranch(
    n_sensors=n_sensors,
    hidden_dim=int(branch_state['mlp.backbone.0.weight'].shape[0]),
    n_modes=int(branch_state['mlp.head_u.weight'].shape[0]),
    n_layers=len([k for k in branch_state.keys() if k.startswith('mlp.backbone.') and k.endswith('.weight')]) + 1,
    input_scale=1.0,
    encoder_channels=int(branch_state['encoder.0.weight'].shape[0]),
)
branch.load_state_dict(branch_state)
branch.eval()

meas_tensor = torch.from_numpy(meas).float().unsqueeze(0)
with torch.no_grad():
    branch_out = branch(meas_tensor)

coeffs_u_branch, coeffs_v_branch = branch_out[0].numpy()[0], branch_out[1].numpy()[0]
print(f"   Branch output shapes: u={coeffs_u_branch.shape}, v={coeffs_v_branch.shape}")

# ============================================================================
# 9. RUN TRUNK FORWARD PASS
# ============================================================================
print("\n9. Running trunk network (outputs basis functions)...")

trunk = DualHeadMLP(
    input_dim=int(trunk_state['backbone.0.weight'].shape[1]),
    hidden_dim=int(trunk_state['backbone.0.weight'].shape[0]),
    n_modes=int(trunk_state['head_u.weight'].shape[0]),
    n_layers=len([k for k in trunk_state.keys() if k.startswith('backbone.') and k.endswith('.weight')]) + 1,
)
trunk.load_state_dict(trunk_state)
trunk.eval()

x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
t = np.linspace(0, 1, Nt)
t_val = t[t_idx]

X, Y = np.meshgrid(x, y, indexing='ij')
coords = np.stack([
    X.flatten('C'),
    Y.flatten('C'),
    np.full((Nx * Ny,), t_val)
], axis=1).astype(np.float32)

coords_tensor = torch.from_numpy(coords).float()
with torch.no_grad():
    trunk_out = trunk(coords_tensor)

# Trunk outputs basis functions (NOT coefficients!)
trunk_basis_u = trunk_out[0].numpy()  # (Nx*Ny, n_modes) — basis functions
trunk_basis_v = trunk_out[1].numpy()  # (Nx*Ny, n_modes) — basis functions
print(f"   Trunk output shapes (basis functions): u={trunk_basis_u.shape}, v={trunk_basis_v.shape}")
print(f"   Trunk basis_u range: [{trunk_basis_u.min():.4e}, {trunk_basis_u.max():.4e}]")
print(f"   Trunk basis_v range: [{trunk_basis_v.min():.4e}, {trunk_basis_v.max():.4e}]")

# ============================================================================
# 10. COMPUTE RECONSTRUCTIONS: DIFFERENT METHODS
# ============================================================================
print("\n10. Computing different reconstruction methods...")

# Method 1: Full DeepONet = Trunk basis @ Branch coefficients
u_m1_flat = trunk_basis_u @ coeffs_u_branch  # (Nx*Ny,)
v_m1_flat = trunk_basis_v @ coeffs_v_branch  # (Nx*Ny,)
u_m1 = u_m1_flat.reshape(Nx, Ny, order='C')
v_m1 = v_m1_flat.reshape(Nx, Ny, order='C')
print(f"   Method 1 (Full DeepONet): Trunk_basis @ Branch_coeffs")

# Method 2: Trunk basis with ground-truth SVD coefficients
# = Trunk basis @ (Sigma * SVD_coeff_gt)
u_m2_flat = trunk_basis_u @ (Sigma_u * u_coeffs_svd)  # (Nx*Ny,)
v_m2_flat = trunk_basis_v @ (Sigma_v * v_coeffs_svd)  # (Nx*Ny,)
u_m2 = u_m2_flat.reshape(Nx, Ny, order='C')
v_m2 = v_m2_flat.reshape(Nx, Ny, order='C')
print(f"   Method 2 (Trunk basis + GT SVD coeff): Trunk_basis @ (Sigma * SVD_coeff)")

# Method 3: True SVD basis with Branch coefficients
# = True_SVD_basis @ Branch_coeffs
u_m3_flat = U_basis[spatial_indices, :] @ (Sigma_u * u_coeffs_svd)  # coeffs_u_branch  # (Nx*Ny,)
v_m3_flat = U_basis_v[spatial_indices, :] @ (Sigma_v * v_coeffs_svd) # coeffs_v_branch  # (Nx*Ny,)
u_m3 = u_m3_flat.reshape(Nx, Ny, order='C')
v_m3 = v_m3_flat.reshape(Nx, Ny, order='C')
print(f"   Method 3 (True SVD basis + Branch coeffs): True_SVD_basis @ Branch_coeffs")


print("   Methods computed")

print(Sigma_u)
print(coeffs_u_branch)

# ============================================================================
# 11. PLOT ALL METHODS
# ============================================================================
print("\n11. Plotting all methods...")

methods = [
    ("Original Data", u_sample, v_sample),
    ("SVD Reconstruction", u_svd_recon, v_svd_recon),
    ("Method 1: Trunk * Branch", u_m1, v_m1),
    ("Method 2: Trunk * SVD Sigma*Coeff", u_m2, v_m2),
    ("Method 3: SVD basis * Branch", u_m3, v_m3),
]

fig, axes = plt.subplots(len(methods), 2, figsize=(10, 4 * len(methods)))

for row, (title, u_data, v_data) in enumerate(methods):
    im_u = axes[row, 0].imshow(u_data, cmap='RdBu_r', aspect='auto')
    axes[row, 0].set_title(f"{title} (u)")
    axes[row, 0].set_xlabel("y")
    axes[row, 0].set_ylabel("x")
    plt.colorbar(im_u, ax=axes[row, 0])
    
    im_v = axes[row, 1].imshow(v_data, cmap='RdBu_r', aspect='auto')
    axes[row, 1].set_title(f"{title} (v)")
    axes[row, 1].set_xlabel("y")
    axes[row, 1].set_ylabel("x")
    plt.colorbar(im_v, ax=axes[row, 1])

plt.suptitle(f"DeepONet Reconstruction Methods (t=0, sample {sample_idx})", fontsize=14, y=0.995)
plt.tight_layout()

output_path = Path("outputs") / "debug_deeponet_methods.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: {output_path}")

# ============================================================================
# 12. COMPUTE ERRORS vs ORIGINAL
# ============================================================================
print("\n12. Computing L2 errors vs original...")

def l2_error(gt, pred):
    return np.sqrt(np.mean((gt - pred) ** 2))

errors_u = {
    'SVD': l2_error(u_sample, u_svd_recon),
    'Trunk * Branch': l2_error(u_sample, u_m1),
    'Trunk * SVD Coeff': l2_error(u_sample, u_m2),
    'SVD basis * Branch': l2_error(u_sample, u_m3),
}

errors_v = {
    'SVD': l2_error(v_sample, v_svd_recon),
    'Trunk * Branch': l2_error(v_sample, v_m1),
    'Trunk * SVD Coeff': l2_error(v_sample, v_m2),
    'SVD basis * Branch': l2_error(v_sample, v_m3),
}

print("\n   L2 Errors (u):")
for method, error in errors_u.items():
    print(f"      {method:30s}: {error:.4e}")

print("\n   L2 Errors (v):")
for method, error in errors_v.items():
    print(f"      {method:30s}: {error:.4e}")

print("\n" + "=" * 70)
print("✓ Debug script complete!")
print("=" * 70)