"""
Minimal Evaluation: Load Trunk + Branch Predictions, Compute Dot Product
- Load trunk predictions (trunk_sample0_predictions.npz) - full spatial grid at 5 times
- Load branch predictions (branch_sample0_predictions_*.npz)
- Compute dot product: u(x,y,t) = Σᵢ branch_i × trunk_i(x,y,t)
- Compare against ground truth for sample 0
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import h5py

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 70)
print("MINIMAL DEEPONET EVALUATION (SAMPLE 0, FULL SPATIAL GRID)")
print("=" * 70)

# ============================================================
# 1. LOAD TRUNK PREDICTIONS (FULL SPATIAL GRID AT 5 TIMES)
# ============================================================
print("\n1. Loading trunk predictions...")

trunk_pred_path = os.path.join(SCRIPT_DIR, 'data/trunk_sample0_predictions.npz')
if not os.path.exists(trunk_pred_path):
    raise FileNotFoundError(f"Trunk predictions not found: {trunk_pred_path}")

trunk_data = np.load(trunk_pred_path)
trunk_basis = trunk_data['predicted_basis']  # (5, Nx*Ny, N_MODES)
trunk_gt_basis = trunk_data['ground_truth_basis']  # (5, Nx*Ny, N_MODES)
time_instants = trunk_data['time_instants']  # (5,)
Nx, Ny = trunk_data['grid_shape']

n_times, n_spatial, N_MODES = trunk_basis.shape
print(f"✓ Trunk predictions loaded: {trunk_basis.shape}")
print(f"  Grid: {Nx}×{Ny} = {n_spatial} spatial points")
print(f"  Time instants: {time_instants}")
print(f"  Modes: {N_MODES}")

# ============================================================
# 2. LOAD BRANCH PREDICTIONS (COEFFICIENTS FOR SAMPLE 0)
# ============================================================
print("\n2. Loading branch predictions...")

checkpoint_dir = os.path.join(SCRIPT_DIR, 'data')
branch_pred_files = sorted(Path(checkpoint_dir).glob('branch_sample0_predictions_*.npz'))

if not branch_pred_files:
    raise FileNotFoundError(f"No branch predictions found in {checkpoint_dir}")

branch_pred_path = branch_pred_files[-1]
branch_data = np.load(branch_pred_path)
branch_coeffs = branch_data['predicted_coefficients']  # (N_MODES,)
branch_gt_coeffs = branch_data['ground_truth_coefficients']  # (N_MODES,)

print(f"✓ Branch predictions loaded: {branch_coeffs.shape}")
print(f"  From: {branch_pred_path.name}")

# ============================================================
# 3. COMPUTE DEEPONET PREDICTION (DOT PRODUCT)
# ============================================================
print("\n3. Computing DeepONet predictions (dot product)...")

# Load SVD singular values (Sigma) for proper reconstruction
svd_data_path = os.path.join(SCRIPT_DIR, 'data/svd_basis_data.npy')
svd_data = np.load(svd_data_path, allow_pickle=True).item()
Sigma = svd_data['singular_values'][:N_MODES]  # (N_MODES,)

print(f"  Loaded Sigma: {Sigma.shape}")

# DeepONet: u(x,y,t) = Σᵢ (Sigma_i * branch_i) × trunk_i(x,y,t)
# SVD reconstruction requires: U_basis @ (Sigma * VT)
# trunk_basis: (5, Nx*Ny, N_MODES) - already represents U_basis values
# branch_coeffs: (N_MODES,) - represents VT values
# We need to multiply branch coefficients by Sigma!

branch_coeffs_scaled = Sigma * branch_coeffs  # (N_MODES,)
branch_gt_coeffs_scaled = Sigma * branch_gt_coeffs  # (N_MODES,)

u_deeponet = np.einsum('tsi,i->ts', trunk_basis, branch_coeffs_scaled)  # (5, Nx*Ny) - PREDICTIONS
u_gt = np.einsum('tsi,i->ts', trunk_gt_basis, branch_gt_coeffs_scaled)  # (5, Nx*Ny) - GROUND TRUTH

print(f"✓ DeepONet prediction shape: {u_deeponet.shape}")
print(f"  Prediction range: [{u_deeponet.min():.6e}, {u_deeponet.max():.6e}]")
print(f"  Ground truth range: [{u_gt.min():.6e}, {u_gt.max():.6e}]")

# ============================================================
# 4. LOAD GROUND TRUTH SOLUTION FROM MATLAB
# ============================================================
print("\n4. Loading ground truth solution from MATLAB...")

matlab_data_path = os.path.join(SCRIPT_DIR, 'data/test_cases.mat')
if not os.path.exists(matlab_data_path):
    raise FileNotFoundError(f"MATLAB data not found: {matlab_data_path}")

# MATLAB v7.3 files use HDF5 format, need h5py
with h5py.File(matlab_data_path, 'r') as f:
    # h5py reads MATLAB arrays in transposed format, so we transpose back
    u_fom = np.array(f['U_data']).T  # Now shape: (Nx, Ny, Nt, N_samples)

u_sample0 = u_fom[:, :, :, 0]  # (Nx, Ny, Nt) - sample 0

# Extract at the 5 time instants
time_indices = trunk_data['time_indices']
u_gt_matlab = np.zeros((len(time_indices), Nx * Ny))
for i, t_idx in enumerate(time_indices):
    u_gt_matlab[i, :] = u_sample0[:, :, t_idx].flatten('F')  # (Nx*Ny,)

print(f"✓ Ground truth from MATLAB: {u_gt_matlab.shape}")

# ============================================================
# 5. COMPUTE ERRORS
# ============================================================
print("\n5. Computing errors...")

# Use MATLAB ground truth
abs_error = np.abs(u_deeponet - u_gt_matlab)
rel_error = abs_error / (np.abs(u_gt_matlab) + 1e-12)

mae_overall = abs_error.mean()
max_err_overall = abs_error.max()
mre_overall = rel_error.mean()

print(f"  Overall Mean Absolute Error: {mae_overall:.6e}")
print(f"  Overall Max Absolute Error: {max_err_overall:.6e}")
print(f"  Overall Mean Relative Error: {mre_overall:.6f}")

print(f"\nPer-time statistics:")
print(f"{'Time':<8} {'MAE':<15} {'Max Error':<15} {'Mean Rel Err':<15}")
print("-" * 60)
for i, t in enumerate(time_instants):
    mae_t = abs_error[i].mean()
    max_t = abs_error[i].max()
    mre_t = rel_error[i].mean()
    print(f"{t:<8.2f} {mae_t:<15.6e} {max_t:<15.6e} {mre_t:<15.6f}")

# ============================================================
# 6. VISUALIZATION
# ============================================================
print("\n6. Creating visualizations...")

# Create 5x3 grid: GT, Prediction, Error for each time instant
fig, axes = plt.subplots(len(time_instants), 3, figsize=(15, 4 * len(time_instants)))

for i, t in enumerate(time_instants):
    # Reshape to 2D spatial grids
    gt_2d = u_gt_matlab[i].reshape(Nx, Ny, order='F')
    pred_2d = u_deeponet[i].reshape(Nx, Ny, order='F')
    error_2d = abs_error[i].reshape(Nx, Ny, order='F')
    
    # Find common colorbar range
    vmin = min(gt_2d.min(), pred_2d.min())
    vmax = max(gt_2d.max(), pred_2d.max())
    
    # Ground Truth
    im0 = axes[i, 0].imshow(gt_2d, cmap='seismic', origin='lower', vmin=vmin, vmax=vmax)
    axes[i, 0].set_title(f'Ground Truth (t={t:.2f})', fontsize=11)
    axes[i, 0].set_ylabel('x', fontsize=10)
    axes[i, 0].set_xlabel('y', fontsize=10)
    plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)
    
    # Prediction
    im1 = axes[i, 1].imshow(pred_2d, cmap='seismic', origin='lower', vmin=vmin, vmax=vmax)
    axes[i, 1].set_title(f'DeepONet Prediction (t={t:.2f})', fontsize=11)
    axes[i, 1].set_ylabel('x', fontsize=10)
    axes[i, 1].set_xlabel('y', fontsize=10)
    plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)
    
    # Error
    im2 = axes[i, 2].imshow(error_2d, cmap='hot', origin='lower')
    axes[i, 2].set_title(f'Absolute Error (t={t:.2f})', fontsize=11)
    axes[i, 2].set_ylabel('x', fontsize=10)
    axes[i, 2].set_xlabel('y', fontsize=10)
    plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)

plt.tight_layout()
save_path = os.path.join(SCRIPT_DIR, 'data/deeponet_minimal_evaluation.png')
plt.savefig(save_path, dpi=150)
print(f"✓ Visualization saved to {save_path}")
plt.close()

print("\n" + "=" * 70)
print("✓ EVALUATION COMPLETE")
print("=" * 70)
