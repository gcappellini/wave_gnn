"""
Visualize SVD modes and activation coefficients for sample 0.

Loads pre-computed SVD data and shows:
1. Individual basis modes (spatiotemporal functions)
2. Scalar coefficients that activate each mode for sample 0
3. How modes combine to reconstruct the solution
"""

import os
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SVD_DATA_PATH = os.path.join(SCRIPT_DIR, 'data/svd_basis_data.npy')

if not os.path.exists(SVD_DATA_PATH):
    raise FileNotFoundError(f"SVD data not found: {SVD_DATA_PATH}")

# Load pre-computed SVD data
print(f"Loading SVD data from {SVD_DATA_PATH}...")
svd_data = np.load(SVD_DATA_PATH, allow_pickle=True).item()

U_basis = svd_data['basis']              # (Nx*Ny*Nt, p)
Sigma = svd_data['singular_values']     # (p,)
VT = svd_data['coefficients']            # (p, N_samples)
grid_info = svd_data['grid_info']        # [Nx, Ny, Nt]

Nx, Ny, Nt = int(grid_info[0]), int(grid_info[1]), int(grid_info[2])
actual_modes = U_basis.shape[1]
N_samples = VT.shape[1]

print(f"Loaded SVD decomposition:")
print(f"  Modes (U_basis): {U_basis.shape}")
print(f"  Singular values: {Sigma.shape}")
print(f"  Coefficients (VT): {VT.shape}")
print(f"  Grid: Nx={Nx}, Ny={Ny}, Nt={Nt}")
print(f"  Actual modes: {actual_modes}")
print(f"  N_samples: {N_samples}")

# ============================================================
# 1. VISUALIZE FIRST 18 SVD MODES
# ============================================================
print("\n[1/4] Visualizing first 18 modes at middle time step...")

# Reshape modes back to spatiotemporal: (Nx, Ny, Nt, actual_modes)
# Use Fortran order to match original flattening
modes_reshaped = U_basis.reshape(Nx, Ny, Nt, actual_modes, order='F')

# Choose a time slice to visualize (middle of simulation)
t_idx = Nt // 2
n_modes_to_plot = min(18, actual_modes)

fig, axes = plt.subplots(6, 3, figsize=(15, 30))
for i in range(n_modes_to_plot):
    row = i // 3
    col = i % 3
    mode_slice = modes_reshaped[:, :, t_idx, i]
    im = axes[row, col].imshow(mode_slice, cmap='seismic', origin='lower')
    axes[row, col].set_title(f"SVD Mode {i} (t={t_idx}/{Nt-1})")
    axes[row, col].set_xticks([])
    axes[row, col].set_yticks([])
    plt.colorbar(im, ax=axes[row, col], fraction=0.046)

# Hide unused subplots if we have fewer than 18 modes
for i in range(n_modes_to_plot, 18):
    row = i // 3
    col = i % 3
    axes[row, col].axis('off')

plt.suptitle(f"SVD Basis Functions ({actual_modes} modes) at t={t_idx}/{Nt-1}", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'data/modes_visualization.png'), dpi=150)
print(f"  Saved to data/modes_visualization.png")
plt.close()

# ============================================================
# 2. VISUALIZE ACTIVATION COEFFICIENTS FOR SAMPLE 0
# ============================================================
print("\n[2/4] Visualizing activation coefficients for sample 0...")

sample_idx = 0
coeff_sample = VT[:actual_modes, sample_idx]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Bar chart of coefficient magnitudes
ax1 = axes[0, 0]
n_coeff_show = min(30, actual_modes)
colors = ['steelblue' if c >= 0 else 'coral' for c in coeff_sample[:n_coeff_show]]
ax1.bar(range(n_coeff_show), coeff_sample[:n_coeff_show], color=colors)
ax1.set_xlabel('Mode Index', fontsize=11)
ax1.set_ylabel('Coefficient Value', fontsize=11)
ax1.set_title(f'SVD Coefficients for Sample {sample_idx}\n(How much each mode is "activated")', fontsize=12)
ax1.grid(True, alpha=0.3, axis='y')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Plot 2: Absolute value of coefficients
ax2 = axes[0, 1]
ax2.bar(range(n_coeff_show), np.abs(coeff_sample[:n_coeff_show]), color='steelblue', alpha=0.7)
ax2.set_xlabel('Mode Index', fontsize=11)
ax2.set_ylabel('|Coefficient|', fontsize=11)
ax2.set_title(f'Magnitude of Coefficients (Sample {sample_idx})\nShows which modes contribute most', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Cumulative contribution
ax3 = axes[1, 0]
sorted_indices = np.argsort(np.abs(coeff_sample))[::-1]
sorted_coeffs = np.abs(coeff_sample[sorted_indices])
cumulative_sum = np.cumsum(sorted_coeffs)
cumulative_pct = (cumulative_sum / cumulative_sum[-1]) * 100
ax3.plot(cumulative_pct[:50], linewidth=2.5, color='darkgreen', marker='o', markersize=4)
ax3.axhline(y=90, color='red', linestyle='--', linewidth=1.5, label='90%')
ax3.axhline(y=99, color='orange', linestyle='--', linewidth=1.5, label='99%')
ax3.set_xlabel('Number of Largest Modes', fontsize=11)
ax3.set_ylabel('Cumulative Contribution (%)', fontsize=11)
ax3.set_title(f'Cumulative Mode Contribution (Sample {sample_idx})\nHow many modes needed to explain solution?', fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)
ax3.set_ylim([0, 105])

# Plot 4: Top 10 modes
ax4 = axes[1, 1]
top_n = min(10, actual_modes)
top_indices = sorted_indices[:top_n]
top_values = coeff_sample[top_indices]
bars = ax4.barh(range(top_n), top_values, color='steelblue')
ax4.set_yticks(range(top_n))
ax4.set_yticklabels([f'Mode {i}' for i in top_indices])
ax4.set_xlabel('Coefficient Value', fontsize=11)
ax4.set_title(f'Top {top_n} Most Active Modes (Sample {sample_idx})', fontsize=12)
ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'data/coefficients_sample0.png'), dpi=150)
print(f"  Saved to data/coefficients_sample0.png")
plt.close()

# ============================================================
# 3. VISUALIZE MODE TEMPORAL EVOLUTION
# ============================================================
print("\n[3/4] Visualizing temporal evolution of modes...")

center_x, center_y = Nx // 2, Ny // 2

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: First 4 modes temporal evolution at center point
ax1 = axes[0, 0]
for mode_idx in range(min(4, actual_modes)):
    mode_at_center = modes_reshaped[center_x, center_y, :, mode_idx]
    ax1.plot(mode_at_center, marker='o', label=f'Mode {mode_idx}', linewidth=2.5, markersize=4)
ax1.set_xlabel('Time Step', fontsize=11)
ax1.set_ylabel('Mode Value', fontsize=11)
ax1.set_title(f'Modes 0-3 Temporal Evolution\nat center point ({center_x}, {center_y})', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Modes 5-8 temporal evolution
ax2 = axes[0, 1]
for mode_idx in range(4, min(8, actual_modes)):
    mode_at_center = modes_reshaped[center_x, center_y, :, mode_idx]
    ax2.plot(mode_at_center, marker='s', label=f'Mode {mode_idx}', linewidth=2.5, markersize=4)
ax2.set_xlabel('Time Step', fontsize=11)
ax2.set_ylabel('Mode Value', fontsize=11)
ax2.set_title(f'Modes 4-7 Temporal Evolution\nat center point ({center_x}, {center_y})', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Mode spatial structure at t=0
ax3 = axes[1, 0]
mode_0_spatial = modes_reshaped[:, :, 0, 0]
im3 = ax3.imshow(mode_0_spatial, cmap='seismic', origin='lower')
ax3.set_title('Mode 0 Spatial Structure\nat t=0', fontsize=12)
ax3.set_xticks([])
ax3.set_yticks([])
plt.colorbar(im3, ax=ax3, fraction=0.046)

# Plot 4: Mode spatial structure at t=Nt//2
ax4 = axes[1, 1]
mode_0_mid = modes_reshaped[:, :, Nt//2, 0]
im4 = ax4.imshow(mode_0_mid, cmap='seismic', origin='lower')
ax4.set_title(f'Mode 0 Spatial Structure\nat t={Nt//2}/{Nt-1}', fontsize=12)
ax4.set_xticks([])
ax4.set_yticks([])
plt.colorbar(im4, ax=ax4, fraction=0.046)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'data/modes_temporal_evolution.png'), dpi=150)
print(f"  Saved to data/modes_temporal_evolution.png")
plt.close()

# ============================================================
# 4. VISUALIZE RECONSTRUCTION FOR SAMPLE 0
# ============================================================
print("\n[4/4] Visualizing reconstruction for sample 0...")

# Reconstruct solution: u = U_basis @ (Sigma * VT[:, sample_idx])
reconstructed_coeffs = Sigma[:actual_modes] * VT[:actual_modes, sample_idx]
u_flat = U_basis @ reconstructed_coeffs

# Reshape back to (Nx, Ny, Nt)
u_reconstructed = u_flat.reshape(Nx, Ny, Nt, order='F')

# Extract temporal evolution at center and spatially averaged
u_center = u_reconstructed[center_x, center_y, :]
u_avg = np.mean(u_reconstructed, axis=(0, 1))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Temporal evolution at center
ax1 = axes[0, 0]
ax1.plot(u_center, linewidth=2.5, color='steelblue', marker='o', markersize=5)
ax1.set_xlabel('Time Step', fontsize=11)
ax1.set_ylabel('Solution Value', fontsize=11)
ax1.set_title(f'Reconstructed Solution at Center\nSample {sample_idx}, ({center_x}, {center_y})', fontsize=12)
ax1.grid(True, alpha=0.3)

# Plot 2: Spatially-averaged temporal evolution
ax2 = axes[0, 1]
ax2.plot(u_avg, linewidth=2.5, color='coral', marker='s', markersize=5)
ax2.set_xlabel('Time Step', fontsize=11)
ax2.set_ylabel('Spatially-Averaged Value', fontsize=11)
ax2.set_title(f'Reconstructed Solution (Spatial Average)\nSample {sample_idx}', fontsize=12)
ax2.grid(True, alpha=0.3)

# Plot 3: Spatial snapshot at t=0
ax3 = axes[1, 0]
im3 = ax3.imshow(u_reconstructed[:, :, 0], cmap='seismic', origin='lower')
ax3.set_title(f'Reconstructed Solution at t=0', fontsize=12)
ax3.set_xticks([])
ax3.set_yticks([])
plt.colorbar(im3, ax=ax3, fraction=0.046)

# Plot 4: Spatial snapshot at t=Nt//2
ax4 = axes[1, 1]
im4 = ax4.imshow(u_reconstructed[:, :, Nt//2], cmap='seismic', origin='lower')
ax4.set_title(f'Reconstructed Solution at t={Nt//2}/{Nt-1}', fontsize=12)
ax4.set_xticks([])
ax4.set_yticks([])
plt.colorbar(im4, ax=ax4, fraction=0.046)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'data/reconstruction_sample0.png'), dpi=150)
print(f"  Saved to data/reconstruction_sample0.png")
plt.close()

# ============================================================
# SUMMARY STATISTICS
# ============================================================
print("\n" + "="*70)
print("SUMMARY FOR SAMPLE 0")
print("="*70)

print(f"\nModes (U_basis): shape {U_basis.shape}")
print(f"  → Reshaped to (Nx={Nx}, Ny={Ny}, Nt={Nt}, modes={actual_modes})")
print(f"  → Each mode is a 4D spatiotemporal field")
print(f"  → SHARED across all {N_samples} samples (characteristic of the problem)")

print(f"\nCoefficients (VT): shape {VT.shape}")
print(f"  → VT[i, {sample_idx}] = activation coefficient for mode i in sample {sample_idx}")
print(f"  → Sample {sample_idx} has {actual_modes} scalar coefficients")
print(f"  → These VARY across different initial conditions")

top_3_idx = np.argsort(np.abs(coeff_sample))[::-1][:3]
print(f"\nTop 3 most active modes for sample {sample_idx}:")
for rank, mode_idx in enumerate(top_3_idx, 1):
    print(f"  {rank}. Mode {mode_idx}: coefficient = {coeff_sample[mode_idx]:.6f}")

print(f"\nReconstruction formula for sample {sample_idx}:")
print(f"  u[x,y,t] = Σᵢ VT[i,{sample_idx}] × Σᵢ × U_basis[x,y,t,i]")
print(f"  where:")
print(f"    - VT[i,{sample_idx}] is a scalar (activation)")
print(f"    - Σᵢ is singular value (importance)")
print(f"    - U_basis[x,y,t,i] is a field (mode shape)")

print("\n" + "="*70)
print("Visualizations saved to data/:")
print("  - modes_visualization.png")
print("  - coefficients_sample0.png")
print("  - modes_temporal_evolution.png")
print("  - reconstruction_sample0.png")
print("="*70)
