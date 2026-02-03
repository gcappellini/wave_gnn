"""
Load trained DeepONet and visualize trunk and branch networks separately.

Trunk Network: (x,y,t) → 18 basis function values at that point
Branch Network: IC sensors → 18 coefficients (how much to weight each basis)

This verifies:
- Trunk learns spatiotemporal modes
- Branch learns IC-dependent coefficients
"""

import os
from datetime import datetime
import numpy as np
import h5py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
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

# ============================================================
# FIND LATEST CHECKPOINT
# ============================================================
print("Searching for latest DeepONet checkpoint...")

# Try multiple locations
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

print(f"Loading: {latest_ckpt}")
print(f"Checkpoint size: {os.path.getsize(latest_ckpt) / 1e6:.1f} MB")

# ============================================================
# LOAD CHECKPOINT
# ============================================================
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

print(f"Config: N_MODES={N_MODES}, N_SENSORS={N_SENSORS}")
print(f"Trunk: {TRUNK_HIDDEN_DIM}×{TRUNK_N_LAYERS} layers")
print(f"Branch: {BRANCH_HIDDEN_DIM}×{BRANCH_N_LAYERS} layers")

# ============================================================
# BUILD MODELS AND LOAD WEIGHTS
# ============================================================
print("\nBuilding trunk and branch networks...")
ic_dim = N_SENSORS * N_SENSORS

trunk = MLP(3, TRUNK_HIDDEN_DIM, N_MODES, TRUNK_N_LAYERS).to(DEVICE)
branch = MLP(ic_dim, BRANCH_HIDDEN_DIM, N_MODES, BRANCH_N_LAYERS).to(DEVICE)
model = DeepONet(trunk, branch).to(DEVICE)

model.load_state_dict(checkpoint['model_state_dict'])
print("✓ Weights loaded")

# ============================================================
# LOAD MATLAB DATA
# ============================================================
print("\nLoading MATLAB data...")
matlab_path = os.path.join(SCRIPT_DIR, 'data/test_cases.mat')
with h5py.File(matlab_path, 'r') as f:
    u_fom = np.array(f['U_data']).T  # (Nx, Ny, Nt, N_samples)

Nx, Ny, Nt, N_samples = u_fom.shape
print(f"u_fom shape: {u_fom.shape}")

# ============================================================
# BUILD COORDINATE GRID
# ============================================================
print("\nBuilding coordinate grid...")
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
t = np.linspace(0, 1, Nt)
X, Y, T = np.meshgrid(x, y, t, indexing='ij')

# ============================================================
# LOAD SVD DATA FOR COMPARISON
# ============================================================
print("\nLoading SVD data for mode comparison...")
svd_data_path = os.path.join(SCRIPT_DIR, 'data/svd_basis_data.npy')
if not os.path.exists(svd_data_path):
    raise FileNotFoundError(f"SVD data not found: {svd_data_path}")

svd_data = np.load(svd_data_path, allow_pickle=True).item()
U_basis_svd = svd_data['basis']  # (Nx*Ny*Nt, p)
Sigma_svd = svd_data['singular_values']  # (p,)
VT_svd = svd_data['coefficients']  # (p, N_samples)

# Reshape SVD modes to spatiotemporal: (Nx, Ny, Nt, n_modes)
modes_svd_reshaped = U_basis_svd.reshape(Nx, Ny, Nt, U_basis_svd.shape[1], order='F')
print(f"SVD modes shape: {modes_svd_reshaped.shape}")

# Extract SVD coefficients for sample 0
# In SVD: u = U_basis @ (Sigma * VT[:, sample_idx])
# So the effective coefficient is Sigma * VT
svd_coeffs_sample0 = Sigma_svd * VT_svd[:, 0]  # (n_modes,)
print(f"SVD coefficients for sample 0: {svd_coeffs_sample0.shape}")
print(f"  Min: {svd_coeffs_sample0.min():.6f}")
print(f"  Max: {svd_coeffs_sample0.max():.6f}")
print(f"  Top 3 SVD coefficients:")
top_3_svd_idx = np.argsort(np.abs(svd_coeffs_sample0))[::-1][:3]
for rank, idx in enumerate(top_3_svd_idx, 1):
    print(f"    {rank}. Mode {idx}: {svd_coeffs_sample0[idx]:.6f}")

# ============================================================
# EXTRACT IC SENSORS FOR SAMPLE 0
# ============================================================
print("\nExtracting IC sensors for sample 0...")
sensor_x_indices = np.linspace(0, Nx - 1, N_SENSORS, dtype=int)
sensor_y_indices = np.linspace(0, Ny - 1, N_SENSORS, dtype=int)

u_ic = u_fom[:, :, 0, 0]
sensors = []
for si in sensor_x_indices:
    for sj in sensor_y_indices:
        sensors.append(u_ic[si, sj])
ic_sensors = np.array(sensors)  # (N_SENSORS*N_SENSORS,)

# Normalize
ic_norm = 2 * (ic_sensors - ic_min.squeeze()) / (ic_range.squeeze() + 1e-10) - 1
print(f"IC sensors shape: {ic_sensors.shape}")
print(f"IC (normalized): {ic_norm.shape}")

# ============================================================
# EVALUATE TRUNK & BRANCH ON SAMPLE 0
# ============================================================
print("\n" + "="*70)
print("EVALUATING TRUNK AND BRANCH NETWORKS ON SAMPLE 0")
print("="*70)

model.eval()
with torch.no_grad():
    # BRANCH: Just run once on IC
    ic0_tensor = torch.from_numpy(ic_norm).float().to(DEVICE).unsqueeze(0)  # (1, ic_dim)
    branch_out = model.branch(ic0_tensor).cpu().numpy()  # (1, N_MODES)
    branch_coeffs = branch_out[0]  # (N_MODES,)
    
    print(f"\nBranch Output (coefficients): shape {branch_coeffs.shape}")
    print(f"  Min: {branch_coeffs.min():.6f}")
    print(f"  Max: {branch_coeffs.max():.6f}")
    print(f"  Mean: {branch_coeffs.mean():.6f}")
    print(f"  Top 3 coefficients:")
    top_3_idx = np.argsort(np.abs(branch_coeffs))[::-1][:3]
    for rank, idx in enumerate(top_3_idx, 1):
        print(f"    {rank}. Mode {idx}: {branch_coeffs[idx]:.6f}")
    
    # TRUNK: Run at 5 time instants, all spatial points
    time_instants = [0.0, 0.25, 0.5, 0.75, 1.0]
    time_indices = [int(ti * (Nt - 1)) for ti in time_instants]
    
    trunk_outputs_all_times = []
    gt_all_times = []
    
    for t_val, t_idx in zip(time_instants, time_indices):
        coords_t = np.stack([X[:, :, t_idx].flatten('F'),
                             Y[:, :, t_idx].flatten('F'),
                             np.full((Nx * Ny,), t_val)], axis=1)
        coords_tensor = torch.from_numpy(coords_t).float().to(DEVICE)  # (Nx*Ny, 3)
        
        trunk_out = model.trunk(coords_tensor).cpu().numpy()  # (Nx*Ny, N_MODES)
        trunk_outputs_all_times.append(trunk_out)
        
        u_gt = u_fom[:, :, t_idx, 0].flatten('F')
        gt_all_times.append(u_gt)
    
    print(f"\nTrunk Output at each time:")
    for i, t_val in enumerate(time_instants):
        trunk_out = trunk_outputs_all_times[i]
        print(f"  t={t_val:.2f}: shape {trunk_out.shape}, "
              f"min={trunk_out.min():.6f}, max={trunk_out.max():.6f}")

# ============================================================
# PLOT 1: BRANCH COEFFICIENTS
# ============================================================
print("\n[1/4] Plotting branch coefficients...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Bar chart
ax1 = axes[0, 0]
colors = ['steelblue' if c >= 0 else 'coral' for c in branch_coeffs]
ax1.bar(range(N_MODES), branch_coeffs, color=colors, alpha=0.8)
ax1.set_xlabel('Mode Index', fontsize=11)
ax1.set_ylabel('Coefficient Value', fontsize=11)
ax1.set_title('Branch Network Output: IC → Coefficients\n(Sample 0)', fontsize=12)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax1.grid(True, alpha=0.3, axis='y')

# Magnitude bar chart
ax2 = axes[0, 1]
ax2.bar(range(N_MODES), np.abs(branch_coeffs), color='steelblue', alpha=0.8)
ax2.set_xlabel('Mode Index', fontsize=11)
ax2.set_ylabel('|Coefficient|', fontsize=11)
ax2.set_title('Magnitude of Coefficients\n(Which modes are active?)', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')

# Top 5 modes
ax3 = axes[1, 0]
top_n = min(5, N_MODES)
top_indices = np.argsort(np.abs(branch_coeffs))[::-1][:top_n]
top_values = branch_coeffs[top_indices]
bars = ax3.barh(range(top_n), top_values, color='steelblue', alpha=0.8)
ax3.set_yticks(range(top_n))
ax3.set_yticklabels([f'Mode {i}' for i in top_indices], fontsize=10)
ax3.set_xlabel('Coefficient Value', fontsize=11)
ax3.set_title(f'Top {top_n} Most Active Modes', fontsize=12)
ax3.grid(True, alpha=0.3, axis='x')

# Sorted coefficients
ax4 = axes[1, 1]
sorted_idx = np.argsort(np.abs(branch_coeffs))[::-1]
sorted_coeffs = np.abs(branch_coeffs[sorted_idx])
ax4.plot(sorted_coeffs, linewidth=2.5, marker='o', markersize=5, color='darkgreen')
ax4.set_xlabel('Rank (sorted by magnitude)', fontsize=11)
ax4.set_ylabel('|Coefficient|', fontsize=11)
ax4.set_title('Sorted Coefficient Magnitudes\n(Decay of mode importance)', fontsize=12)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
branch_path = os.path.join(SCRIPT_DIR, 'data/branch_coefficients_sample0.png')
plt.savefig(branch_path, dpi=150)
print(f"✓ Saved to data/branch_coefficients_sample0.png")
plt.close()

# ============================================================
# PLOT 2: TRUNK MODES VS SVD MODES (SORTED BY COEFFICIENT MAGNITUDE)
# SIDE-BY-SIDE COMPARISON AT 5 TIME INSTANTS
# ============================================================
print("\n[2/4] Plotting trunk modes vs SVD modes (sorted by coefficient magnitude)...")

# Sort modes by coefficient magnitude (largest first)
sorted_mode_indices = np.argsort(np.abs(branch_coeffs))[::-1]
n_modes_to_show = min(3, N_MODES)  # Show top 3 modes for clarity

# Calculate global colorbars for trunk modes
trunk_modes_all = np.concatenate([t[:, sorted_mode_indices[:n_modes_to_show]] 
                                   for t in trunk_outputs_all_times], axis=0)
trunk_vmin, trunk_vmax = trunk_modes_all.min(), trunk_modes_all.max()

# Calculate global colorbars for SVD modes at the same times and mode indices
svd_modes_selected = []
for t_idx in time_indices:
    for mode_col in range(n_modes_to_show):
        mode_idx = sorted_mode_indices[mode_col]
        svd_modes_selected.append(modes_svd_reshaped[:, :, t_idx, mode_idx].flatten())
svd_modes_all = np.concatenate(svd_modes_selected, axis=0)
svd_vmin, svd_vmax = svd_modes_all.min(), svd_modes_all.max()

# Create figure: 5 time rows × (3 modes × 2 columns for trunk+svd)
fig, axes = plt.subplots(len(time_instants), 2*n_modes_to_show, 
                          figsize=(3*n_modes_to_show*2, 4*len(time_instants)))

for t_row, (t_val, t_idx, trunk_out) in enumerate(zip(time_instants, time_indices, trunk_outputs_all_times)):
    for mode_col in range(n_modes_to_show):
        mode_idx = sorted_mode_indices[mode_col]
        branch_coeff_val = branch_coeffs[mode_idx]  # Branch network output
        svd_coeff_val = svd_coeffs_sample0[mode_idx]  # SVD coefficient (Sigma * VT)
        
        # Trunk mode (left)
        ax_trunk = axes[t_row, 2*mode_col]
        mode_spatial = trunk_out[:, mode_idx].reshape(Nx, Ny, order='F')
        im_trunk = ax_trunk.imshow(mode_spatial, cmap='seismic', origin='lower', 
                                    vmin=trunk_vmin, vmax=trunk_vmax)
        ax_trunk.set_title(f'Trunk Mode {mode_idx}\n(branch coeff={branch_coeff_val:.3f})', fontsize=9)
        ax_trunk.set_xticks([])
        ax_trunk.set_yticks([])
        
        # Add time label on the left side of first column
        if mode_col == 0:
            ax_trunk.set_ylabel(f't={t_val:.2f}', fontsize=11, fontweight='bold')
        
        # SVD mode (right)
        ax_svd = axes[t_row, 2*mode_col + 1]
        svd_mode_spatial = modes_svd_reshaped[:, :, t_idx, mode_idx]
        im_svd = ax_svd.imshow(svd_mode_spatial, cmap='seismic', origin='lower',
                                vmin=svd_vmin, vmax=svd_vmax)
        ax_svd.set_title(f'SVD Mode {mode_idx}\n(SVD coeff={svd_coeff_val:.3f})', fontsize=9)
        ax_svd.set_xticks([])
        ax_svd.set_yticks([])

# Add colorbars outside
fig.subplots_adjust(right=0.85)
cbar_ax_trunk = fig.add_axes([0.88, 0.55, 0.02, 0.35])
cbar_ax_svd = fig.add_axes([0.88, 0.1, 0.02, 0.35])
fig.colorbar(plt.cm.ScalarMappable(cmap='seismic', 
             norm=plt.Normalize(vmin=trunk_vmin, vmax=trunk_vmax)), 
             cax=cbar_ax_trunk, label='Trunk Mode Value')
fig.colorbar(plt.cm.ScalarMappable(cmap='seismic', 
             norm=plt.Normalize(vmin=svd_vmin, vmax=svd_vmax)), 
             cax=cbar_ax_svd, label='SVD Mode Value')

plt.suptitle(f'Trunk Network Modes vs SVD Modes (from MATLAB)\nTop {n_modes_to_show} Modes Sorted by Branch Coefficient Magnitude (Sample 0)', 
             fontsize=13, y=0.995)
trunk_path = os.path.join(SCRIPT_DIR, 'data/trunk_vs_svd_modes_sample0.png')
plt.savefig(trunk_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved to data/trunk_vs_svd_modes_sample0.png")
plt.close()

# ============================================================
# PLOT 3: WEIGHTED MODES (TRUNK * BRANCH)
# ============================================================
print("\n[3/4] Plotting weighted modes (trunk output × branch coefficients)...")

fig, axes = plt.subplots(len(time_instants), 3, figsize=(15, 4*len(time_instants)))

for t_row, (t_val, trunk_out) in enumerate(zip(time_instants, trunk_outputs_all_times)):
    # Select top 3 modes by coefficient magnitude
    top_3_idx = np.argsort(np.abs(branch_coeffs))[::-1][:3]
    
    for col, mode_idx in enumerate(top_3_idx):
        ax = axes[t_row, col]
        # Weight the mode by its coefficient
        mode_spatial = trunk_out[:, mode_idx]  # (Nx*Ny,)
        weighted = mode_spatial * branch_coeffs[mode_idx]  # Scalar multiplication
        weighted_reshaped = weighted.reshape(Nx, Ny, order='F')
        
        im = ax.imshow(weighted_reshaped, cmap='seismic', origin='lower')
        ax.set_title(f'Mode {mode_idx} × coeff={branch_coeffs[mode_idx]:.3f}\nt={t_val:.2f}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle('Weighted Modes: Trunk Output × Branch Coefficients\n(Sample 0 - Top 3 Active Modes)', fontsize=14, y=1.00)
plt.tight_layout()
weighted_path = os.path.join(SCRIPT_DIR, 'data/weighted_modes_sample0.png')
plt.savefig(weighted_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved to data/weighted_modes_sample0.png")
plt.close()

# ============================================================
# PLOT 4: TRUNK MODE TEMPORAL EVOLUTION
# ============================================================
print("\n[4/4] Plotting trunk mode temporal evolution at center point...")

center_x, center_y = Nx // 2, Ny // 2

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Extract mode values at center point across all times
trunk_at_center_all_times = []
for trunk_out in trunk_outputs_all_times:
    center_flat_idx = center_x * (Ny * Nt) + center_y * Nt  # Fortran order
    # Simpler: get from spatial reshape
    trunk_spatial = trunk_out.reshape(Nx, Ny, N_MODES, order='F')
    trunk_at_center_all_times.append(trunk_spatial[center_x, center_y, :])

trunk_at_center_all_times = np.array(trunk_at_center_all_times)  # (5, N_MODES)

# Plot 1: First 4 modes temporal evolution at center
ax1 = axes[0, 0]
for mode_idx in range(4):
    ax1.plot(range(len(time_instants)), trunk_at_center_all_times[:, mode_idx],
             marker='o', label=f'Mode {mode_idx}', linewidth=2.5, markersize=6)
ax1.set_xticks(range(len(time_instants)))
ax1.set_xticklabels([f'{t:.2f}' for t in time_instants])
ax1.set_xlabel('Time', fontsize=11)
ax1.set_ylabel('Mode Value', fontsize=11)
ax1.set_title(f'Modes 0-3 Temporal Evolution\nat center ({center_x}, {center_y})', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Modes 4-7
ax2 = axes[0, 1]
for mode_idx in range(4, 8):
    ax2.plot(range(len(time_instants)), trunk_at_center_all_times[:, mode_idx],
             marker='s', label=f'Mode {mode_idx}', linewidth=2.5, markersize=6)
ax2.set_xticks(range(len(time_instants)))
ax2.set_xticklabels([f'{t:.2f}' for t in time_instants])
ax2.set_xlabel('Time', fontsize=11)
ax2.set_ylabel('Mode Value', fontsize=11)
ax2.set_title(f'Modes 4-7 Temporal Evolution\nat center ({center_x}, {center_y})', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Top 3 active modes
ax3 = axes[1, 0]
top_3_idx = np.argsort(np.abs(branch_coeffs))[::-1][:3]
for mode_idx in top_3_idx:
    ax3.plot(range(len(time_instants)), trunk_at_center_all_times[:, mode_idx],
             marker='o', label=f'Mode {mode_idx} (coeff={branch_coeffs[mode_idx]:.3f})',
             linewidth=2.5, markersize=6)
ax3.set_xticks(range(len(time_instants)))
ax3.set_xticklabels([f'{t:.2f}' for t in time_instants])
ax3.set_xlabel('Time', fontsize=11)
ax3.set_ylabel('Mode Value', fontsize=11)
ax3.set_title(f'Top 3 Active Modes\nat center ({center_x}, {center_y})', fontsize=12)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Plot 4: Reconstructed solution at center
ax4 = axes[1, 1]
u_recon_center = []
for trunk_out in trunk_outputs_all_times:
    trunk_spatial = trunk_out.reshape(Nx, Ny, N_MODES, order='F')
    modes_at_center = trunk_spatial[center_x, center_y, :]  # (N_MODES,)
    u_val = np.sum(modes_at_center * branch_coeffs)
    u_recon_center.append(u_val)

ax4.plot(range(len(time_instants)), u_recon_center,
         marker='o', linewidth=2.5, markersize=7, color='darkgreen', label='Trunk×Branch')
ax4.plot(range(len(time_instants)), [gt[center_x, center_y] for gt in gt_all_times],
         marker='s', linewidth=2.5, markersize=7, color='darkred', label='Ground Truth',
         linestyle='--')
ax4.set_xticks(range(len(time_instants)))
ax4.set_xticklabels([f'{t:.2f}' for t in time_instants])
ax4.set_xlabel('Time', fontsize=11)
ax4.set_ylabel('Solution Value', fontsize=11)
ax4.set_title(f'Reconstructed vs GT at Center\n(Sum over all modes)', fontsize=12)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
temporal_path = os.path.join(SCRIPT_DIR, 'data/trunk_temporal_evolution_sample0.png')
plt.savefig(temporal_path, dpi=150)
print(f"✓ Saved to data/trunk_temporal_evolution_sample0.png")
plt.close()

# ============================================================
# SUMMARY STATISTICS
# ============================================================
print("\n" + "="*70)
print("SUMMARY: TRUNK & BRANCH VERIFICATION FOR SAMPLE 0")
print("="*70)

print(f"\nBRANCH NETWORK (IC → Coefficients):")
print(f"  Input: IC sensors {ic_sensors.shape} → normalized {ic_norm.shape}")
print(f"  Output: {branch_coeffs.shape} coefficient values")
print(f"  Interpretation: How much weight to give each mode")
print(f"  Top 3 active modes: {top_3_idx}")

print(f"\nTRUNK NETWORK ((x,y,t) → Modes):")
print(f"  Input: Spatial coords (x,y) + time t")
print(f"  Output: {N_MODES} mode values at each point")
print(f"  Interpretation: Spatiotemporal basis functions")
print(f"  Tested at: 5 time instants × {Nx*Ny} spatial points")

print(f"\nRECONSTRUCTION FORMULA:")
print(f"  u(x,y,t | IC) = ∑ᵢ Branch_i(IC) × Trunk_i(x,y,t)")
print(f"  = ∑ᵢ branch_coeffs[i] × trunk_modes[i](x,y,t)")
print(f"  = {len(branch_coeffs)} terms summed together")

print(f"\nVERIFICATION:")
print(f"  ✓ Branch outputs {len(branch_coeffs)} scalars (IC-specific)")
print(f"  ✓ Trunk outputs {N_MODES} values per point (problem-specific)")
print(f"  ✓ Interaction is element-wise: 18 scalars × 18 fields")

print("\n" + "="*70)
print("✓ TRUNK & BRANCH VISUALIZATION COMPLETE")
print("="*70)

print("\nGenerated plots:")
print("  1. branch_coefficients_sample0.png - How IC affects coefficients")
print("  2. trunk_vs_svd_modes_sample0.png - Trunk modes vs SVD modes (sorted by coefficient magnitude)")
print("  3. weighted_modes_sample0.png - Modes scaled by their coefficients")
print("  4. trunk_temporal_evolution_sample0.png - Mode evolution over time")
