import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. CONFIGURAZIONE
DATA_PATH = 'data/test_cases.mat'  # Il file generato da MATLAB
N_MODES = 128                    # Numero di basi da estrarre (p)
VISUALIZE = True

print(f"Loading data from {DATA_PATH}...")
# MATLAB v7.3 files use HDF5 format, need h5py instead of scipy.io
with h5py.File(os.path.join(SCRIPT_DIR, DATA_PATH), 'r') as f:
    # h5py reads MATLAB arrays in transposed format, so we transpose back
    U_raw = np.array(f['U_data']).T  # Now shape: (Nx, Ny, Nt, N_samples) 
print(f"Original Shape: {U_raw.shape}")

# 2. RESHAPING PER SVD
# Vogliamo appiattire (Nx, Ny, Nt) in una sola dimensione "Features"
# E tenere N_samples come dimensione "Samples"
Nx, Ny, Nt, N_samples = U_raw.shape
features_dim = Nx * Ny * Nt

print("Reshaping for SVD...")
# Flattening: (Nx*Ny*Nt, N_samples)
# Use Fortran order to match coordinate generation (x varies fastest, then y, then t)
U_matrix = U_raw.reshape(features_dim, N_samples, order='F') 

print(f"Matrix for SVD: {U_matrix.shape} (Rows=SpaceTime Points, Cols=Samples)")

# 3. CALCOLO SVD (Randomized per velocità su matrici grandi)
# Maximum modes = min(features_dim, N_samples)
max_modes = min(features_dim, N_samples)
n_modes_to_compute = min(N_MODES, max_modes)

print(f"Computing Randomized SVD (requested k={N_MODES}, actual k={n_modes_to_compute})...")
# U_basis: Le basi spaziotemporali (Features x p) — SHARED across all samples
# Sigma: I valori singolari (p,)
# VT: I coefficienti temporali/sample (p x Samples) — VARIES per sample
U_basis, Sigma, VT = randomized_svd(U_matrix, n_components=n_modes_to_compute, random_state=42)

actual_modes = U_basis.shape[1]
print(f"\nSVD Decomposition Complete:")
print(f"  U_basis shape: {U_basis.shape} (Basis functions - SHARED across all samples)")
print(f"  Sigma shape: {Sigma.shape} (Singular values)")
print(f"  VT shape: {VT.shape} (Coefficients - varies per sample)")
print(f"  Actual modes: {actual_modes}")

# Compute cumulative energy captured by modes
cumulative_energy = np.cumsum(Sigma**2) / np.sum(Sigma**2)
energy_captured = cumulative_energy[-1] * 100
print(f"\nEnergy captured by {actual_modes} modes: {energy_captured:.2f}%")
print(f"Energy by first 50 modes: {cumulative_energy[min(49, actual_modes-1)]*100:.2f}%")
print(f"Energy by first 100 modes: {cumulative_energy[min(99, actual_modes-1)]*100:.2f}%")

# 4. SALVATAGGIO DATI PROCESSATI
# Salviamo le basi U_basis e i valori singolari. 
# La Trunk Net dovrà imparare a predire U_basis[x,y,t] dato (x,y,t).
# La Branch Net dovrà imparare a predire i coefficienti (VT) dall'IC.
output_dict = {
    'basis': U_basis,              # (Nx*Ny*Nt, p) — SHARED modes across all samples
    'singular_values': Sigma,      # (p,) — importance of each mode
    'coefficients': VT,            # (p, N_samples) — how much each mode appears in each sample
    'grid_info': np.array([Nx, Ny, Nt])
}
np.save(os.path.join(SCRIPT_DIR, 'data/svd_basis_data.npy'), output_dict)
print("\nSVD data saved to svd_basis_data.npy")
print(f"  Modes (U_basis): {U_basis.shape} - Problem characteristic, same for all ICs")
print(f"  Coefficients (VT): {VT.shape} - IC-dependent, varies across {N_samples} samples")

# 5. VISUALIZZAZIONE (Check Fisico)
if VISUALIZE:
    print("Visualizing first 18 modes...")
    # Reshape delle basi per plottarle: (Nx, Ny, Nt, actual_modes)
    # Use Fortran order to match the flattening order
    modes_reshaped = U_basis.reshape(Nx, Ny, Nt, actual_modes, order='F')
    
    # Plottiamo i primi 18 modi al tempo t=Nt/2 (metà simulazione)
    t_idx = Nt // 3
    n_modes_to_plot = min(18, actual_modes)
    
    fig, axes = plt.subplots(6, 3, figsize=(15, 30))
    for i in range(n_modes_to_plot):
        row = i // 3
        col = i % 3
        mode_slice = modes_reshaped[:, :, t_idx, i]
        im = axes[row, col].imshow(mode_slice, cmap='seismic', origin='lower')
        axes[row, col].set_title(f"SVD Mode {i} (at t={t_idx})")
        plt.colorbar(im, ax=axes[row, col])
    
    # Hide unused subplots if we have fewer than 18 modes
    for i in range(n_modes_to_plot, 18):
        row = i // 3
        col = i % 3
        axes[row, col].axis('off')
    
    plt.suptitle(f"Trunk Basis Functions ({actual_modes} modes captured)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'data/svd_modes_check.png'), dpi=150)
    plt.show()

    # Plot Decadimento Valori Singolari (Energia)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Singular values decay
    axes[0].semilogy(Sigma)
    axes[0].set_title("Singular Values Decay")
    axes[0].set_xlabel("Mode Index")
    axes[0].set_ylabel("Sigma (Log Scale)")
    axes[0].grid(True, which="both", ls="--")
    
    # Right: Cumulative energy
    axes[1].plot(cumulative_energy * 100, linewidth=2)
    axes[1].axhline(y=90, color='r', linestyle='--', label='90% energy')
    axes[1].axhline(y=99, color='orange', linestyle='--', label='99% energy')
    axes[1].set_title("Cumulative Energy Captured")
    axes[1].set_xlabel("Number of Modes")
    axes[1].set_ylabel("Energy (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'data/svd_energy_decay.png'))
    plt.show()
    
    # Se la curva scende rapidamente, significa che i modi disponibili sono sufficienti.

    # ============================================================
    # COEFFICIENT ANALYSIS: How modes vary across samples
    # ============================================================
    print("\nVisualizing SVD Coefficients...")
    
    # VT shape: (p, N_samples)
    # Each row = one mode across all samples
    # Each column = all modes for one sample
    
    n_modes_to_show = min(30, actual_modes)
    n_samples_to_show = min(20, N_samples)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Heatmap: Which modes are important for which samples?
    ax1 = axes[0, 0]
    im1 = ax1.imshow(VT[:n_modes_to_show, :n_samples_to_show], 
                     cmap='RdBu_r', aspect='auto', interpolation='nearest')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Mode Index')
    ax1.set_title(f'SVD Coefficients (first {n_modes_to_show} modes × {n_samples_to_show} samples)')
    plt.colorbar(im1, ax=ax1, label='Coefficient Value')
    
    # 2. Mode magnitude across samples (bar plot)
    ax2 = axes[0, 1]
    mean_coeff = np.abs(VT[:min(20, actual_modes), :]).mean(axis=1)
    ax2.bar(range(len(mean_coeff)), mean_coeff, color='steelblue')
    ax2.set_xlabel('Mode Index')
    ax2.set_ylabel('Mean |Coefficient| across all samples')
    ax2.set_title('Mode Activation: Which modes are used most?')
    ax2.grid(True, alpha=0.3)
    
    # 3. Coefficient statistics for first 3 samples
    ax3 = axes[1, 0]
    for sample_idx in range(min(3, N_samples)):
        coeffs = VT[:min(20, actual_modes), sample_idx]
        ax3.plot(coeffs, marker='o', label=f'Sample {sample_idx}', linewidth=2)
    ax3.set_xlabel('Mode Index')
    ax3.set_ylabel('Coefficient Value')
    ax3.set_title('Coefficient profiles for first 3 samples (show variation)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Variance of coefficients across samples
    ax4 = axes[1, 1]
    coeff_std = np.std(VT[:min(30, actual_modes), :], axis=1)
    coeff_mean = np.mean(np.abs(VT[:min(30, actual_modes), :]), axis=1)
    ax4.bar(range(len(coeff_std)), coeff_std, color='coral', alpha=0.7, label='Std Dev')
    ax4.plot(range(len(coeff_mean)), coeff_mean, 'bo-', linewidth=2, markersize=6, label='Mean |Coeff|')
    ax4.set_xlabel('Mode Index')
    ax4.set_ylabel('Value')
    ax4.set_title('Coefficient Variability Across Samples')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'data/svd_coefficients_analysis.png'), dpi=150)
    plt.show()
    
    # ============================================================
    # TEMPORAL EVOLUTION: Reconstruct one sample and visualize evolution
    # ============================================================
    print("Visualizing temporal evolution for one sample...")
    
    # ============================================================
    # UNDERSTAND MODES vs COEFFICIENTS
    # ============================================================
    print("\n" + "="*70)
    print("MODES vs COEFFICIENTS EXPLANATION")
    print("="*70)
    
    # Modes are spatiotemporal functions (U_basis reshaped)
    modes_reshaped = U_basis.reshape(Nx, Ny, Nt, actual_modes, order='F')
    
    # Coefficients are scalars (one per mode per sample)
    sample_idx = 0
    coeff_sample = VT[:actual_modes, sample_idx]
    
    print(f"\nU_basis (Modes): shape {U_basis.shape}")
    print(f"  → Reshaped to (Nx={Nx}, Ny={Ny}, Nt={Nt}, modes={actual_modes})")
    print(f"  → Each mode is a 4D spatiotemporal field")
    print(f"  → SHARED across all {N_samples} samples (characteristic of the problem)")
    
    print(f"\nVT (Coefficients): shape {VT.shape}")
    print(f"  → VT[i, j] = coefficient for mode i in sample j")
    print(f"  → For sample {sample_idx}: {coeff_sample.shape}")
    print(f"  → These are SCALARS (one number per mode)")
    print(f"  → VARY across different initial conditions")
    
    print(f"\nReconstruction formula:")
    print(f"  u_j(x,y,t) = Σᵢ VT[i,j] × U_basis_reshaped[x,y,t,i]")
    print(f"  where VT[i,j] is a scalar and U_basis_reshaped[x,y,t,i] is a field")
    print("="*70)
    
    # Reconstruct the solution for sample 0: u = U_basis @ (Sigma * VT[:, 0])
    reconstructed_sample = U_basis @ (Sigma[:actual_modes] * VT[:actual_modes, sample_idx])
    
    # Reshape back to spatiotemporal: (Nx, Ny, Nt)
    u_reconstructed = reconstructed_sample.reshape(Nx, Ny, Nt, order='F')
    
    # Extract temporal evolution at center point
    center_x, center_y = Nx//2, Ny//2
    u_center_time = u_reconstructed[center_x, center_y, :]
    
    # Also get spatial average over time
    u_spatial_mean_time = np.mean(u_reconstructed, axis=(0, 1))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Temporal evolution at center spatial point
    ax1 = axes[0, 0]
    ax1.plot(u_center_time, linewidth=2, color='steelblue', marker='o', markersize=4)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Solution Value')
    ax1.set_title(f'RECONSTRUCTED Solution at Center\n(Sample {sample_idx}, location ({center_x}, {center_y}))')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Spatial-mean temporal evolution
    ax2 = axes[0, 1]
    ax2.plot(u_spatial_mean_time, linewidth=2, color='coral', marker='s', markersize=4)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Spatially-Averaged Solution Value')
    ax2.set_title(f'RECONSTRUCTED Solution (Spatially Averaged)\n(Sample {sample_idx})')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Scalar coefficients for sample 0
    ax3 = axes[1, 0]
    ax3.bar(range(min(20, actual_modes)), np.abs(coeff_sample[:min(20, actual_modes)]), color='steelblue')
    ax3.set_xlabel('Mode Index')
    ax3.set_ylabel('|Coefficient| (scalar)')
    ax3.set_title(f'SVD Coefficients for Sample {sample_idx}\n(Scalars - how much of each mode is used)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Mode temporal evolution at center
    ax4 = axes[1, 1]
    for mode_idx in range(min(4, actual_modes)):
        mode_at_center = modes_reshaped[center_x, center_y, :, mode_idx]
        ax4.plot(mode_at_center, marker='o', label=f'Mode {mode_idx}', linewidth=2)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Mode Value')
    ax4.set_title(f'Individual MODE Temporal Evolution at Center\n(Basis functions, scaled by coefficient)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'data/svd_temporal_evolution.png'), dpi=150)
    plt.show()
    
    print("\nTemporal evolution visualization saved!")

print(f"\nNote: With {N_samples} samples, max achievable modes = {max_modes}")
print(f"To get {N_MODES} modes, increase N_samples in MATLAB to at least {N_MODES}.")