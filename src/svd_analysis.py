"""
SVD Basis Extraction and Analysis

Performs POD (Proper Orthogonal Decomposition) via SVD on ground truth data.
Extracts basis functions (U_basis), singular values, and coefficients.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd
from datetime import datetime


def extract_svd_basis(
    u_fom: np.ndarray,
    n_modes: int = 128,
    visualize: bool = True,
    output_dir: str = None,
) -> dict:
    """
    Extract SVD basis from ground truth solutions via randomized SVD.
    
    Args:
        u_fom: Displacement field (Nx, Ny, Nt, N_samples)
        n_modes: Number of modes to extract
        visualize: Generate visualization plots
        output_dir: Directory for saving plots (if None, no plots saved)
    
    Returns:
        dict with keys: {'basis', 'singular_values', 'coefficients', 'grid_info'}
    """
    
    print("=" * 70)
    print("EXTRACTING SVD BASIS")
    print("=" * 70)
    
    Nx, Ny, Nt, N_samples = u_fom.shape
    n_space_time = Nx * Ny * Nt
    features_dim = n_space_time
    
    print(f"\nInput shape: {u_fom.shape} (Nx, Ny, Nt, N_samples)")
    print(f"Space-time points: {n_space_time}")
    print(f"Samples: {N_samples}")
    
    # Reshape for SVD
    print("\nReshaping for SVD...")
    U_matrix = u_fom.reshape(features_dim, N_samples, order='F')
    print(f"Matrix shape: {U_matrix.shape} (space-time × samples)")
    
    # Compute SVD
    max_modes = min(features_dim, N_samples)
    n_modes_to_compute = min(n_modes, max_modes)
    
    print(f"\nComputing randomized SVD (k={n_modes_to_compute})...")
    U_basis, Sigma, VT = randomized_svd(
        U_matrix, 
        n_components=n_modes_to_compute, 
        random_state=42
    )
    
    actual_modes = U_basis.shape[1]
    print(f"✓ SVD complete: {actual_modes} modes extracted")
    print(f"  U_basis: {U_basis.shape} (basis functions)")
    print(f"  Sigma: {Sigma.shape} (singular values)")
    print(f"  VT: {VT.shape} (coefficients)")
    
    # Energy analysis
    cumulative_energy = np.cumsum(Sigma**2) / np.sum(Sigma**2)
    energy_captured = cumulative_energy[-1] * 100
    
    print(f"\nEnergy captured by {actual_modes} modes: {energy_captured:.2f}%")
    print(f"Energy by first 50 modes: {cumulative_energy[min(49, actual_modes-1)]*100:.2f}%")
    print(f"Energy by first 100 modes: {cumulative_energy[min(99, actual_modes-1)]*100:.2f}%")
    
    # Visualization
    if visualize and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        _visualize_svd_basis(
            U_basis, Sigma, VT, U_matrix, Nx, Ny, Nt, actual_modes, output_dir
        )
    
    print("\n✓ SVD analysis complete")
    print("=" * 70)
    
    return {
        'basis': U_basis,
        'singular_values': Sigma,
        'coefficients': VT,
        'grid_info': np.array([Nx, Ny, Nt]),
    }


def _visualize_svd_basis(
    U_basis: np.ndarray,
    Sigma: np.ndarray,
    VT: np.ndarray,
    U_matrix: np.ndarray,
    Nx: int,
    Ny: int,
    Nt: int,
    actual_modes: int,
    output_dir: str,
):
    """Generate visualization plots for SVD analysis."""
    
    print("\nGenerating visualization plots...")
    
    # Reshape basis for visualization
    modes_reshaped = U_basis.reshape(Nx, Ny, Nt, actual_modes, order='F')
    t_idx = Nt // 20
    n_modes_to_plot = min(18, actual_modes)
    
    # Plot 1: Basis functions
    fig, axes = plt.subplots(6, 3, figsize=(15, 30))
    for i in range(n_modes_to_plot):
        row, col = i // 3, i % 3
        mode_slice = modes_reshaped[:, :, t_idx, i]
        im = axes[row, col].imshow(mode_slice, cmap='seismic', origin='lower')
        axes[row, col].set_title(f"SVD Mode {i} (t={t_idx})")
        plt.colorbar(im, ax=axes[row, col])
    
    for i in range(n_modes_to_plot, 18):
        row, col = i // 3, i % 3
        axes[row, col].axis('off')
    
    plt.suptitle(f"Trunk Basis Functions ({actual_modes} modes)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'svd_modes_check.png'), dpi=150)
    plt.close()
    print(f"  ✓ {os.path.join(output_dir, 'svd_modes_check.png')}")
    
    # Plot 2: Energy decay
    cumulative_energy = np.cumsum(Sigma**2) / np.sum(Sigma**2)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].semilogy(Sigma)
    axes[0].set_title("Singular Values Decay")
    axes[0].set_xlabel("Mode Index")
    axes[0].set_ylabel("Sigma (Log Scale)")
    axes[0].grid(True, which="both", ls="--")
    
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
    plt.savefig(os.path.join(output_dir, 'svd_energy_decay.png'), dpi=150)
    plt.close()
    print(f"  ✓ {os.path.join(output_dir, 'svd_energy_decay.png')}")
    
    # Plot 3: Coefficients analysis
    n_modes_to_show = min(30, actual_modes)
    n_samples_to_show = min(20, VT.shape[1])
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Heatmap of coefficients
    im1 = axes[0, 0].imshow(
        VT[:n_modes_to_show, :n_samples_to_show], 
        cmap='RdBu_r', aspect='auto', interpolation='nearest'
    )
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Mode Index')
    axes[0, 0].set_title(f'SVD Coefficients (first {n_modes_to_show}×{n_samples_to_show})')
    plt.colorbar(im1, ax=axes[0, 0], label='Coefficient Value')
    
    # Mode magnitude
    mean_coeff = np.abs(VT[:min(20, actual_modes), :]).mean(axis=1)
    axes[0, 1].bar(range(len(mean_coeff)), mean_coeff, color='steelblue')
    axes[0, 1].set_xlabel('Mode Index')
    axes[0, 1].set_ylabel('Mean |Coefficient| across samples')
    axes[0, 1].set_title('Mode Activation')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Coefficient profiles for first 3 samples
    for sample_idx in range(min(3, VT.shape[1])):
        coeffs = VT[:min(20, actual_modes), sample_idx]
        axes[1, 0].plot(coeffs, marker='o', label=f'Sample {sample_idx}', linewidth=2)
    axes[1, 0].set_xlabel('Mode Index')
    axes[1, 0].set_ylabel('Coefficient Value')
    axes[1, 0].set_title('Coefficient Profiles (first 3 samples)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Coefficient variability
    coeff_std = np.std(VT[:min(30, actual_modes), :], axis=1)
    coeff_mean = np.mean(np.abs(VT[:min(30, actual_modes), :]), axis=1)
    axes[1, 1].bar(range(len(coeff_std)), coeff_std, color='coral', alpha=0.7, label='Std Dev')
    axes[1, 1].plot(range(len(coeff_mean)), coeff_mean, 'bo-', linewidth=2, label='Mean |Coeff|')
    axes[1, 1].set_xlabel('Mode Index')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Coefficient Variability Across Samples')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'svd_coefficients_analysis.png'), dpi=150)
    plt.close()
    print(f"  ✓ {os.path.join(output_dir, 'svd_coefficients_analysis.png')}")

    # Plot 4: Reconstruction vs ground truth
    U_recon = (U_basis * Sigma) @ VT  # equivalent to U @ diag(Sigma) @ VT
    n_samples_to_compare = min(3, VT.shape[1])
    n_times_to_compare = min(3, Nt)
    time_indices = [Nt * k // (n_times_to_compare + 1) for k in range(1, n_times_to_compare + 1)]

    fig, axes = plt.subplots(
        n_samples_to_compare * 3,
        n_times_to_compare,
        figsize=(5 * n_times_to_compare, 5 * n_samples_to_compare * 3),
    )
    # Ensure axes is always 2D
    if axes.ndim == 1:
        axes = axes[:, np.newaxis]

    for s in range(n_samples_to_compare):
        orig_s = U_matrix[:, s].reshape(Nx, Ny, Nt, order='F')
        recon_s = U_recon[:, s].reshape(Nx, Ny, Nt, order='F')
        for t_col, t_idx_r in enumerate(time_indices):
            orig_slice = orig_s[:, :, t_idx_r]
            recon_slice = recon_s[:, :, t_idx_r]
            err_slice = orig_slice - recon_slice

            vmin, vmax = orig_slice.min(), orig_slice.max()
            row_orig = s * 3
            row_recon = s * 3 + 1
            row_err = s * 3 + 2

            im0 = axes[row_orig, t_col].imshow(orig_slice, cmap='seismic', origin='lower', vmin=vmin, vmax=vmax)
            axes[row_orig, t_col].set_title(f"GT  s={s} t={t_idx_r}")
            plt.colorbar(im0, ax=axes[row_orig, t_col])

            im1 = axes[row_recon, t_col].imshow(recon_slice, cmap='seismic', origin='lower', vmin=vmin, vmax=vmax)
            axes[row_recon, t_col].set_title(f"Recon  s={s} t={t_idx_r}")
            plt.colorbar(im1, ax=axes[row_recon, t_col])

            abs_err = np.abs(err_slice)
            im2 = axes[row_err, t_col].imshow(abs_err, cmap='hot', origin='lower')
            rel_err = abs_err.max() / (np.abs(orig_slice).max() + 1e-12)
            axes[row_err, t_col].set_title(f"|Error|  max_rel={rel_err:.2e}")
            plt.colorbar(im2, ax=axes[row_err, t_col])

    plt.suptitle(
        f"Reconstruction vs Ground Truth ({actual_modes} modes)\n"
        "Rows: [GT, Recon, |Error|] per sample",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'svd_reconstruction_check.png'), dpi=150)
    plt.close()
    print(f"  ✓ {os.path.join(output_dir, 'svd_reconstruction_check.png')}")
