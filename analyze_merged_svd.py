"""
SVD Analysis for Merged Dataset

Performs POD (Proper Orthogonal Decomposition) via SVD on the merged dataset
(data/merged.mat and data/svd_merged.npy).

This script loads displacement (U) and velocity (V) data and performs SVD analysis,
saving all figures to the data directory.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.utils.extmath import randomized_svd
from datetime import datetime
import h5py


def load_merged_data(data_dir: str = "data"):
    """
    Load U_data and V_data from merged.mat HDF5 file.
    
    Returns:
        tuple: (U_data, V_data, grid_info) where shapes are:
               U_data: (N_samples, Nt, Nx, Ny)
               V_data: (N_samples, Nt, Nx, Ny)
               grid_info: dict with metadata
    """
    merged_path = Path(data_dir) / "merged.mat"
    
    if not merged_path.exists():
        raise FileNotFoundError(f"merged.mat not found at {merged_path}")
    
    print(f"Loading data from {merged_path}...")
    
    with h5py.File(merged_path, "r") as f:
        print(f"Available keys: {list(f.keys())}")
        
        # Load displacement and velocity data
        U_data = np.array(f["U_data"])  # (N_samples, Nt, Nx, Ny)
        V_data = np.array(f["V_data"])  # (N_samples, Nt, Nx, Ny)
        
        print(f"U_data shape: {U_data.shape}")
        print(f"V_data shape: {V_data.shape}")
        
        # Load grid info if available
        grid_info = {}
        for key in ["x_grid", "y_grid", "tlist"]:
            if key in f:
                grid_info[key] = np.array(f[key])
    
    return U_data, V_data, grid_info


def reshape_for_svd(data: np.ndarray) -> np.ndarray:
    """
    Reshape (N_samples, Nt, Nx, Ny) to (space_time, N_samples) for SVD.
    
    Args:
        data: (N_samples, Nt, Nx, Ny)
    
    Returns:
        Reshaped matrix: (Nt*Nx*Ny, N_samples) in Fortran order
    """
    N_samples, Nt, Nx, Ny = data.shape
    n_space_time = Nt * Nx * Ny
    
    # Transpose to (Nt, Nx, Ny, N_samples) then reshape
    data_transposed = data.transpose(1, 2, 3, 0)  # (Nt, Nx, Ny, N_samples)
    matrix = data_transposed.reshape(n_space_time, N_samples, order='F')
    
    return matrix


def extract_svd_basis(
    U_data: np.ndarray,
    V_data: np.ndarray = None,
    n_modes: int = 128,
    output_dir: str = "data",
) -> dict:
    """
    Extract SVD basis from displacement and velocity data.
    
    Args:
        U_data:     (N_samples, Nt, Nx, Ny)
        V_data:     (N_samples, Nt, Nx, Ny), optional
        n_modes:    Number of modes to extract
        output_dir: Directory for saving plots
    
    Returns:
        dict with SVD components and analysis results
    """
    print("=" * 70)
    print("EXTRACTING SVD BASIS FROM MERGED DATASET")
    print("=" * 70)
    
    N_samples, Nt, Nx, Ny = U_data.shape
    n_space_time = Nt * Nx * Ny
    
    print(f"\nInput shape: ({N_samples}, {Nt}, {Nx}, {Ny}) (N_samples, Nt, Nx, Ny)")
    print(f"Space-time points: {n_space_time}")
    print(f"Samples: {N_samples}")
    
    # Reshape for SVD
    print("\nReshaping for SVD...")
    U_matrix = reshape_for_svd(U_data)
    print(f"U_matrix shape: {U_matrix.shape} (space-time × samples)")
    
    # Compute SVD
    max_modes = min(n_space_time, N_samples)
    n_modes_to_compute = min(n_modes, max_modes)
    
    print(f"\nComputing randomized SVD (k={n_modes_to_compute})...")
    U_basis, Sigma, VT = randomized_svd(
        U_matrix,
        n_components=n_modes_to_compute,
        random_state=42
    )
    
    actual_modes = U_basis.shape[1]
    print(f"✓ SVD complete: {actual_modes} modes extracted")
    print(f"  U_basis: {U_basis.shape}")
    print(f"  Sigma: {Sigma.shape}")
    print(f"  VT: {VT.shape}")
    
    # Energy analysis
    cumulative_energy = np.cumsum(Sigma**2) / np.sum(Sigma**2)
    energy_captured = cumulative_energy[-1] * 100
    
    print(f"\nEnergy captured by {actual_modes} modes: {energy_captured:.2f}%")
    print(f"Energy by first 50 modes: {cumulative_energy[min(49, actual_modes-1)]*100:.2f}%")
    print(f"Energy by first 100 modes: {cumulative_energy[min(99, actual_modes-1)]*100:.2f}%")
    
    result = {
        'basis_u': U_basis,
        'singular_values_u': Sigma,
        'coefficients_u': VT,
        'grid_shape': np.array([Nt, Nx, Ny]),
    }
    
    # Optional: SVD on velocity field
    if V_data is not None:
        print("\n" + "-" * 50)
        print("Computing SVD for velocity field...")
        V_matrix = reshape_for_svd(V_data)
        
        V_basis, Sigma_v, VT_v = randomized_svd(
            V_matrix,
            n_components=n_modes_to_compute,
            random_state=42
        )
        actual_modes_v = V_basis.shape[1]
        cumulative_energy_v = np.cumsum(Sigma_v**2) / np.sum(Sigma_v**2)
        
        print(f"✓ V SVD: {actual_modes_v} modes, energy={cumulative_energy_v[-1]*100:.2f}%")
        print(f"  (50 modes: {cumulative_energy_v[min(49, actual_modes_v-1)]*100:.2f}%)")
        
        result['basis_v'] = V_basis
        result['singular_values_v'] = Sigma_v
        result['coefficients_v'] = VT_v
    
    # Visualize and save results
    visualize_svd_analysis(
        svd_data=result,
        output_dir=output_dir,
        n_modes=n_modes,
        U_data=U_data,
        V_data=V_data,
        Nt=Nt,
        Nx=Nx,
        Ny=Ny,
    )
    
    print("\n✓ SVD analysis complete")
    print("=" * 70)
    
    return result


def visualize_svd_analysis(
    svd_data: dict,
    output_dir: str,
    n_modes: int,
    U_data: np.ndarray,
    V_data: np.ndarray = None,
    Nt: int = None,
    Nx: int = None,
    Ny: int = None,
):
    """Generate SVD visualization plots."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if Nt is None:
        Nt, Nx, Ny = svd_data['grid_shape']
    
    print("\nGenerating visualization plots...")
    
    # Displacement field visualizations
    print("\n  • Visualizing displacement SVD...")
    _visualize_field_svd(
        field_name='displacement',
        field_prefix='u',
        basis=svd_data['basis_u'],
        sigma=svd_data['singular_values_u'],
        vt=svd_data['coefficients_u'],
        Nt=Nt,
        Nx=Nx,
        Ny=Ny,
        output_dir=output_dir,
        n_modes=n_modes,
        raw_data=U_data,
    )
    
    # Velocity field visualizations
    if 'basis_v' in svd_data:
        print("\n  • Visualizing velocity SVD...")
        _visualize_field_svd(
            field_name='velocity',
            field_prefix='v',
            basis=svd_data['basis_v'],
            sigma=svd_data['singular_values_v'],
            vt=svd_data['coefficients_v'],
            Nt=Nt,
            Nx=Nx,
            Ny=Ny,
            output_dir=output_dir,
            n_modes=n_modes,
            raw_data=V_data,
        )


def _visualize_field_svd(
    field_name: str,
    field_prefix: str,
    basis: np.ndarray,
    sigma: np.ndarray,
    vt: np.ndarray,
    Nt: int,
    Nx: int,
    Ny: int,
    output_dir: str,
    n_modes: int = None,
    raw_data: np.ndarray = None,
):
    """Generate SVD plots for one field (u or v)."""
    
    title_fs = 14
    label_fs = 12
    tick_fs = 10
    suptitle_fs = 18
    
    actual_modes = basis.shape[1]
    k_modes = actual_modes if n_modes is None else min(int(n_modes), actual_modes)
    
    # Plot 1: Energy decay
    cumulative_energy = np.cumsum(sigma**2) / np.sum(sigma**2)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    
    axes[0].semilogy(sigma)
    axes[0].set_title("Singular Values Decay", fontsize=title_fs)
    axes[0].set_xlabel("Mode Index", fontsize=label_fs)
    axes[0].set_ylabel("Sigma (Log Scale)", fontsize=label_fs)
    axes[0].tick_params(labelsize=tick_fs)
    axes[0].grid(True, which="both", ls="--")
    
    axes[1].plot(cumulative_energy * 100, linewidth=2)
    axes[1].axhline(y=90, color='r', linestyle='--', label='90% energy')
    axes[1].axhline(y=99, color='orange', linestyle='--', label='99% energy')
    axes[1].set_title("Cumulative Energy Captured", fontsize=title_fs)
    axes[1].set_xlabel("Number of Modes", fontsize=label_fs)
    axes[1].set_ylabel("Energy (%)", fontsize=label_fs)
    axes[1].tick_params(labelsize=tick_fs)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_ylim([0, 105])
    
    energy_plot = os.path.join(output_dir, f'svd_{field_prefix}_energy_decay.png')
    fig.savefig(energy_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ {energy_plot}")
    
    # Plot 2: Basis modes
    modes_reshaped = basis[:, :k_modes].reshape(Nt, Nx, Ny, k_modes, order='F')
    t_idx = Nt // 2
    n_modes_to_plot = min(18, k_modes)
    modes_for_plot = modes_reshaped[t_idx, :, :, :n_modes_to_plot]
    mode_abs_max = np.max(np.abs(modes_for_plot)) + 1e-12
    
    fig, axes = plt.subplots(6, 3, figsize=(16, 30), constrained_layout=True)
    mode_im = None
    for i in range(n_modes_to_plot):
        row, col = i // 3, i % 3
        mode_slice = modes_reshaped[t_idx, :, :, i]
        mode_im = axes[row, col].imshow(
            mode_slice,
            cmap='seismic',
            origin='lower',
            vmin=-mode_abs_max,
            vmax=mode_abs_max,
        )
        axes[row, col].set_title(f"SVD Mode {i} (t={t_idx})", fontsize=title_fs)
        axes[row, col].tick_params(labelsize=tick_fs)
    
    for i in range(n_modes_to_plot, 18):
        row, col = i // 3, i % 3
        axes[row, col].axis('off')
    
    if mode_im is not None:
        mode_cbar = fig.colorbar(
            mode_im,
            ax=axes.ravel().tolist(),
            location='right',
            pad=0.01,
            shrink=0.98,
        )
        mode_cbar.set_label('Mode Value', fontsize=label_fs)
        mode_cbar.ax.tick_params(labelsize=tick_fs)
    
    plt.suptitle(
        f"{field_name.title()} SVD Basis Functions ({k_modes} modes)",
        fontsize=suptitle_fs,
    )
    basis_plot = os.path.join(output_dir, f'svd_{field_prefix}_modes.png')
    fig.savefig(basis_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ {basis_plot}")
    
    # Plot 3: Basis and coefficients heatmap
    n_modes_to_show = min(30, k_modes)
    n_samples_to_show = min(20, vt.shape[1])
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
    
    basis_view = basis[:, :n_modes_to_show].T
    im0 = axes[0].imshow(
        basis_view,
        cmap='RdBu_r',
        aspect='auto',
        interpolation='nearest',
    )
    axes[0].set_xlabel('Flattened Space-Time Index', fontsize=label_fs)
    axes[0].set_ylabel('Mode Index', fontsize=label_fs)
    axes[0].set_title(
        f'{field_name.title()} Basis Matrix (first {n_modes_to_show} modes)',
        fontsize=title_fs,
    )
    axes[0].tick_params(labelsize=tick_fs)
    cbar0 = plt.colorbar(im0, ax=axes[0], label='Basis Value')
    cbar0.ax.tick_params(labelsize=tick_fs)
    
    sigma_coeff = sigma[:n_modes_to_show, None] * vt[:n_modes_to_show, :]
    sigma_coeff_view = sigma_coeff[:, :n_samples_to_show]
    im1 = axes[1].imshow(
        sigma_coeff_view,
        cmap='RdBu_r',
        aspect='auto',
        interpolation='nearest',
    )
    axes[1].set_xlabel('Sample Index', fontsize=label_fs)
    axes[1].set_ylabel('Mode Index', fontsize=label_fs)
    axes[1].set_title(
        f'{field_name.title()} Sigma*Coefficients '
        f'(first {n_modes_to_show}x{n_samples_to_show})',
        fontsize=title_fs,
    )
    axes[1].tick_params(labelsize=tick_fs)
    cbar1 = plt.colorbar(im1, ax=axes[1], label='Sigma*Coeff Value')
    cbar1.ax.tick_params(labelsize=tick_fs)
    
    components_plot = os.path.join(output_dir, f'svd_{field_prefix}_components.png')
    fig.savefig(components_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ {components_plot}")
    
    # Plot 4: Reconstruction vs ground truth (if raw data available)
    if raw_data is not None:
        print(f"    • Generating reconstruction comparison...")
        
        # Reshape raw data for comparison
        N_samples = raw_data.shape[0]
        raw_matrix = reshape_for_svd(raw_data)
        U_recon = (basis[:, :k_modes] * sigma[:k_modes]) @ vt[:k_modes, :]
        
        n_samples_to_compare = min(3, N_samples)
        n_times_to_compare = min(3, Nt)
        time_indices = [Nt * k // (n_times_to_compare + 1) for k in range(1, n_times_to_compare + 1)]
        
        fig, axes = plt.subplots(
            n_samples_to_compare * 3,
            n_times_to_compare,
            figsize=(5 * n_times_to_compare, 5 * n_samples_to_compare * 3),
            constrained_layout=True,
        )
        if axes.ndim == 1:
            axes = axes[:, np.newaxis]
        
        # Compute min/max for consistent scaling
        gt_recon_abs_max = 1e-12
        err_abs_max = 1e-12
        
        for s in range(n_samples_to_compare):
            orig_s = raw_matrix[:, s].reshape(Nt, Nx, Ny, order='F')
            recon_s = U_recon[:, s].reshape(Nt, Nx, Ny, order='F')
            for t_col, t_idx_r in enumerate(time_indices):
                orig_slice = orig_s[t_idx_r, :, :]
                recon_slice = recon_s[t_idx_r, :, :]
                err_slice = orig_slice - recon_slice
                
                gt_recon_abs_max = max(
                    gt_recon_abs_max,
                    np.abs(orig_slice).max(),
                    np.abs(recon_slice).max(),
                )
                err_abs_max = max(err_abs_max, np.abs(err_slice).max())
        
        gt_recon_abs_max += 1e-12
        err_abs_max += 1e-12
        
        # Plot comparisons
        gt_recon_im = None
        err_im = None
        gt_recon_axes = []
        err_axes = []
        
        for s in range(n_samples_to_compare):
            orig_s = raw_matrix[:, s].reshape(Nt, Nx, Ny, order='F')
            recon_s = U_recon[:, s].reshape(Nt, Nx, Ny, order='F')
            
            for t_col, t_idx_r in enumerate(time_indices):
                orig_slice = orig_s[t_idx_r, :, :]
                recon_slice = recon_s[t_idx_r, :, :]
                err_slice = orig_slice - recon_slice
                
                row_orig = s * 3
                row_recon = s * 3 + 1
                row_err = s * 3 + 2
                
                gt_recon_im = axes[row_orig, t_col].imshow(
                    orig_slice,
                    cmap='seismic',
                    origin='lower',
                    vmin=-gt_recon_abs_max,
                    vmax=gt_recon_abs_max,
                )
                axes[row_orig, t_col].set_title(f"GT  s={s} t={t_idx_r}", fontsize=title_fs)
                axes[row_orig, t_col].tick_params(labelsize=tick_fs)
                gt_recon_axes.append(axes[row_orig, t_col])
                
                axes[row_recon, t_col].imshow(
                    recon_slice,
                    cmap='seismic',
                    origin='lower',
                    vmin=-gt_recon_abs_max,
                    vmax=gt_recon_abs_max,
                )
                axes[row_recon, t_col].set_title(f"Recon  s={s} t={t_idx_r}", fontsize=title_fs)
                axes[row_recon, t_col].tick_params(labelsize=tick_fs)
                gt_recon_axes.append(axes[row_recon, t_col])
                
                abs_err = np.abs(err_slice)
                err_im = axes[row_err, t_col].imshow(
                    abs_err,
                    cmap='hot',
                    origin='lower',
                    vmin=0.0,
                    vmax=err_abs_max,
                )
                rel_err = abs_err.max() / (np.abs(orig_slice).max() + 1e-12)
                axes[row_err, t_col].set_title(f"|Error|  max_rel={rel_err:.2e}", fontsize=title_fs)
                axes[row_err, t_col].tick_params(labelsize=tick_fs)
                err_axes.append(axes[row_err, t_col])
        
        if gt_recon_im is not None and err_im is not None:
            cbar_gt_recon = fig.colorbar(
                gt_recon_im,
                ax=gt_recon_axes,
                location='right',
                pad=0.01,
                shrink=0.95,
            )
            cbar_gt_recon.set_label('Field Value (GT/Recon)', fontsize=label_fs)
            
            cbar_err = fig.colorbar(
                err_im,
                ax=err_axes,
                location='right',
                pad=0.01,
                shrink=0.95,
            )
            cbar_err.set_label('Absolute Error', fontsize=label_fs)
        
        plt.suptitle(
            f"{field_name.title()} Reconstruction vs Ground Truth ({k_modes} modes)\n"
            "Rows: [GT, Recon, |Error|] per sample",
            fontsize=suptitle_fs,
        )
        recon_plot = os.path.join(output_dir, f'svd_{field_prefix}_reconstruction.png')
        fig.savefig(recon_plot, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    ✓ {recon_plot}")


def main():
    """Main execution."""
    
    # Set data directory
    data_dir = "data"
    
    # Load merged dataset
    U_data, V_data, grid_info = load_merged_data(data_dir)
    
    # Extract and visualize SVD
    svd_result = extract_svd_basis(
        U_data=U_data,
        V_data=V_data,
        n_modes=128,
        output_dir=data_dir,
    )
    
    print(f"\n✓ All figures saved to {data_dir}/")


if __name__ == "__main__":
    main()
