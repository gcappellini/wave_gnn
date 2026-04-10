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
    v_fom: np.ndarray = None,
) -> dict:
    """
    Extract SVD basis from ground truth solutions via randomized SVD.

    When v_fom is provided the function runs a second SVD on the velocity field
    and returns additional keys: 'basis_v', 'singular_values_v', 'coefficients_v'.
    The displacement keys ('basis', 'singular_values', 'coefficients') always
    contain the u results for backward compatibility.
    
    Args:
        u_fom:      Displacement field (Nx, Ny, Nt, N_samples)
        n_modes:    Number of modes to extract (same for u and v)
        visualize:  Generate visualization plots
        output_dir: Directory for saving plots (if None, no plots saved)
        v_fom:      Velocity field (Nx, Ny, Nt, N_samples), optional
    
    Returns:
        dict with keys:
          always:           {'basis', 'singular_values', 'coefficients', 'grid_info'}
          when v_fom given: additionally {'basis_v', 'singular_values_v', 'coefficients_v'}
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
    
    result = {
        'basis': U_basis,
        'singular_values': Sigma,
        'coefficients': VT,
        'grid_info': np.array([Nx, Ny, Nt]),
    }

    # Optional: SVD on velocity field
    if v_fom is not None:
        print("\n" + "-" * 50)
        print("Computing SVD for velocity field (v_fom)...")
        V_matrix = v_fom.reshape(features_dim, N_samples, order='F')

        V_basis, Sigma_v, VT_v = randomized_svd(
            V_matrix,
            n_components=n_modes_to_compute,
            random_state=42
        )
        actual_modes_v = V_basis.shape[1]

        cumulative_energy_v = np.cumsum(Sigma_v**2) / np.sum(Sigma_v**2)
        print(f"✓ v SVD: {actual_modes_v} modes, "
              f"energy={cumulative_energy_v[-1]*100:.2f}%  "
              f"(50 modes: {cumulative_energy_v[min(49, actual_modes_v-1)]*100:.2f}%)")

        result['basis_v'] = V_basis
        result['singular_values_v'] = Sigma_v
        result['coefficients_v'] = VT_v

    if visualize and output_dir:
        visualize_svd_analysis(
            svd_data=result,
            output_dir=output_dir,
            n_modes=n_modes,
            u_fom=u_fom,
            v_fom=v_fom,
        )

    print("\n✓ SVD analysis complete")
    print("=" * 70)

    return result


def visualize_svd_analysis(
    svd_data: dict,
    output_dir: str,
    n_modes: int = None,
    u_fom: np.ndarray = None,
    v_fom: np.ndarray = None,
):
    """Visualize SVD decomposition for displacement and velocity from precomputed data."""

    if not output_dir:
        return {}

    os.makedirs(output_dir, exist_ok=True)

    if 'grid_info' not in svd_data:
        print("Warning: grid_info missing in svd_data; cannot visualize SVD basis.")
        return {}

    Nx, Ny, Nt = [int(v) for v in svd_data['grid_info']]
    stats = []

    print("\nGenerating SVD visualization plots...")

    u_matrix = None
    if u_fom is not None:
        u_matrix = u_fom.reshape(Nx * Ny * Nt, u_fom.shape[3], order='F')
    stats.extend(
        _visualize_single_field_svd(
            field_name='displacement',
            field_prefix='u',
            basis=svd_data['basis'],
            sigma=svd_data['singular_values'],
            vt=svd_data['coefficients'],
            Nx=Nx,
            Ny=Ny,
            Nt=Nt,
            output_dir=output_dir,
            n_modes=n_modes,
            raw_matrix=u_matrix,
        )
    )

    has_velocity = all(
        key in svd_data for key in ('basis_v', 'singular_values_v', 'coefficients_v')
    )
    if has_velocity:
        v_matrix = None
        if v_fom is not None:
            v_matrix = v_fom.reshape(Nx * Ny * Nt, v_fom.shape[3], order='F')
        stats.extend(
            _visualize_single_field_svd(
                field_name='velocity',
                field_prefix='v',
                basis=svd_data['basis_v'],
                sigma=svd_data['singular_values_v'],
                vt=svd_data['coefficients_v'],
                Nx=Nx,
                Ny=Ny,
                Nt=Nt,
                output_dir=output_dir,
                n_modes=n_modes,
                raw_matrix=v_matrix,
            )
        )
    else:
        print("  • Velocity SVD keys not found; skipping velocity visualizations.")

    return _save_magnitude_summary(stats=stats, output_dir=output_dir)


def _visualize_single_field_svd(
    field_name: str,
    field_prefix: str,
    basis: np.ndarray,
    sigma: np.ndarray,
    vt: np.ndarray,
    Nx: int,
    Ny: int,
    Nt: int,
    output_dir: str,
    n_modes: int = None,
    raw_matrix: np.ndarray = None,
):
    """Generate SVD plots and magnitude stats for one field."""

    print(f"\n  • Visualizing {field_name} SVD...")

    title_fs = 14
    label_fs = 12
    tick_fs = 10
    suptitle_fs = 18

    actual_modes = basis.shape[1]
    k_modes = actual_modes if n_modes is None else min(int(n_modes), actual_modes)

    sigma_coeff = sigma[:k_modes, None] * vt[:k_modes, :]
    stats = [
        _compute_magnitude_stats(f"svd_basis_{field_prefix}", basis[:, :k_modes]),
        _compute_magnitude_stats(f"sigma_coeff_{field_prefix}", sigma_coeff),
    ]
    if raw_matrix is not None:
        stats.insert(0, _compute_raw_min_max_stats(f"raw_{field_prefix}", raw_matrix))

    # Reshape basis for visualization
    modes_reshaped = basis[:, :k_modes].reshape(Nx, Ny, Nt, k_modes, order='F')
    t_idx = Nt // 2
    n_modes_to_plot = min(18, k_modes)
    modes_for_plot = modes_reshaped[:, :, t_idx, :n_modes_to_plot]
    mode_abs_max = np.max(np.abs(modes_for_plot)) + 1e-12

    # Plot 1: Basis functions
    fig, axes = plt.subplots(6, 3, figsize=(16, 30), constrained_layout=True)
    mode_im = None
    for i in range(n_modes_to_plot):
        row, col = i // 3, i % 3
        mode_slice = modes_reshaped[:, :, t_idx, i]
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
    basis_plot = os.path.join(output_dir, f'svd_{field_prefix}_modes_check.png')
    fig.savefig(basis_plot, dpi=150)
    plt.close()
    print(f"    ✓ {basis_plot}")

    # Plot 2: Energy decay
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
    fig.savefig(energy_plot, dpi=150)
    plt.close()
    print(f"    ✓ {energy_plot}")

    # Plot 3: Decomposition components (basis and Sigma*coefficients)
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
    cbar0.set_label('Basis Value', fontsize=label_fs)

    sigma_coeff_view = sigma_coeff[:n_modes_to_show, :n_samples_to_show]
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
    cbar1.set_label('Sigma*Coeff Value', fontsize=label_fs)

    components_plot = os.path.join(output_dir, f'svd_{field_prefix}_components.png')
    fig.savefig(components_plot, dpi=150)
    plt.close()
    print(f"    ✓ {components_plot}")

    # Plot 4: Reconstruction vs raw ground truth
    if raw_matrix is None:
        print(
            f"    • Raw {field_name} data not provided; skipping reconstruction-vs-raw plot."
        )
        return stats

    U_recon = (basis[:, :k_modes] * sigma[:k_modes]) @ vt[:k_modes, :]
    n_samples_to_compare = min(3, vt.shape[1])
    n_times_to_compare = min(3, Nt)
    time_indices = [Nt * k // (n_times_to_compare + 1) for k in range(1, n_times_to_compare + 1)]

    fig, axes = plt.subplots(
        n_samples_to_compare * 3,
        n_times_to_compare,
        figsize=(5 * n_times_to_compare, 5 * n_samples_to_compare * 3),
        constrained_layout=True,
    )
    # Ensure axes is always 2D
    if axes.ndim == 1:
        axes = axes[:, np.newaxis]

    for s in range(n_samples_to_compare):
        orig_s = raw_matrix[:, s].reshape(Nx, Ny, Nt, order='F')
        recon_s = U_recon[:, s].reshape(Nx, Ny, Nt, order='F')
        for t_col, t_idx_r in enumerate(time_indices):
            orig_slice = orig_s[:, :, t_idx_r]
            recon_slice = recon_s[:, :, t_idx_r]
            err_slice = orig_slice - recon_slice

            gt_recon_abs = max(np.abs(orig_slice).max(), np.abs(recon_slice).max())
            if s == 0 and t_col == 0:
                gt_recon_abs_max = gt_recon_abs
                err_abs_max = np.abs(err_slice).max()
            else:
                gt_recon_abs_max = max(gt_recon_abs_max, gt_recon_abs)
                err_abs_max = max(err_abs_max, np.abs(err_slice).max())

    gt_recon_abs_max = gt_recon_abs_max + 1e-12
    err_abs_max = err_abs_max + 1e-12

    gt_recon_im = None
    err_im = None
    gt_recon_axes = []
    err_axes = []

    for s in range(n_samples_to_compare):
        orig_s = raw_matrix[:, s].reshape(Nx, Ny, Nt, order='F')
        recon_s = U_recon[:, s].reshape(Nx, Ny, Nt, order='F')
        for t_col, t_idx_r in enumerate(time_indices):
            orig_slice = orig_s[:, :, t_idx_r]
            recon_slice = recon_s[:, :, t_idx_r]
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

    if gt_recon_im is not None and err_im is not None and gt_recon_axes and err_axes:
        cbar_gt_recon = fig.colorbar(
            gt_recon_im,
            ax=gt_recon_axes,
            location='right',
            pad=0.01,
            shrink=0.95,
        )
        cbar_gt_recon.set_label('Field Value (GT/Recon)', fontsize=label_fs)
        cbar_gt_recon.ax.tick_params(labelsize=tick_fs)

        cbar_err = fig.colorbar(
            err_im,
            ax=err_axes,
            location='right',
            pad=0.01,
            shrink=0.95,
        )
        cbar_err.set_label('Absolute Error', fontsize=label_fs)
        cbar_err.ax.tick_params(labelsize=tick_fs)

    plt.suptitle(
        f"{field_name.title()} Reconstruction vs Ground Truth ({k_modes} modes)\n"
        "Rows: [GT, Recon, |Error|] per sample",
        fontsize=suptitle_fs,
    )
    recon_plot = os.path.join(output_dir, f'svd_{field_prefix}_reconstruction_check.png')
    fig.savefig(recon_plot, dpi=150)
    plt.close()
    print(f"    ✓ {recon_plot}")

    return stats


def _compute_magnitude_stats(name: str, values: np.ndarray) -> dict:
    """Compute min/max/mean/std magnitude statistics for an array."""

    abs_values = np.abs(values)
    return {
        'name': name,
        'min': float(np.min(abs_values)),
        'max': float(np.max(abs_values)),
        'mean': float(np.mean(abs_values)),
        'std': float(np.std(abs_values)),
    }


def _compute_raw_min_max_stats(name: str, values: np.ndarray) -> dict:
    """Compute signed min/max statistics for raw field values."""

    return {
        'name': name,
        'min': float(np.min(values)),
        'max': float(np.max(values)),
    }


def _stats_to_summary(stats: list) -> dict:
    """Convert a list of stat dictionaries into a summary keyed by stat name."""

    return {
        stat['name']: {
            key: value
            for key, value in stat.items()
            if key != 'name'
        }
        for stat in stats
    }


def compute_svd_magnitude_summary(
    svd_data: dict,
    n_modes: int = None,
    u_fom: np.ndarray = None,
    v_fom: np.ndarray = None,
    f_fom: np.ndarray = None,
) -> dict:
    """Compute SVD magnitude summary plus raw-field extrema when available."""

    stats = []

    if u_fom is not None:
        stats.append(_compute_raw_min_max_stats('raw_u', u_fom))

    if v_fom is not None:
        stats.append(_compute_raw_min_max_stats('raw_v', v_fom))

    if f_fom is not None:
        stats.append(_compute_raw_min_max_stats('raw_f', f_fom))

    basis = svd_data.get('basis', None)
    sigma = svd_data.get('singular_values', None)
    vt = svd_data.get('coefficients', None)
    if basis is not None and sigma is not None and vt is not None:
        k_u = basis.shape[1] if n_modes is None else min(int(n_modes), basis.shape[1])
        sigma_coeff_u = sigma[:k_u, None] * vt[:k_u, :]
        stats.append(_compute_magnitude_stats('svd_basis_u', basis[:, :k_u]))
        stats.append(_compute_magnitude_stats('sigma_coeff_u', sigma_coeff_u))

    basis_v = svd_data.get('basis_v', None)
    sigma_v = svd_data.get('singular_values_v', None)
    vt_v = svd_data.get('coefficients_v', None)
    if basis_v is not None and sigma_v is not None and vt_v is not None:
        k_v = basis_v.shape[1] if n_modes is None else min(int(n_modes), basis_v.shape[1])
        sigma_coeff_v = sigma_v[:k_v, None] * vt_v[:k_v, :]
        stats.append(_compute_magnitude_stats('svd_basis_v', basis_v[:, :k_v]))
        stats.append(_compute_magnitude_stats('sigma_coeff_v', sigma_coeff_v))

    return _stats_to_summary(stats)


def _save_magnitude_summary(stats: list, output_dir: str):
    """Print and save SVD magnitude summary statistics."""

    if not stats:
        return {}

    print("\nSVD magnitude summary:")
    lines = [
        f"SVD magnitude summary generated at {datetime.now().isoformat()}",
        "",
    ]
    summary_dict = _stats_to_summary(stats)

    for stat in stats:
        metric_parts = []
        for key, value in stat.items():
            if key == 'name':
                continue
            metric_parts.append(f"{key}={value:.3e}")
        row = f"{stat['name']}: " + ", ".join(metric_parts)
        print(f"  {row}")
        lines.append(row)

    summary_path = os.path.join(output_dir, 'svd_magnitude_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("\n".join(lines))

    print(f"  ✓ {summary_path}")

    return summary_dict
