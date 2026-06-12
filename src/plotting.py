"""
Validation and Visualization Functions

Generates comparison plots and validation metrics for DeepONet models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import torch
import torch.nn as nn
from pathlib import Path

from .models import MLP, DualHeadMLP, DualHeadSensorBranch, DeepONet


def plot_validation_basic(
    output_dir: str,
    u_fom: np.ndarray = None,
    svd_data: dict = None,
    data_dir: str = None,
    models_dir: str = None,
    device: torch.device = None,
    n_samples_plot: int = 3,
    cfg: dict = None,
    problem_type: str = 'free_evolution',
    v_fom: np.ndarray = None,
):
    """
    Generate comprehensive validation plots.
    
    Args:
        output_dir: Directory to save plots
        u_fom: Ground truth data (Nx, Ny, Nt, N_samples)
        svd_data: SVD decomposition dict
        data_dir: Path to data directory
        models_dir: Path to models directory
        device: Torch device
        n_samples_plot: Number of samples to plot
        problem_type: 'free_evolution' or 'constant_force'
    """
    
    print("\n" + "=" * 70)
    print("GENERATING VALIDATION PLOTS")
    print("=" * 70)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Auto-detect paths if not provided
    if data_dir is None:
        script_dir = Path(output_dir).parent.parent
        data_dir = script_dir / "data"
        models_dir = script_dir / "models"
    else:
        data_dir = Path(data_dir)
        models_dir = Path(models_dir)
    
    # Load data if not provided
    if u_fom is None or svd_data is None:
        print("Loading data...")
        import h5py
        
        mat_file = data_dir / "free_evolution.mat"
        if mat_file.exists():
            with h5py.File(mat_file, 'r') as f:
                u_fom = np.array(f['U_data']).T
        
        svd_file = data_dir / "svd_free_evolution.npy"
        if svd_file.exists():
            svd_data = np.load(svd_file, allow_pickle=True).item()
    
    if u_fom is None or svd_data is None:
        print("⚠ Warning: Could not load data. Skipping plots.")
        return
    
    Nx, Ny, Nt, N_samples = u_fom.shape
    n_modes = svd_data['basis'].shape[0]
    
    print(f"Data shape: {u_fom.shape}")
    print(f"SVD modes: {n_modes}")
    
    # # ========================================================================
    # # 1. SVD RECONSTRUCTION ERROR
    # # ========================================================================
    # print("\n1. Computing SVD reconstruction...")
    
    # U_basis = svd_data['basis']  # (Nx*Ny*Nt, n_modes)
    # Sigma = svd_data['singular_values']  # (n_modes,)
    # VT = svd_data['coefficients']  # (n_modes, N_samples)
    
    # print(f"  U_basis shape: {U_basis.shape}")
    # print(f"  Sigma shape: {Sigma.shape}")
    # print(f"  VT shape: {VT.shape}")
    
    # # Reconstruct: U_recon = U_basis @ diag(Sigma) @ VT
    # U_recon = U_basis @ np.diag(Sigma) @ VT  # (Nx*Ny*Nt, N_samples)
    
    # # Reshape back to original format
    # u_svd = U_recon.reshape(Nx, Ny, Nt, N_samples, order='F')
    
    # svd_error = np.mean((u_fom - u_svd) ** 2)
    # print(f"  SVD MSE: {svd_error:.6e}")
    
    # # Get number of modes from SVD data
    # n_modes = Sigma.shape[0]
    
    # # ========================================================================
    # # 2. PLOT COMPARISONS
    # # ========================================================================
    # print("\n2. Generating plots...")
    
    # if cfg.svd.visualize:
    #     # Plot: GT vs SVD Reconstruction vs Error
    #     # Rows: different (time, sample) combinations
    #     # Columns: GT, SVD Recon, Error
    #     t_indices = [0, Nt//2, Nt-1]
    #     t_labels = ['t=0', f't={Nt//2}', f't={Nt-1}']
        
    #     n_rows = len(t_indices) * n_samples_plot
    #     n_cols = 3
        
    #     fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    #     if n_rows == 1:
    #         axes = axes[np.newaxis, :]
        
    #     row_idx = 0
    #     for t_i, (t_idx, t_label) in enumerate(zip(t_indices, t_labels)):
    #         for s_i in range(min(n_samples_plot, N_samples)):
    #             # Get data for this (time, sample) pair
    #             gt = u_fom[:, :, t_idx, s_i]
    #             svd = u_svd[:, :, t_idx, s_i]
    #             error = np.abs(gt - svd)
                
    #             # Find common vmin/vmax for GT and SVD
    #             vmin = min(gt.min(), svd.min())
    #             vmax = max(gt.max(), svd.max())
                
    #             # Column 0: Ground Truth
    #             im0 = axes[row_idx, 0].imshow(gt, cmap='seismic', origin='lower', vmin=vmin, vmax=vmax)
    #             axes[row_idx, 0].set_title(f'{t_label} Sample {s_i} - GT', fontsize=10)
    #             axes[row_idx, 0].set_xticks([])
    #             axes[row_idx, 0].set_yticks([])
    #             plt.colorbar(im0, ax=axes[row_idx, 0], fraction=0.046)
                
    #             # Column 1: SVD Reconstruction
    #             im1 = axes[row_idx, 1].imshow(svd, cmap='seismic', origin='lower', vmin=vmin, vmax=vmax)
    #             axes[row_idx, 1].set_title(f'{t_label} Sample {s_i} - SVD', fontsize=10)
    #             axes[row_idx, 1].set_xticks([])
    #             axes[row_idx, 1].set_yticks([])
    #             plt.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046)
                
    #             # Column 2: Error
    #             im2 = axes[row_idx, 2].imshow(error, cmap='hot', origin='lower')
    #             axes[row_idx, 2].set_title(f'{t_label} Sample {s_i} - Error', fontsize=10)
    #             axes[row_idx, 2].set_xticks([])
    #             axes[row_idx, 2].set_yticks([])
    #             cbar = plt.colorbar(im2, ax=axes[row_idx, 2], fraction=0.046)
    #             cbar.set_label('|Error|', fontsize=8)
                
    #             row_idx += 1
        
    #     plt.suptitle('SVD Reconstruction Validation\n(Rows: time instant × sample, Columns: GT | SVD | Error)', 
    #                 fontsize=12, y=0.995)
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(output_dir, 'validation_svd_reconstruction.png'), dpi=150, bbox_inches='tight')
    #     plt.close()
    #     print(f"  ✓ Saved: validation_svd_reconstruction.png")
    
    # ========================================================================
    # SECTION 3: BRANCH VALIDATION
    # ========================================================================
    # if cfg.training.branch_n_epochs > 0:
    #     plot_branch_validation(output_dir, u_fom, svd_data, models_dir, device, 
    #                           problem_type=problem_type)
    
    # ========================================================================
    # SECTION 4: TRUNK VALIDATION
    # ========================================================================
    # if cfg.training.trunk_n_epochs > 0:
    #     plot_trunk_validation(output_dir, svd_data, models_dir, device, 
    #                          problem_type=problem_type)
    
    # ========================================================================
    # SECTION 5: DEEPONET VALIDATION
    # ========================================================================
    plot_deeponet_validation(output_dir, u_fom, svd_data, models_dir, device, 
                            problem_type=problem_type, v_fom=v_fom)
    plot_deeponet_test(output_dir, data_dir, models_dir, device, problem_type=problem_type)
        
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print("SVD Reconstruction MSE: n/a")
    print("=" * 70)
    print("✓ Validation plots complete")
    print("=" * 70)


def plot_branch_validation(
    output_dir: str,
    u_fom: np.ndarray,
    svd_data: dict,
    models_dir: str,
    device: torch.device = None,
    problem_type: str = 'free_evolution',
    v_fom: np.ndarray = None,
    f_fom: np.ndarray = None,
    sample_idx: int = 2,
):
    """
    Compare branch network coefficient predictions with SVD coefficients.
    
    Args:
        output_dir: Directory to save plots
        u_fom: Ground truth data (Nx, Ny, Nt, N_samples)
        svd_data: SVD decomposition dict
        models_dir: Path to models directory
        device: Torch device
        problem_type: 'free_evolution' or 'constant_force'
    """
    
    print("\n" + "=" * 70)
    print("BRANCH VALIDATION: Comparing Coefficients with SVD")
    print("=" * 70)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    output_dir = Path(output_dir)
    models_dir = Path(models_dir)
    branch_checkpoint = models_dir / f"branch_svd_{problem_type}.pth"

    # Backward compatibility: older runs may have branch checkpoint in output_dir
    if not branch_checkpoint.exists():
        fallback_ckpt = output_dir / f"branch_svd_{problem_type}.pth"
        if fallback_ckpt.exists():
            branch_checkpoint = fallback_ckpt
    
    if not branch_checkpoint.exists():
        print(f"⚠ Warning: Branch checkpoint not found: {branch_checkpoint}")
        print("  Skipping branch validation.")
        return
    
    # Load branch model
    print("Loading branch model...")
    ckpt = torch.load(branch_checkpoint, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    state_dict = ckpt.get('model_state_dict', ckpt)
    input_scale = ckpt.get('input_scale', 1.0)  # Default to 1.0 for backward compatibility
    dual = bool(ckpt.get('dual', False))
    uses_sensor_encoder = bool(ckpt.get('uses_sensor_encoder', False))
    u_output_scale = float(ckpt.get('u_output_scale', 1.0) or 1.0)
    v_output_scale = float(ckpt.get('v_output_scale', 1.0) or 1.0)
    input_normalization = ckpt.get('input_normalization', {})

    def _resolve_raw_range(field_name):
        if field_name == 'raw_u':
            min_key, max_key = 'raw_u_min', 'raw_u_max'
        elif field_name == 'raw_v':
            min_key, max_key = 'raw_v_min', 'raw_v_max'
        else:
            min_key, max_key = 'raw_f_min', 'raw_f_max'

        if min_key in input_normalization and max_key in input_normalization:
            min_value = float(input_normalization[min_key])
            max_value = float(input_normalization[max_key])
        else:
            summary = config.get('svd_magnitude_summary', {})
            try:
                min_value = float(summary[field_name]['min'])
                max_value = float(summary[field_name]['max'])
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    f"Branch checkpoint is missing {field_name}.min/max for input normalization"
                ) from exc

        if max_value <= min_value:
            raise ValueError(
                f"Invalid branch input normalization for {field_name}: "
                f"min={min_value}, max={max_value}"
            )
        return min_value, max_value
    
    # Infer architecture from checkpoint keys
    if dual and uses_sensor_encoder and any(k.startswith('encoder.') for k in state_dict.keys()):
        n_sensors_inferred = int(ckpt.get('n_sensors', config.get('n_sensors', 8)))
        hidden_dim = state_dict['mlp.backbone.0.weight'].shape[0]
        output_dim = state_dict['mlp.head_u.weight'].shape[0]
        n_layers = len([k for k in state_dict.keys() if k.startswith('mlp.backbone.') and k.endswith('.weight')]) + 1
        input_channels = int(state_dict['encoder.0.weight'].shape[1])

        print("  Detected model architecture:")
        print(f"    Type: DualHeadSensorBranch")
        print(f"    Sensors: {n_sensors_inferred}x{n_sensors_inferred}")
        print(f"    Hidden dim: {hidden_dim}")
        print(f"    Output dim (n_modes): {output_dim}")
        print(f"    N layers: {n_layers}")
        print(f"    Input scale: {input_scale}")
        print(f"    Output scales: u={u_output_scale:.3e}, v={v_output_scale:.3e}")

        branch = DualHeadSensorBranch(
            n_sensors=n_sensors_inferred,
            hidden_dim=hidden_dim,
            n_modes=output_dim,
            n_layers=n_layers,
            input_scale=input_scale,
            input_channels=input_channels,
            u_output_scale=u_output_scale,
            v_output_scale=v_output_scale,
        ).to(device)
    elif dual and any(k.startswith('backbone.') for k in state_dict.keys()):
        # Legacy dual-head MLP branch with flattened concatenated sensors
        first_layer_weight = state_dict['backbone.0.weight']
        input_dim = first_layer_weight.shape[1]
        hidden_dim = first_layer_weight.shape[0]
        output_dim = state_dict['head_u.weight'].shape[0]
        n_layers = len([k for k in state_dict.keys() if k.startswith('backbone.') and k.endswith('.weight')]) + 1

        print("  Detected model architecture:")
        print("    Type: DualHeadMLP (legacy)")
        print(f"    Input dim: {input_dim}")
        print(f"    Hidden dim: {hidden_dim}")
        print(f"    Output dim (n_modes): {output_dim}")
        print(f"    N layers: {n_layers}")

        branch = DualHeadMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_modes=output_dim,
            n_layers=n_layers,
            input_scale=input_scale,
            u_output_scale=u_output_scale,
            v_output_scale=v_output_scale,
        ).to(device)
        n_sensors_inferred = int(np.sqrt(input_dim // 2))
    else:
        # Single-head fallback
        first_layer_weight = state_dict['net.0.weight']
        input_dim = first_layer_weight.shape[1]
        hidden_dim = first_layer_weight.shape[0]
        last_layer_key = [k for k in state_dict.keys() if k.startswith('net.') and k.endswith('.weight')][-1]
        last_layer_weight = state_dict[last_layer_key]
        output_dim = last_layer_weight.shape[0]
        n_layers = len([k for k in state_dict.keys() if k.endswith('.weight')])

        print("  Detected model architecture:")
        print("    Type: MLP (single-head)")
        print(f"    Input dim: {input_dim}")
        print(f"    Hidden dim: {hidden_dim}")
        print(f"    Output dim (n_modes): {output_dim}")
        print(f"    N layers: {n_layers}")
        print(f"    Input scale: {input_scale}")

        branch = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            input_scale=input_scale,
        ).to(device)
        n_sensors_inferred = int(np.sqrt(input_dim))
    
    branch.load_state_dict(state_dict)
    branch.eval()
    
    # Extract initial conditions from u_fom
    Nx, Ny, Nt, N_samples = u_fom.shape

    # Extract sensors matching training procedure
    sensor_x_indices = np.linspace(0, Nx - 1, n_sensors_inferred, dtype=int)
    sensor_y_indices = np.linspace(0, Ny - 1, n_sensors_inferred, dtype=int)
    
    u_sensors = []
    for s in range(N_samples):
        u_ic = u_fom[:, :, 0, s]
        sensors = [u_ic[si, sj] for si in sensor_x_indices for sj in sensor_y_indices]
        u_sensors.append(sensors)
    
    u_sensors = np.array(u_sensors)

    raw_u_min, raw_u_max = _resolve_raw_range('raw_u')
    u_norm = 2 * (u_sensors - raw_u_min) / (raw_u_max - raw_u_min + 1e-10) - 1

    if dual:
        if v_fom is None:
            print("⚠ Warning: v_fom not provided; skipping dual branch validation.")
            return

        v_sensors = []
        for s in range(N_samples):
            v_ic = v_fom[:, :, 0, s]
            sensors = [v_ic[si, sj] for si in sensor_x_indices for sj in sensor_y_indices]
            v_sensors.append(sensors)
        v_sensors = np.array(v_sensors)

        raw_v_min, raw_v_max = _resolve_raw_range('raw_v')
        v_norm = 2 * (v_sensors - raw_v_min) / (raw_v_max - raw_v_min + 1e-10) - 1

        if uses_sensor_encoder:
            u_grid = u_norm.reshape(N_samples, n_sensors_inferred, n_sensors_inferred)
            v_grid = v_norm.reshape(N_samples, n_sensors_inferred, n_sensors_inferred)
            channels = [u_grid, v_grid]
            if int(state_dict['encoder.0.weight'].shape[1]) == 3:
                if f_fom is None:
                    raise ValueError("3-channel branch validation requires f_fom")
                f_sensors = []
                for s in range(N_samples):
                    f_field = f_fom[:, :, s]
                    sensors = [f_field[si, sj] for si in sensor_x_indices for sj in sensor_y_indices]
                    f_sensors.append(sensors)
                f_sensors = np.array(f_sensors)
                raw_f_min, raw_f_max = _resolve_raw_range('raw_f')
                f_norm = 2 * (f_sensors - raw_f_min) / (raw_f_max - raw_f_min + 1e-10) - 1
                channels.append(f_norm.reshape(N_samples, n_sensors_inferred, n_sensors_inferred))
            branch_in = np.stack(channels, axis=1)
        else:
            branch_in = np.concatenate([u_norm, v_norm], axis=1)
    else:
        branch_in = u_norm

    ic_tensor = torch.tensor(branch_in, dtype=torch.float32).to(device)
    
    # Get predictions
    print("Generating predictions...")
    with torch.no_grad():
        pred_out = branch(ic_tensor)

    VT = svd_data['coefficients'][:output_dim, :]
    Sigma = svd_data['singular_values'][:output_dim]
    coeffs_true_u = (Sigma[:, None] * VT).T

    if dual:
        coeffs_pred_u = pred_out[0].cpu().numpy()
        coeffs_pred_v = pred_out[1].cpu().numpy()

        VT_v = svd_data['coefficients_v'][:output_dim, :]
        Sigma_v = svd_data['singular_values_v'][:output_dim]
        coeffs_true_v = (Sigma_v[:, None] * VT_v).T

        branch_error_u = np.mean((coeffs_true_u - coeffs_pred_u) ** 2)
        branch_error_v = np.mean((coeffs_true_v - coeffs_pred_v) ** 2)
        print(f"  Branch coefficient MSE (u physical): {branch_error_u:.6e}")
        print(f"  Branch coefficient MSE (v physical): {branch_error_v:.6e}")
    else:
        coeffs_pred_u = pred_out.cpu().numpy()
        branch_error_u = np.mean((coeffs_true_u - coeffs_pred_u) ** 2)
        print(f"  Branch coefficient MSE (physical): {branch_error_u:.6e}")
    
    # Create comparison plots
    print("  Generating comparison plot...")

    def _plot_coeff_panel(panel_axes, coeffs_true, coeffs_pred, label_name):
        mode_errors = np.mean((coeffs_true - coeffs_pred) ** 2, axis=0)
        sorted_mode_indices = np.argsort(mode_errors)

        for ax in panel_axes.flat:
            ax.tick_params(axis='both', labelsize=18)

        selected_modes = []
        selected_mode_specs = [
            ('Best', int(sorted_mode_indices[0]), 'tab:green'),
            ('Worst', int(sorted_mode_indices[-1]), 'tab:red'),
            ('Median', int(sorted_mode_indices[len(sorted_mode_indices) // 2]), 'tab:orange'),
        ]
        for role_name, mode_idx, color in selected_mode_specs:
            if mode_idx not in {item[1] for item in selected_modes}:
                selected_modes.append((role_name, mode_idx, color))

        # Keep histogram aligned with the worst-performing mode in this panel.
        histogram_role_name = 'Worst'
        histogram_mode_idx = int(sorted_mode_indices[-1])

        # 1) Coefficient scatter (best, worst, and median modes by MSE)
        ax = panel_axes[0, 0]
        plotted_indices = [mode_idx for _, mode_idx, _ in selected_modes]
        for role_name, mode_idx, color in selected_modes:
            ax.scatter(
                coeffs_true[:, mode_idx],
                coeffs_pred[:, mode_idx],
                alpha=0.6,
                s=1,
                color=color,
                label='_nolegend_',
            )
        coeff_range_plot = [
            min(coeffs_true[:, plotted_indices].min(), coeffs_pred[:, plotted_indices].min()),
            max(coeffs_true[:, plotted_indices].max(), coeffs_pred[:, plotted_indices].max()),
        ]
        perfect_line = Line2D([0], [0], color='k', linestyle='--', linewidth=2, label='Perfect')
        ax.plot(coeff_range_plot, coeff_range_plot, 'k--', linewidth=2)
        ax.set_xlabel('SVD Coefficients (True, physical)', fontsize=15)
        ax.set_ylabel('Branch Predictions (physical)', fontsize=15)
        ax.set_title(f'{label_name} Branch vs SVD Coefficients', fontsize=20, fontweight='bold')
        ax.legend(handles=[perfect_line], fontsize=18)
        ax.grid(True, alpha=0.3)

        # 2) Error by mode
        ax = panel_axes[0, 1]
        bar_colors = ['lightgray'] * output_dim
        for _, mode_idx, color in selected_modes:
            bar_colors[mode_idx] = color
        ax.bar(range(output_dim), mode_errors, color=bar_colors, alpha=0.8)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.set_xlabel('Mode Index', fontsize=22)
        ax.set_ylabel('MSE', fontsize=22)
        ax.set_title(f'{label_name} Branch Error by Mode', fontsize=18, fontweight='bold')
        legend_handles = [
            Line2D([0], [0], marker='s', linestyle='None', markersize=10, markerfacecolor=color, markeredgecolor=color, label=f'{role_name} Mode {mode_idx}')
            for role_name, mode_idx, color in selected_modes
        ]
        ax.legend(handles=legend_handles, fontsize=18)
        ax.grid(True, alpha=0.3, axis='y')

        # 3) Coefficient histogram for one of the selected ranked modes
        ax = panel_axes[1, 0]
        if label_name == 'u':
            mode_label = 'Deformation'
        else:
            mode_label = 'Velocity'
        ax.hist(
            coeffs_true[:, histogram_mode_idx],
            bins=30,
            alpha=0.6,
            label=f'SVD {mode_label}',
            color='blue',
        )
        ax.hist(
            coeffs_pred[:, histogram_mode_idx],
            bins=30,
            alpha=0.6,
            label=f'Branch {mode_label}',
            color='red',
        )
        ax.set_xlabel('Coefficient Value (physical)', fontsize=15)
        ax.set_ylabel('Count', fontsize=15)
        ax.set_title('Worst Mode distribution', fontsize=20, fontweight='bold')
        ax.legend(fontsize=18)
        ax.grid(True, alpha=0.3)

        # 4) Error by sample
        ax = panel_axes[1, 1]
        sample_errors = np.mean((coeffs_true - coeffs_pred) ** 2, axis=1)
        ax.plot(sample_errors, 'o-', color='steelblue', markersize=4, linewidth=1.5)
        ax.set_xlabel('Sample Index', fontsize=15)
        ax.set_ylabel('MSE', fontsize=15)
        ax.set_title(f'{label_name} Branch Error by Sample', fontsize=18, fontweight='bold')
        ax.grid(True, alpha=0.3)

    if dual:
        fig, axes = plt.subplots(4, 2, figsize=(12, 20))
        _plot_coeff_panel(axes[0:2, :], coeffs_true_u, coeffs_pred_u, 'u')
        _plot_coeff_panel(axes[2:4, :], coeffs_true_v, coeffs_pred_v, 'v')
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        _plot_coeff_panel(axes, coeffs_true_u, coeffs_pred_u, 'u')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'validation_branch_coefficients.pdf')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: validation_branch_coefficients.pdf")

    # --- Per-sample σ·coefficient comparison plot ---
    if sample_idx >= N_samples:
        print(f"  ⚠ sample_idx={sample_idx} out of range (N_samples={N_samples}); skipping per-sample plot.")
    else:
        print(f"  Generating per-sample σ·coefficient plot (sample {sample_idx})...")
        modes_axis = np.arange(output_dim)
        bar_width = 0.4

        if dual:
            fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8))
            for ax2, c_true, c_pred, label_name in [
                (axes2[0], coeffs_true_u, coeffs_pred_u, 'u'),
                (axes2[1], coeffs_true_v, coeffs_pred_v, 'v'),
            ]:
                ax2.bar(modes_axis - bar_width / 2, c_true[sample_idx],
                        width=bar_width, alpha=0.7, label='True (σ·coeff)', color='steelblue')
                ax2.bar(modes_axis + bar_width / 2, c_pred[sample_idx],
                        width=bar_width, alpha=0.7, label='Predicted (σ·coeff)', color='tomato')
                ax2.set_xlabel('Mode Index', fontsize=15)
                ax2.set_ylabel('σ · coefficient', fontsize=15)
                ax2.set_title(
                    f'{label_name}: True vs Predicted σ·Coefficients (Sample {sample_idx})',
                    fontsize=18,
                    fontweight='bold',
                )
                ax2.legend(fontsize=14)
                ax2.grid(True, alpha=0.3, axis='y')
        else:
            fig2, ax2 = plt.subplots(figsize=(12, 4))
            ax2.bar(modes_axis - bar_width / 2, coeffs_true_u[sample_idx],
                    width=bar_width, alpha=0.7, label='True (σ·coeff)', color='steelblue')
            ax2.bar(modes_axis + bar_width / 2, coeffs_pred_u[sample_idx],
                    width=bar_width, alpha=0.7, label='Predicted (σ·coeff)', color='tomato')
            ax2.set_xlabel('Mode Index', fontsize=15)
            ax2.set_ylabel('σ · coefficient', fontsize=15)
            ax2.set_title(
                f'{label_name}: True vs Predicted σ·Coefficients (Sample {sample_idx})',
                fontsize=18,
                fontweight='bold',
            )
            ax2.legend(fontsize=14)
            ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        sample_save_path = os.path.join(output_dir, f'validation_branch_sample{sample_idx}_coefficients.pdf')
        plt.savefig(sample_save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: validation_branch_sample{sample_idx}_coefficients.pdf")

    print(f"  ✓ Branch validation complete")


def plot_trunk_validation(
    output_dir: str,
    svd_data: dict,
    models_dir: str,
    device: torch.device = None,
    time_instant: float = 0.33,
    problem_type: str = 'free_evolution',
):
    """
    Compare trunk network predictions with SVD modes at a given time instant.
    
    Args:
        output_dir: Directory to save plots
        svd_data: SVD decomposition dict
        models_dir: Path to models directory
        device: Torch device
        time_instant: Time instant for comparison (0.0 to 1.0)
        problem_type: 'free_evolution' or 'constant_force'
    """
    
    print("\n" + "=" * 70)
    print("TRUNK VALIDATION: Comparing with SVD Modes")
    print("=" * 70)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    output_dir = Path(output_dir)
    models_dir = Path(models_dir)
    trunk_checkpoint = models_dir / f"trunk_svd_{problem_type}.pth"

    # Backward compatibility: older runs may have the trunk checkpoint in output_dir
    if not trunk_checkpoint.exists():
        fallback_ckpt = output_dir / f"trunk_svd_{problem_type}.pth"
        if fallback_ckpt.exists():
            trunk_checkpoint = fallback_ckpt
    
    if not trunk_checkpoint.exists():
        print(f"⚠ Warning: Trunk checkpoint not found: {trunk_checkpoint}")
        print("  Skipping trunk validation.")
        return
    
    # Load trunk model
    print("Loading trunk model...")
    ckpt = torch.load(trunk_checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    config = ckpt.get('config', {})

    trunk_output_scaled = bool(ckpt.get('trunk_output_scaled', False))
    u_output_scale = float(ckpt.get('u_output_scale', 1.0) or 1.0)
    v_output_scale = float(ckpt.get('v_output_scale', 1.0) or 1.0)

    # Infer architecture from checkpoint keys
    dual = any(k.startswith('backbone.') for k in state_dict.keys()) and 'head_u.weight' in state_dict

    if dual:
        first_layer_weight = state_dict['backbone.0.weight']
        input_dim = first_layer_weight.shape[1]
        trunk_hidden_dim = first_layer_weight.shape[0]
        n_modes = state_dict['head_u.weight'].shape[0]
        n_backbone_linears = len([k for k in state_dict.keys() if k.startswith('backbone.') and k.endswith('.weight')])
        trunk_n_layers = n_backbone_linears + 1

        print(
            f"  Inferred dual trunk: input_dim={input_dim}, hidden_dim={trunk_hidden_dim}, "
            f"n_modes={n_modes}, n_layers={trunk_n_layers}, "
            f"scaled={trunk_output_scaled}, u_scale={u_output_scale:.3e}, v_scale={v_output_scale:.3e}"
        )

        trunk = DualHeadMLP(
            input_dim,
            trunk_hidden_dim,
            n_modes,
            trunk_n_layers,
            u_output_scale=u_output_scale,
            v_output_scale=v_output_scale,
        ).to(device)
    else:
        # Legacy single-head trunk
        first_layer_weight = state_dict['net.0.weight']
        input_dim = first_layer_weight.shape[1]
        trunk_hidden_dim = first_layer_weight.shape[0]
        trunk_net_keys = [k for k in state_dict.keys() if k.startswith('net.') and k.endswith('.weight')]
        n_modes = state_dict[trunk_net_keys[-1]].shape[0]
        trunk_n_layers = len(trunk_net_keys)

        print(f"  Inferred single trunk: input_dim={input_dim}, hidden_dim={trunk_hidden_dim}, n_modes={n_modes}, n_layers={trunk_n_layers}")

        trunk = MLP(input_dim, trunk_hidden_dim, n_modes, trunk_n_layers).to(device)

    trunk.load_state_dict(state_dict)
    trunk.eval()

    print(f"  Trunk loaded: {n_modes} modes, hidden_dim={trunk_hidden_dim}, n_layers={trunk_n_layers}")
    
    # Get SVD modes from basis
    U_basis = svd_data['basis']  # (spatial_time, n_all_modes)
    grid_info = svd_data['grid_info']
    Nx, Ny, Nt = grid_info.astype(int)
    
    # Extract only the spatial-time portion we need
    spatial_size = Nx * Ny * Nt
    U_basis_spatial = U_basis[:spatial_size, :n_modes]  # Get spatial portion for first n_modes
    
    # The SVD basis modes are already in original scale (not normalized)
    # Reshape to (Nx, Ny, Nt, n_modes)
    modes_svd = U_basis_spatial.reshape(Nx, Ny, Nt, n_modes, order='F')

    modes_svd_v = None
    if dual and ('basis_v' in svd_data):
        V_basis = svd_data['basis_v']
        V_basis_spatial = V_basis[:spatial_size, :n_modes]
        modes_svd_v = V_basis_spatial.reshape(Nx, Ny, Nt, n_modes, order='F')
    
    # Legacy normalization parameters (used only for older unscaled checkpoints)
    targets_min = U_basis_spatial.min()
    targets_max = U_basis_spatial.max()
    targets_range = targets_max - targets_min
    
    # Build coordinate grid
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    t = np.linspace(0, 1, Nt)
    X, Y, T = np.meshgrid(x, y, t, indexing='ij')
    
    # Select time instant for side-by-side mode maps.
    t_idx = int(time_instant * (Nt - 1))
    t_actual = t[t_idx]
    print(f"  Time instant: {time_instant:.2f} (index {t_idx}, actual value {t_actual:.6f})")

    # Predict trunk modes on the full space-time grid to compute global validation errors.
    coords_all = np.stack([
        X.flatten('F'),
        Y.flatten('F'),
        T.flatten('F'),
    ], axis=1)
    coords_tensor_all = torch.from_numpy(coords_all).float().to(device)

    with torch.no_grad():
        trunk_out_all = trunk(coords_tensor_all)

    trunk_modes_all_v = None
    if dual:
        trunk_out_all_u = trunk_out_all[0].cpu().numpy()
        trunk_out_all_v = trunk_out_all[1].cpu().numpy()
        if trunk_output_scaled:
            trunk_modes_all = trunk_out_all_u.reshape(Nx, Ny, Nt, n_modes, order='F')
            trunk_modes_all_v = trunk_out_all_v.reshape(Nx, Ny, Nt, n_modes, order='F')
        else:
            # Backward compatibility for older dual checkpoints trained on normalized targets.
            trunk_modes_all_norm = trunk_out_all_u.reshape(Nx, Ny, Nt, n_modes, order='F')
            trunk_modes_all = (trunk_modes_all_norm + 1) * targets_range / 2 + targets_min
            trunk_modes_all_v = None
    else:
        trunk_out_all_np = trunk_out_all.cpu().numpy()
        trunk_modes_all_norm = trunk_out_all_np.reshape(Nx, Ny, Nt, n_modes, order='F')
        trunk_modes_all = (trunk_modes_all_norm + 1) * targets_range / 2 + targets_min

    # Keep selected-time slices for existing side-by-side mode-map plots.
    trunk_modes = trunk_modes_all[:, :, t_idx, :]
    trunk_modes_v = trunk_modes_all_v[:, :, t_idx, :] if trunk_modes_all_v is not None else None

    print(f"  Trunk predictions shape (all time): {trunk_modes_all.shape}")
    print(f"  Trunk value range: [{trunk_modes_all.min():.6f}, {trunk_modes_all.max():.6f}]")
    print(f"  SVD modes value range: [{modes_svd[:, :, :, :n_modes].min():.6f}, {modes_svd[:, :, :, :n_modes].max():.6f}]")

    # Per-mode error histogram (grouped bars) over the full t in [0, 1].
    coeffs_true_u_all = modes_svd[:, :, :, :n_modes].reshape(-1, n_modes, order='F')
    coeffs_pred_u_all = trunk_modes_all.reshape(-1, n_modes, order='F')
    mode_errors_u = np.mean((coeffs_true_u_all - coeffs_pred_u_all) ** 2, axis=0)

    mode_errors_v = None
    has_velocity = dual and modes_svd_v is not None and trunk_modes_all_v is not None
    if has_velocity:
        coeffs_true_v_all = modes_svd_v[:, :, :, :n_modes].reshape(-1, n_modes, order='F')
        coeffs_pred_v_all = trunk_modes_all_v.reshape(-1, n_modes, order='F')
        mode_errors_v = np.mean((coeffs_true_v_all - coeffs_pred_v_all) ** 2, axis=0)

    print("  Generating per-mode error histogram over all time...")
    fig_err, ax_err = plt.subplots(figsize=(12, 5))
    mode_idx_axis = np.arange(n_modes)
    bar_width = 0.42 if has_velocity else 0.7

    if has_velocity:
        ax_err.bar(
            mode_idx_axis - bar_width / 2,
            mode_errors_u,
            width=bar_width,
            alpha=0.8,
            label='Deformation (u)',
            color='steelblue',
        )
        ax_err.bar(
            mode_idx_axis + bar_width / 2,
            mode_errors_v,
            width=bar_width,
            alpha=0.8,
            label='Velocity (v)',
            color='tomato',
        )
    else:
        ax_err.bar(
            mode_idx_axis,
            mode_errors_u,
            width=bar_width,
            alpha=0.85,
            label='Deformation (u)',
            color='steelblue',
        )

    ax_err.set_xticks(mode_idx_axis)
    ax_err.set_xlim(-0.5, n_modes - 0.5)
    ax_err.set_xlabel('Mode Index', fontsize=16, fontweight='bold')
    ax_err.set_ylabel('MSE', fontsize=16, fontweight='bold')
    ax_err.tick_params(axis='both', labelsize=13)
    ax_err.set_title('Trunk Per-Mode Error', fontsize=20, fontweight='bold')
    ax_err.grid(True, alpha=0.3, axis='y')
    ax_err.legend(fontsize=12)

    plt.tight_layout()
    save_stats_path = os.path.join(output_dir, 'validation_trunk_coefficients.png')
    plt.savefig(save_stats_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: validation_trunk_coefficients.png")
    
    # Create comparison plot
    print("  Generating comparison plot...")
    
    # Calculate global vmin/vmax
    all_trunk = trunk_modes.flatten()
    all_svd = modes_svd[:, :, t_idx, :n_modes].flatten()
    all_values = np.concatenate([all_trunk, all_svd])
    vmin_global = all_values.min()
    vmax_global = all_values.max()
    
    # Plot: 9 rows × 4 columns (2 mode pairs per row)
    n_rows = min(9, (n_modes + 1) // 2)
    n_cols = 4
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    
    mode_idx = 0
    for row in range(n_rows):
        for pair in range(2):  # 2 mode pairs per row
            if mode_idx >= n_modes:
                axes[row, 2*pair].axis('off')
                axes[row, 2*pair + 1].axis('off')
                continue
            
            # Trunk mode (left)
            trunk_mode = trunk_modes[:, :, mode_idx]
            ax_trunk = axes[row, 2*pair]
            im = ax_trunk.imshow(trunk_mode, cmap='seismic', origin='lower',
                                vmin=vmin_global, vmax=vmax_global)
            ax_trunk.set_title(f'Trunk Mode {mode_idx}', fontsize=24, fontweight='bold')
            ax_trunk.set_xticks([])
            ax_trunk.set_yticks([])
            
            # SVD mode (right)
            svd_mode = modes_svd[:, :, t_idx, mode_idx]
            ax_svd = axes[row, 2*pair + 1]
            ax_svd.imshow(svd_mode, cmap='seismic', origin='lower',
                         vmin=vmin_global, vmax=vmax_global)
            ax_svd.set_title(f'SVD Mode {mode_idx}', fontsize=24, fontweight='bold')
            ax_svd.set_xticks([])
            ax_svd.set_yticks([])
            
            mode_idx += 1
    
    # Add colorbar
    fig.subplots_adjust(right=0.92, hspace=0.15, wspace=0.06)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    # cbar.set_label('Mode Value', fontsize=28, fontweight='bold')
    cbar.ax.tick_params(labelsize=32)
    
    save_path = os.path.join(output_dir, 'validation_trunk_vs_svd_modes.pdf')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: validation_trunk_vs_svd_modes.pdf")
    
    # Compute error metrics
    print("\\n  Error metrics for each mode:")
    total_l2 = 0.0
    total_max = 0.0
    
    for mode_idx in range(n_modes):
        trunk_mode = trunk_modes[:, :, mode_idx]
        svd_mode = modes_svd[:, :, t_idx, mode_idx]
        
        l2_error = np.linalg.norm(trunk_mode - svd_mode) / (np.linalg.norm(svd_mode) + 1e-10)
        max_error = np.abs(trunk_mode - svd_mode).max()
        
        total_l2 += l2_error
        total_max += max_error
        
        if mode_idx < 5:  # Print first 5 modes
            print(f"    Mode {mode_idx:2d}: Relative L2 = {l2_error:.6e}, Max Abs = {max_error:.6e}")
    
    print(f"  Average Relative L2 Error: {total_l2 / n_modes:.6e}")
    print(f"  Average Max Abs Error: {total_max / n_modes:.6e}")

    if dual:
        if modes_svd_v is None or trunk_modes_v is None:
            print("\n  ⚠ Velocity validation skipped (missing basis_v or incompatible legacy trunk scaling).")
            print("  ✓ Trunk validation complete")
            return

        print("\n  Generating velocity comparison plot...")

        # Calculate global vmin/vmax for velocity panel
        all_trunk_v = trunk_modes_v.flatten()
        all_svd_v = modes_svd_v[:, :, t_idx, :n_modes].flatten()
        all_values_v = np.concatenate([all_trunk_v, all_svd_v])
        vmin_global_v = all_values_v.min()
        vmax_global_v = all_values_v.max()

        fig_v, axes_v = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        if n_rows == 1:
            axes_v = axes_v[np.newaxis, :]

        mode_idx = 0
        for row in range(n_rows):
            for pair in range(2):
                if mode_idx >= n_modes:
                    axes_v[row, 2 * pair].axis('off')
                    axes_v[row, 2 * pair + 1].axis('off')
                    continue

                trunk_mode_v = trunk_modes_v[:, :, mode_idx]
                ax_trunk_v = axes_v[row, 2 * pair]
                im_v = ax_trunk_v.imshow(
                    trunk_mode_v,
                    cmap='viridis',
                    origin='lower',
                    vmin=vmin_global_v,
                    vmax=vmax_global_v,
                )
                ax_trunk_v.set_title(f'Trunk Mode {mode_idx}', fontsize=24, fontweight='bold')
                ax_trunk_v.set_xticks([])
                ax_trunk_v.set_yticks([])

                svd_mode_v = modes_svd_v[:, :, t_idx, mode_idx]
                ax_svd_v = axes_v[row, 2 * pair + 1]
                ax_svd_v.imshow(
                    svd_mode_v,
                    cmap='viridis',
                    origin='lower',
                    vmin=vmin_global_v,
                    vmax=vmax_global_v,
                )
                ax_svd_v.set_title(f'SVD Mode {mode_idx}', fontsize=24, fontweight='bold')
                ax_svd_v.set_xticks([])
                ax_svd_v.set_yticks([])

                mode_idx += 1

        fig_v.subplots_adjust(right=0.92, hspace=0.15, wspace=0.06)
        cbar_ax_v = fig_v.add_axes([0.94, 0.15, 0.015, 0.7])
        cbar_v = fig_v.colorbar(im_v, cax=cbar_ax_v)
        # cbar_v.set_label('Mode Value', fontsize=28, fontweight='bold')
        cbar_v.ax.tick_params(labelsize=32)

        save_path_v = os.path.join(output_dir, 'validation_trunk_vs_svd_modes_velocity.pdf')
        plt.savefig(save_path_v, dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: validation_trunk_vs_svd_modes_velocity.pdf")

        print("\n  Velocity error metrics for each mode:")
        total_l2_v = 0.0
        total_max_v = 0.0

        for mode_idx in range(n_modes):
            trunk_mode_v = trunk_modes_v[:, :, mode_idx]
            svd_mode_v = modes_svd_v[:, :, t_idx, mode_idx]

            l2_error_v = np.linalg.norm(trunk_mode_v - svd_mode_v) / (np.linalg.norm(svd_mode_v) + 1e-10)
            max_error_v = np.abs(trunk_mode_v - svd_mode_v).max()

            total_l2_v += l2_error_v
            total_max_v += max_error_v

            if mode_idx < 5:
                print(f"    V Mode {mode_idx:2d}: Relative L2 = {l2_error_v:.6e}, Max Abs = {max_error_v:.6e}")

        print(f"  Velocity Average Relative L2 Error: {total_l2_v / n_modes:.6e}")
        print(f"  Velocity Average Max Abs Error: {total_max_v / n_modes:.6e}")

    print("  ✓ Trunk validation complete")


def _load_deeponet_from_checkpoint(ckpt_path, device, problem_type='free_evolution'):
    """
    Load a dual-head free-evolution DeepONet from a checkpoint.

    Expects:
      - trunk:    DualHeadMLP  (trunk.backbone.* + trunk.head_u/head_v)
      - branch:   DualHeadSensorBranch (branch_ic.encoder.* + branch_ic.mlp.*)

    Returns:
        deeponet:      loaded, eval-mode model
        n_sensors:     sensor grid side length  (branch input = 2 * n_sensors^2 flattened)
        normalization: dict with raw_u_min/max and raw_v_min/max for input scaling
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)

    # --- Trunk: DualHeadMLP ---
    trunk_backbone_keys = [k for k in state if k.startswith('trunk.backbone.') and k.endswith('.weight')]
    if not trunk_backbone_keys or 'trunk.head_u.weight' not in state:
        raise ValueError(f"Expected a DualHeadMLP trunk (backbone.* + head_u) in {ckpt_path}")
    trunk_hidden_dim = state['trunk.backbone.0.weight'].shape[0]
    n_modes          = state['trunk.head_u.weight'].shape[0]
    trunk_n_layers   = len(trunk_backbone_keys) + 1
    u_output_scale   = float(ckpt.get('u_output_scale', 1.0) or 1.0)
    v_output_scale   = float(ckpt.get('v_output_scale', 1.0) or 1.0)
    trunk_net = DualHeadMLP(
        3, trunk_hidden_dim, n_modes, trunk_n_layers,
        u_output_scale=u_output_scale, v_output_scale=v_output_scale,
    ).to(device)

    # --- Branch: DualHeadSensorBranch ---
    enc_key = 'branch_ic.encoder.0.weight'
    mlp_key = 'branch_ic.mlp.backbone.0.weight'
    if enc_key not in state or mlp_key not in state:
        raise ValueError(f"Expected a DualHeadSensorBranch branch_ic (encoder.* + mlp.*) in {ckpt_path}")
    branch_hidden_dim    = state[mlp_key].shape[0]
    branch_backbone_count = len([k for k in state if k.startswith('branch_ic.mlp.backbone.') and k.endswith('.weight')])
    branch_n_layers      = branch_backbone_count + 1
    encoder_channels     = state[enc_key].shape[0]
    input_channels       = state[enc_key].shape[1]
    n_sensors = int(ckpt.get('config', {}).get('n_sensors') or ckpt.get('n_sensors', 0))
    if n_sensors == 0:
        raise ValueError(f"n_sensors not found in checkpoint: {ckpt_path}")
    input_scale = float(ckpt.get('input_scale', 1.0) or 1.0)
    branch_net = DualHeadSensorBranch(
        n_sensors=n_sensors, hidden_dim=branch_hidden_dim, n_modes=n_modes,
        n_layers=branch_n_layers, input_scale=input_scale,
        input_channels=input_channels,
        encoder_channels=encoder_channels,
    ).to(device)

    deeponet = DeepONet(trunk_net, branch_ic=branch_net, problem_type='free_evolution').to(device)
    deeponet.load_state_dict(state)
    deeponet.eval()

    normalization = dict(ckpt.get('normalization', {}) or {})

    # Backfill raw input normalization for older DeepONet checkpoints.
    # Validation/inference branch input normalization must match branch pretraining.
    has_raw_u = ('raw_u_min' in normalization) and ('raw_u_max' in normalization)
    has_raw_v = ('raw_v_min' in normalization) and ('raw_v_max' in normalization)
    has_raw_f = ('raw_f_min' in normalization) and ('raw_f_max' in normalization)
    if not (has_raw_u and has_raw_v and has_raw_f):
        branch_ckpt_path = ckpt_path.parent / f"branch_svd_{problem_type}.pth"
        if branch_ckpt_path.exists():
            try:
                branch_ckpt = torch.load(branch_ckpt_path, map_location='cpu', weights_only=False)
                input_norm = branch_ckpt.get('input_normalization', {}) or {}
                if 'raw_u_min' in input_norm and 'raw_u_max' in input_norm:
                    normalization['raw_u_min'] = float(input_norm['raw_u_min'])
                    normalization['raw_u_max'] = float(input_norm['raw_u_max'])
                if 'raw_v_min' in input_norm and 'raw_v_max' in input_norm:
                    normalization['raw_v_min'] = float(input_norm['raw_v_min'])
                    normalization['raw_v_max'] = float(input_norm['raw_v_max'])
                if 'raw_f_min' in input_norm and 'raw_f_max' in input_norm:
                    normalization['raw_f_min'] = float(input_norm['raw_f_min'])
                    normalization['raw_f_max'] = float(input_norm['raw_f_max'])
                print("  DeepONet loader: recovered raw_* normalization from branch checkpoint")
            except Exception as exc:
                print(f"  Warning: failed to recover normalization from branch checkpoint: {exc}")

    # Last-resort defaults for legacy checkpoints.
    normalization.setdefault('raw_u_min', float(normalization.get('u_min', 0.0) or 0.0))
    normalization.setdefault('raw_u_max', float(normalization.get('u_max', 1.0) or 1.0))
    normalization.setdefault('raw_v_min', float(normalization.get('v_min', 0.0) or 0.0))
    normalization.setdefault('raw_v_max', float(normalization.get('v_max', 1.0) or 1.0))
    normalization.setdefault('raw_f_min', float(normalization.get('force_measurements_min', -1.0) or -1.0))
    normalization.setdefault('raw_f_max', float(normalization.get('force_measurements_max', 1.0) or 1.0))

    return deeponet, n_sensors, normalization


def _run_deeponet_inference(deeponet, measurements_tensor, coords_np, Nx, Ny, device, dual):
    """
    Run DeepONet inference at a single time slice.

    Args:
        measurements_tensor: (1, meas_dim) sensor values for one sample
        coords_np:           (Nx*Ny, 3) coordinates at one time instant
        dual:                whether model returns (u, v) tuple

    Returns:
        u_pred: (Nx, Ny) displacement prediction
        v_pred: (Nx, Ny) velocity prediction, or None if not dual
    """
    coords_t = torch.from_numpy(coords_np).float().to(device)
    repeat_shape = (coords_t.shape[0],) + (1,) * (measurements_tensor.dim() - 1)
    meas_batch = measurements_tensor.repeat(*repeat_shape)
    with torch.no_grad():
        out = deeponet(meas_batch, coords_t)
    if dual:
        u_pred = out[0].cpu().numpy().reshape(Nx, Ny, order='F')
        v_pred = out[1].cpu().numpy().reshape(Nx, Ny, order='F')
    else:
        u_pred = out.cpu().numpy().reshape(Nx, Ny, order='F')
        v_pred = None
    return u_pred, v_pred


def _load_grid_from_mat(mat_path: Path, Nx: int, Ny: int, Nt: int):
    """Load x/y/t coordinates from .mat if available; otherwise fallback to uniform [0, 1]."""
    import h5py

    x = np.linspace(0.0, 1.0, Nx)
    y = np.linspace(0.0, 1.0, Ny)
    t = np.linspace(0.0, 1.0, Nt)

    if not mat_path.exists():
        return x, y, t

    with h5py.File(mat_path, 'r') as f:
        if 'x_grid' in f:
            x_loaded = np.array(f['x_grid']).reshape(-1)
            if x_loaded.size == Nx:
                x = x_loaded
        if 'y_grid' in f:
            y_loaded = np.array(f['y_grid']).reshape(-1)
            if y_loaded.size == Ny:
                y = y_loaded
        if 'tlist' in f:
            t_loaded = np.array(f['tlist']).reshape(-1)
            if t_loaded.size == Nt:
                t = t_loaded

    return x, y, t


def _plot_field_row(axes_row, gt, pred, label, t_val, field_kind='u', value_limits=None, error_max=None):
    """Plot GT | Pred | |Error| in a single row of axes using shared colorbar limits."""
    err = np.abs(pred - gt)
    if value_limits is None:
        vmin = min(gt.min(), pred.min())
        vmax = max(gt.max(), pred.max())
    else:
        vmin, vmax = value_limits

    if field_kind == 'v':
        value_cmap = 'viridis'
    else:
        value_cmap = 'seismic'
    error_cmap = 'YlOrRd'

    axes_row[0].imshow(gt, cmap=value_cmap, origin='lower', vmin=vmin, vmax=vmax)
    axes_row[0].set_title(f"GT {label} (t={t_val:.2f})", fontsize=10)
    axes_row[0].set_xticks([]); axes_row[0].set_yticks([])
    axes_row[1].imshow(pred, cmap=value_cmap, origin='lower', vmin=vmin, vmax=vmax)
    axes_row[1].set_title(f"Pred {label} (t={t_val:.2f})", fontsize=10)
    axes_row[1].set_xticks([]); axes_row[1].set_yticks([])
    if error_max is None:
        axes_row[2].imshow(err, cmap=error_cmap, origin='lower')
    else:
        axes_row[2].imshow(err, cmap=error_cmap, origin='lower', vmin=0.0, vmax=error_max)
    l2  = np.linalg.norm(pred - gt) / (np.linalg.norm(gt) + 1e-12)
    axes_row[2].set_title(f"|Error| {label}  L2={l2:.2e}", fontsize=10)
    axes_row[2].set_xticks([]); axes_row[2].set_yticks([])
    return l2


def _sample_sensor_grid(field, sensor_x, sensor_y):
    """Sample a full-resolution field on the branch sensor grid."""
    return np.array([field[si, sj] for si in sensor_x for sj in sensor_y], dtype=np.float32)


def _normalize_sensor_values(values, raw_min, raw_max):
    """Map raw sensor values to [-1, 1]."""
    return 2 * (values - raw_min) / (raw_max - raw_min + 1e-10) - 1


def _build_branch_measurement_tensor(
    u_field,
    v_field,
    f_fom,
    sample_idx,
    sensor_x,
    sensor_y,
    raw_u_min,
    raw_u_max,
    raw_v_min,
    raw_v_max,
    raw_f_min,
    raw_f_max,
    input_channels,
    n_sensors,
    device,
):
    """Build normalized branch input tensor from full fields."""
    u_meas_norm = _normalize_sensor_values(
        _sample_sensor_grid(u_field, sensor_x, sensor_y),
        raw_u_min,
        raw_u_max,
    )

    if input_channels == 1:
        meas = u_meas_norm.astype(np.float32)
        return torch.from_numpy(meas).float().unsqueeze(0).to(device)

    v_meas_norm = _normalize_sensor_values(
        _sample_sensor_grid(v_field, sensor_x, sensor_y),
        raw_v_min,
        raw_v_max,
    )

    if input_channels == 3:
        if f_fom is None:
            raise ValueError("3-channel DeepONet rollout requires f_fom")

        if f_fom.ndim == 4:
            f_field = f_fom[:, :, 0, sample_idx]
        else:
            f_field = f_fom[:, :, sample_idx]

        f_meas_norm = _normalize_sensor_values(
            _sample_sensor_grid(f_field, sensor_x, sensor_y),
            raw_f_min,
            raw_f_max,
        )

        meas = np.stack([
            u_meas_norm.reshape(n_sensors, n_sensors),
            v_meas_norm.reshape(n_sensors, n_sensors),
            f_meas_norm.reshape(n_sensors, n_sensors),
        ], axis=0).astype(np.float32)
    else:
        meas = np.concatenate([u_meas_norm, v_meas_norm]).astype(np.float32)

    return torch.from_numpy(meas).float().unsqueeze(0).to(device)


def plot_deeponet_rollout_validation(
    output_dir: str,
    u_fom: np.ndarray,
    svd_data: dict,
    models_dir: str,
    device: torch.device = None,
    sample_idx: int = 0,
    rollout_dt: float = None,
    problem_type: str = 'free_evolution',
    v_fom: np.ndarray = None,
    f_fom: np.ndarray = None,
):
    """Validate iterative one-step DeepONet rollout using predicted u/v as next inputs."""

    print("\n" + "=" * 70)
    print("DEEPONET ROLLOUT VALIDATION: Iterative One-Step Prediction")
    print("=" * 70)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if v_fom is None:
        print("⚠ Warning: Rollout validation requires v_fom for dual-head predictions")
        print("  Skipping rollout validation.")
        return

    output_dir = Path(output_dir)
    ckpt_name = f"deeponet_{problem_type}.pth"
    deeponet_checkpoint = output_dir / ckpt_name
    if not deeponet_checkpoint.exists():
        deeponet_checkpoint = Path(models_dir) / ckpt_name
        if not deeponet_checkpoint.exists():
            print("⚠ Warning: DeepONet checkpoint not found in output_dir or models_dir")
            print("  Skipping rollout validation.")
            return

    deeponet, n_sensors, normalization = _load_deeponet_from_checkpoint(
        deeponet_checkpoint, device, problem_type
    )

    raw_u_min = float(normalization.get('raw_u_min', 0.0))
    raw_u_max = float(normalization.get('raw_u_max', 1.0))
    raw_v_min = float(normalization.get('raw_v_min', 0.0))
    raw_v_max = float(normalization.get('raw_v_max', 1.0))
    raw_f_min = float(normalization.get('raw_f_min', 0.0))
    raw_f_max = float(normalization.get('raw_f_max', 1.0))

    Nx, Ny, Nt, N_samples = u_fom.shape
    if sample_idx >= N_samples:
        raise IndexError(f"sample_idx={sample_idx} out of range for N_samples={N_samples}")
    if Nt < 2:
        print("⚠ Warning: Need at least two time levels for rollout validation")
        print("  Skipping rollout validation.")
        return

    data_dir = Path(models_dir).parent / "data"
    primary_mat = data_dir / "free_evolution.mat"
    fallback_mat = data_dir / f"{problem_type}.mat"
    grid_mat = primary_mat if primary_mat.exists() else fallback_mat
    x, y, t = _load_grid_from_mat(grid_mat, Nx, Ny, Nt)
    X, Y, _ = np.meshgrid(x, y, t, indexing='ij')

    if rollout_dt is None:
        rollout_dt = float(t[1] - t[0])
    if rollout_dt <= 0:
        raise ValueError(f"rollout_dt must be positive, got {rollout_dt}")

    sensor_x = np.linspace(0, Nx - 1, n_sensors, dtype=int)
    sensor_y = np.linspace(0, Ny - 1, n_sensors, dtype=int)
    input_channels = int(getattr(getattr(deeponet, 'branch_ic', None), 'input_channels', 2))

    rollout_targets = np.arange(float(t[0]) + rollout_dt, float(t[-1]) + 0.5 * rollout_dt, rollout_dt)
    step_indices = []
    for target_time in rollout_targets:
        t_idx = int(np.argmin(np.abs(t - target_time)))
        if t_idx <= 0:
            continue
        if step_indices and t_idx == step_indices[-1]:
            continue
        step_indices.append(t_idx)

    if not step_indices:
        print("⚠ Warning: No rollout steps matched the available time grid")
        print("  Skipping rollout validation.")
        return

    print(f"  Generating rollout predictions for sample {sample_idx}...")
    print(f"  Using grid from: {grid_mat}")
    print(f"  Requested rollout_dt={rollout_dt:.6f}")

    current_u = u_fom[:, :, 0, sample_idx].astype(np.float32, copy=True)
    current_v = v_fom[:, :, 0, sample_idx].astype(np.float32, copy=True)
    prev_time = float(t[0])
    rollout_rows = []
    metrics = []

    for step_number, t_idx in enumerate(step_indices, start=1):
        target_time = float(t[t_idx])
        local_dt = target_time - prev_time
        if local_dt <= 0:
            continue

        meas_tensor = _build_branch_measurement_tensor(
            u_field=current_u,
            v_field=current_v,
            f_fom=f_fom,
            sample_idx=sample_idx,
            sensor_x=sensor_x,
            sensor_y=sensor_y,
            raw_u_min=raw_u_min,
            raw_u_max=raw_u_max,
            raw_v_min=raw_v_min,
            raw_v_max=raw_v_max,
            raw_f_min=raw_f_min,
            raw_f_max=raw_f_max,
            input_channels=input_channels,
            n_sensors=n_sensors,
            device=device,
        )

        coords_np = np.stack([
            X[:, :, t_idx].flatten('F'),
            Y[:, :, t_idx].flatten('F'),
            np.full((Nx * Ny,), local_dt, dtype=np.float32),
        ], axis=1)

        u_pred, v_pred = _run_deeponet_inference(deeponet, meas_tensor, coords_np, Nx, Ny, device, True)
        u_gt = u_fom[:, :, t_idx, sample_idx]
        v_gt = v_fom[:, :, t_idx, sample_idx]

        u_l2 = np.linalg.norm(u_pred - u_gt) / (np.linalg.norm(u_gt) + 1e-12)
        v_l2 = np.linalg.norm(v_pred - v_gt) / (np.linalg.norm(v_gt) + 1e-12)
        metrics.append({'t_val': target_time, 'u_l2': u_l2, 'v_l2': v_l2})

        rollout_rows.append({'kind': 'u', 't_val': target_time, 'gt': u_gt, 'pred': u_pred, 'label': 'u'})
        rollout_rows.append({'kind': 'v', 't_val': target_time, 'gt': v_gt, 'pred': v_pred, 'label': 'v'})

        print(
            f"  Step {step_number:02d}: t={prev_time:.6f} -> {target_time:.6f} "
            f"(local_dt={local_dt:.6f}) | u_L2={u_l2:.4e}, v_L2={v_l2:.4e}"
        )

        current_u = u_pred.astype(np.float32, copy=True)
        current_v = v_pred.astype(np.float32, copy=True)
        prev_time = target_time

    snapshot_count = min(4, len(metrics))
    snapshot_step_ids = np.linspace(0, len(metrics) - 1, snapshot_count, dtype=int)
    snapshot_row_ids = []
    for step_id in snapshot_step_ids:
        base_row = 2 * step_id
        snapshot_row_ids.extend([base_row, base_row + 1])
    snapshot_rows = [rollout_rows[row_id] for row_id in snapshot_row_ids]

    n_rows = len(snapshot_rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    u_vals = [arr for row_data in snapshot_rows if row_data['kind'] == 'u' for arr in (row_data['gt'], row_data['pred'])]
    u_limits = (min(arr.min() for arr in u_vals), max(arr.max() for arr in u_vals))

    v_vals = [arr for row_data in snapshot_rows if row_data['kind'] == 'v' for arr in (row_data['gt'], row_data['pred'])]
    v_limits = (min(arr.min() for arr in v_vals), max(arr.max() for arr in v_vals))

    error_max_u = max(
        max(np.abs(row_data['pred'] - row_data['gt']).max() for row_data in snapshot_rows if row_data['kind'] == 'u'),
        1e-12,
    )
    error_max_v = max(
        max(np.abs(row_data['pred'] - row_data['gt']).max() for row_data in snapshot_rows if row_data['kind'] == 'v'),
        1e-12,
    )

    u_rows = []
    v_rows = []
    for row_idx, row_data in enumerate(snapshot_rows):
        if row_data['kind'] == 'u':
            value_limits = u_limits
            error_max = error_max_u
            u_rows.append(row_idx)
        else:
            value_limits = v_limits
            error_max = error_max_v
            v_rows.append(row_idx)

        l2_val = _plot_field_row(
            axes[row_idx],
            row_data['gt'],
            row_data['pred'],
            row_data['label'],
            row_data['t_val'],
            field_kind=row_data['kind'],
            value_limits=value_limits,
            error_max=error_max,
        )
        print(f"  Snapshot t={row_data['t_val']:.2f}  {row_data['kind']}: L2={l2_val:.4e}")

    plt.suptitle(f'Iterative DeepONet Rollout (Sample {sample_idx})', fontsize=13, y=0.998)
    plt.tight_layout(rect=[0.0, 0.0, 0.84, 0.97])

    cbar_x = 0.865
    cbar_w = 0.012
    cbar_h = 0.18
    cbar_gap = 0.03
    cbar_top = 0.93
    cbar_slot = 0

    if u_rows:
        sm_u = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=u_limits[0], vmax=u_limits[1]), cmap='seismic')
        sm_u.set_array([])
        y1 = cbar_top - cbar_slot * (cbar_h + cbar_gap)
        cax_u = fig.add_axes([cbar_x, y1 - cbar_h, cbar_w, cbar_h])
        cb_u = fig.colorbar(sm_u, cax=cax_u)
        cb_u.set_label('Deformation (u)', fontsize=9)
        cbar_slot += 1

    if v_rows:
        sm_v = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=v_limits[0], vmax=v_limits[1]), cmap='viridis')
        sm_v.set_array([])
        y1 = cbar_top - cbar_slot * (cbar_h + cbar_gap)
        cax_v = fig.add_axes([cbar_x, y1 - cbar_h, cbar_w, cbar_h])
        cb_v = fig.colorbar(sm_v, cax=cax_v)
        cb_v.set_label('Velocity (v)', fontsize=9)
        cbar_slot += 1

    sm_err_u = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0.0, vmax=error_max_u), cmap='YlOrRd')
    sm_err_u.set_array([])
    y1 = cbar_top - cbar_slot * (cbar_h + cbar_gap)
    cax_err_u = fig.add_axes([cbar_x, y1 - cbar_h, cbar_w, cbar_h])
    cb_err_u = fig.colorbar(sm_err_u, cax=cax_err_u)
    cb_err_u.set_label('Abs Error (u)', fontsize=9)
    cbar_slot += 1

    sm_err_v = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0.0, vmax=error_max_v), cmap='YlOrRd')
    sm_err_v.set_array([])
    y1 = cbar_top - cbar_slot * (cbar_h + cbar_gap)
    cax_err_v = fig.add_axes([cbar_x, y1 - cbar_h, cbar_w, cbar_h])
    cb_err_v = fig.colorbar(sm_err_v, cax=cax_err_v)
    cb_err_v.set_label('Abs Error (v)', fontsize=9)

    save_path = os.path.join(str(output_dir), f'validation_deeponet_rollout_sample{sample_idx}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: validation_deeponet_rollout_sample{sample_idx}.png")

    metric_times = [item['t_val'] for item in metrics]
    metric_u = [item['u_l2'] for item in metrics]
    metric_v = [item['v_l2'] for item in metrics]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(metric_times, metric_u, marker='o', linewidth=1.8, label='u relative L2')
    ax.plot(metric_times, metric_v, marker='s', linewidth=1.8, label='v relative L2')
    ax.set_xlabel('Time')
    ax.set_ylabel('Relative L2 Error')
    ax.set_title(f'Iterative Rollout Error Accumulation (Sample {sample_idx})')
    ax.grid(True, alpha=0.3)
    ax.legend()
    metric_save_path = os.path.join(str(output_dir), f'validation_deeponet_rollout_metrics_sample{sample_idx}.png')
    plt.tight_layout()
    plt.savefig(metric_save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: validation_deeponet_rollout_metrics_sample{sample_idx}.png")
    print("  ✓ DeepONet rollout validation complete")


def plot_deeponet_validation(
    output_dir: str,
    u_fom: np.ndarray,
    svd_data: dict,
    models_dir: str,
    device: torch.device = None,
    sample_idx: int = 0,
    time_instants: list = [0.0, 0.5, 1.0],
    problem_type: str = 'free_evolution',
    v_fom: np.ndarray = None,
    f_fom: np.ndarray = None,
):
    """
    Validate full DeepONet predictions against ground truth.

    For dual-head models (free_evolution with v_fom) plots both u and v fields.

    Args:
        output_dir:    Directory to save plots
        u_fom:         Ground truth displacement (Nx, Ny, Nt, N_samples)
        svd_data:      SVD decomposition dict
        models_dir:    Path to models directory
        device:        Torch device
        sample_idx:    Sample index to validate
        time_instants: List of time instants (0.0 to 1.0)
        problem_type:  'free_evolution' or 'constant_force'
        v_fom:         Ground truth velocity (Nx, Ny, Nt, N_samples), optional
    """
    
    print("\n" + "=" * 70)
    print("DEEPONET VALIDATION: Full Reconstruction vs Ground Truth")
    print("=" * 70)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    output_dir = Path(output_dir)
    ckpt_name = f"deeponet_{problem_type}.pth"
    deeponet_checkpoint = output_dir / ckpt_name
    if not deeponet_checkpoint.exists():
        deeponet_checkpoint = Path(models_dir) / ckpt_name
        if not deeponet_checkpoint.exists():
            print(f"⚠ Warning: DeepONet checkpoint not found in output_dir or models_dir")
            print("  Skipping DeepONet validation.")
            return
    
    print("Loading DeepONet model...")
    deeponet, n_sensors, normalization = _load_deeponet_from_checkpoint(
        deeponet_checkpoint, device, problem_type
    )
    print(f"  DeepONet loaded — n_sensors={n_sensors}")

    raw_u_min = float(normalization.get('raw_u_min', 0.0))
    raw_u_max = float(normalization.get('raw_u_max', 1.0))
    raw_v_min = float(normalization.get('raw_v_min', 0.0))
    raw_v_max = float(normalization.get('raw_v_max', 1.0))
    raw_f_min = float(normalization.get('raw_f_min', 0.0))
    raw_f_max = float(normalization.get('raw_f_max', 1.0))

    Nx, Ny, Nt, N_samples = u_fom.shape
    if sample_idx >= N_samples:
        raise IndexError(f"sample_idx={sample_idx} out of range for N_samples={N_samples}")

    dual = v_fom is not None

    data_dir = Path(models_dir).parent / "data"
    primary_mat = data_dir / "free_evolution.mat"
    fallback_mat = data_dir / f"{problem_type}.mat"
    grid_mat = primary_mat if primary_mat.exists() else fallback_mat
    x, y, t = _load_grid_from_mat(grid_mat, Nx, Ny, Nt)

    sensor_x = np.linspace(0, Nx - 1, n_sensors, dtype=int)
    sensor_y = np.linspace(0, Ny - 1, n_sensors, dtype=int)

    u_ic = u_fom[:, :, 0, sample_idx]
    u_meas_raw = np.array([u_ic[si, sj] for si in sensor_x for sj in sensor_y], dtype=np.float32)
    u_meas_norm = 2 * (u_meas_raw - raw_u_min) / (raw_u_max - raw_u_min + 1e-10) - 1

    if dual:
        v_ic = v_fom[:, :, 0, sample_idx]
        v_meas_raw = np.array([v_ic[si, sj] for si in sensor_x for sj in sensor_y], dtype=np.float32)
        v_meas_norm = 2 * (v_meas_raw - raw_v_min) / (raw_v_max - raw_v_min + 1e-10) - 1
        input_channels = int(getattr(getattr(deeponet, 'branch_ic', None), 'input_channels', 2))
        if input_channels == 3:
            if f_fom is None:
                raise ValueError("plot_deeponet_validation requires f_fom for 3-channel branch input")
            f_ic = f_fom[:, :, sample_idx]
            f_meas_raw = np.array([f_ic[si, sj] for si in sensor_x for sj in sensor_y], dtype=np.float32)
            f_meas_norm = 2 * (f_meas_raw - raw_f_min) / (raw_f_max - raw_f_min + 1e-10) - 1
            u_grid = u_meas_norm.reshape(n_sensors, n_sensors)
            v_grid = v_meas_norm.reshape(n_sensors, n_sensors)
            f_grid = f_meas_norm.reshape(n_sensors, n_sensors)
            meas = np.stack([u_grid, v_grid, f_grid], axis=0).astype(np.float32)
        else:
            meas = np.concatenate([u_meas_norm, v_meas_norm])
    else:
        meas = u_meas_norm
    meas_tensor = torch.from_numpy(meas).float().unsqueeze(0).to(device)

    X, Y, _ = np.meshgrid(x, y, t, indexing='ij')

    print(f"  Generating DeepONet predictions for sample {sample_idx}...")
    print(f"  Using grid from: {grid_mat}")

    rows_data = []
    for t_val in time_instants:
        t_idx = int(np.argmin(np.abs(t - t_val)))
        t_actual = float(t[t_idx])
        print(f"  Requested t={t_val:.3f} -> nearest t[{t_idx}]={t_actual:.6f}")

        coords_np = np.stack([
            X[:, :, t_idx].flatten('F'),
            Y[:, :, t_idx].flatten('F'),
            np.full((Nx * Ny,), t_actual),
        ], axis=1)

        u_pred, v_pred = _run_deeponet_inference(deeponet, meas_tensor, coords_np, Nx, Ny, device, dual)
        u_gt = u_fom[:, :, t_idx, sample_idx]
        rows_data.append({'kind': 'u', 't_val': t_actual, 'gt': u_gt, 'pred': u_pred, 'label': 'u'})

        if dual and v_pred is not None:
            v_gt = v_fom[:, :, t_idx, sample_idx]
            rows_data.append({'kind': 'v', 't_val': t_actual, 'gt': v_gt, 'pred': v_pred, 'label': 'v'})

    n_rows = len(rows_data)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    u_vals = [arr for row_data in rows_data if row_data['kind'] == 'u' for arr in (row_data['gt'], row_data['pred'])]
    u_limits = (min(arr.min() for arr in u_vals), max(arr.max() for arr in u_vals))

    v_vals = [arr for row_data in rows_data if row_data['kind'] == 'v' for arr in (row_data['gt'], row_data['pred'])]
    v_limits = None
    if v_vals:
        v_limits = (min(arr.min() for arr in v_vals), max(arr.max() for arr in v_vals))

    error_u_vals = [np.abs(row_data['pred'] - row_data['gt']).max() for row_data in rows_data if row_data['kind'] == 'u']
    error_max_u = max(max(error_u_vals), 1e-12)

    error_v_vals = [np.abs(row_data['pred'] - row_data['gt']).max() for row_data in rows_data if row_data['kind'] == 'v']
    error_max_v = max(max(error_v_vals), 1e-12) if error_v_vals else None

    u_rows = []
    v_rows = []
    for row_idx, row_data in enumerate(rows_data):
        if row_data['kind'] == 'u':
            value_limits = u_limits
            error_max = error_max_u
            u_rows.append(row_idx)
        else:
            value_limits = v_limits
            error_max = error_max_v
            v_rows.append(row_idx)

        l2_val = _plot_field_row(
            axes[row_idx],
            row_data['gt'],
            row_data['pred'],
            row_data['label'],
            row_data['t_val'],
            field_kind=row_data['kind'],
            value_limits=value_limits,
            error_max=error_max,
        )
        print(f"  t={row_data['t_val']:.2f}  {row_data['kind']}: L2={l2_val:.4e}")

    plt.suptitle(f'Branch + SVD Basis Reconstruction (Sample {sample_idx})', fontsize=13, y=0.998)
    plt.tight_layout(rect=[0.0, 0.0, 0.84, 0.97])

    cbar_x = 0.865
    cbar_w = 0.012
    cbar_h = 0.18
    cbar_gap = 0.03
    cbar_top = 0.93
    cbar_slot = 0

    if u_rows:
        sm_u = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=u_limits[0], vmax=u_limits[1]), cmap='seismic')
        sm_u.set_array([])
        y1 = cbar_top - cbar_slot * (cbar_h + cbar_gap)
        cax_u = fig.add_axes([cbar_x, y1 - cbar_h, cbar_w, cbar_h])
        cb_u = fig.colorbar(sm_u, cax=cax_u)
        cb_u.set_label('Deformation (u)', fontsize=9)
        cbar_slot += 1

    if v_rows and v_limits is not None:
        sm_v = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=v_limits[0], vmax=v_limits[1]), cmap='viridis')
        sm_v.set_array([])
        y1 = cbar_top - cbar_slot * (cbar_h + cbar_gap)
        cax_v = fig.add_axes([cbar_x, y1 - cbar_h, cbar_w, cbar_h])
        cb_v = fig.colorbar(sm_v, cax=cax_v)
        cb_v.set_label('Velocity (v)', fontsize=9)
        cbar_slot += 1

    sm_err_u = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0.0, vmax=error_max_u), cmap='YlOrRd')
    sm_err_u.set_array([])
    y1 = cbar_top - cbar_slot * (cbar_h + cbar_gap)
    cax_err_u = fig.add_axes([cbar_x, y1 - cbar_h, cbar_w, cbar_h])
    cb_err_u = fig.colorbar(sm_err_u, cax=cax_err_u)
    cb_err_u.set_label('Abs Error (u)', fontsize=9)
    cbar_slot += 1

    if error_max_v is not None:
        sm_err_v = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0.0, vmax=error_max_v), cmap='YlOrRd')
        sm_err_v.set_array([])
        y1 = cbar_top - cbar_slot * (cbar_h + cbar_gap)
        cax_err_v = fig.add_axes([cbar_x, y1 - cbar_h, cbar_w, cbar_h])
        cb_err_v = fig.colorbar(sm_err_v, cax=cax_err_v)
        cb_err_v.set_label('Abs Error (v)', fontsize=9)

    save_path = os.path.join(str(output_dir), f'validation_deeponet_sample{sample_idx}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: validation_deeponet_sample{sample_idx}.png")
    print("  ✓ DeepONet validation complete")


def plot_deeponet_test(
    output_dir: str,
    data_dir: str = None,
    models_dir: str = None,
    device: torch.device = None,
    time_instants: list = [0.0, 0.5, 1.0],
    problem_type: str = 'free_evolution',
):
    """
    Test DeepONet predictions against the out-of-range single-sample dataset.
    """
    
    print("\n" + "=" * 70)
    print(f"DEEPONET TEST: Out-of-Range Single Sample ({problem_type})")
    print("=" * 70)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if data_dir is None:
        script_dir = Path(output_dir).parent.parent
        data_dir = script_dir / "data"
        models_dir = script_dir / "models"
    else:
        data_dir = Path(data_dir)
        models_dir = Path(models_dir)
    
    import h5py
    test_file = data_dir / f"{problem_type}_test.mat"
    if not test_file.exists():
        print(f"⚠ Warning: Test file not found: {test_file}")
        print("  Skipping DeepONet test.")
        return
    
    with h5py.File(test_file, 'r') as f:
        u_fom = np.array(f['U_data']).T
        v_fom = np.array(f['V_data']).T if 'V_data' in f else None
        f_fom = np.array(f['F_data']).T if 'F_data' in f else None
        x_from_file = np.array(f['x_grid']).reshape(-1) if 'x_grid' in f else None
        y_from_file = np.array(f['y_grid']).reshape(-1) if 'y_grid' in f else None
        t_from_file = np.array(f['tlist']).reshape(-1) if 'tlist' in f else None
    
    if u_fom.ndim == 3:
        u_fom = u_fom[..., np.newaxis]
    if v_fom is not None and v_fom.ndim == 3:
        v_fom = v_fom[..., np.newaxis]
    
    Nx, Ny, Nt, N_samples = u_fom.shape
    print(f"  Test data shape: {u_fom.shape}")
    if N_samples != 1:
        print(f"  Warning: Expected 1 test sample, got {N_samples}. Using first only.")
        u_fom = u_fom[..., :1]
        if v_fom is not None:
            v_fom = v_fom[..., :1]
    
    output_dir = Path(output_dir)
    ckpt_name = f"deeponet_{problem_type}.pth"
    deeponet_checkpoint = output_dir / ckpt_name
    if not deeponet_checkpoint.exists():
        deeponet_checkpoint = Path(models_dir) / ckpt_name
        if not deeponet_checkpoint.exists():
            print(f"⚠ Warning: DeepONet checkpoint not found: {ckpt_name}")
            print("  Skipping DeepONet test.")
            return
    
    print("Loading DeepONet model...")
    deeponet, n_sensors, normalization = _load_deeponet_from_checkpoint(
        deeponet_checkpoint, device, problem_type
    )
    print(f"  DeepONet loaded — n_sensors={n_sensors}")

    raw_u_min = float(normalization.get('raw_u_min', 0.0))
    raw_u_max = float(normalization.get('raw_u_max', 1.0))
    raw_v_min = float(normalization.get('raw_v_min', 0.0))
    raw_v_max = float(normalization.get('raw_v_max', 1.0))
    raw_f_min = float(normalization.get('raw_f_min', 0.0))
    raw_f_max = float(normalization.get('raw_f_max', 1.0))
    dual = v_fom is not None

    sensor_x = np.linspace(0, Nx - 1, n_sensors, dtype=int)
    sensor_y = np.linspace(0, Ny - 1, n_sensors, dtype=int)
    
    u_ic = u_fom[:, :, 0, 0]
    u_meas_raw = np.array([u_ic[si, sj] for si in sensor_x for sj in sensor_y], dtype=np.float32)
    u_meas_norm = 2 * (u_meas_raw - raw_u_min) / (raw_u_max - raw_u_min + 1e-10) - 1
    if dual and v_fom is not None:
        v_ic = v_fom[:, :, 0, 0]
        v_meas_raw = np.array([v_ic[si, sj] for si in sensor_x for sj in sensor_y], dtype=np.float32)
        v_meas_norm = 2 * (v_meas_raw - raw_v_min) / (raw_v_max - raw_v_min + 1e-10) - 1
        input_channels = int(getattr(getattr(deeponet, 'branch_ic', None), 'input_channels', 2))
        if input_channels == 3:
            if f_fom is None:
                raise ValueError("plot_deeponet_test requires F_data for 3-channel branch input")
            f_ic = f_fom[:, :, 0]
            f_meas_raw = np.array([f_ic[si, sj] for si in sensor_x for sj in sensor_y], dtype=np.float32)
            f_meas_norm = 2 * (f_meas_raw - raw_f_min) / (raw_f_max - raw_f_min + 1e-10) - 1
            u_grid = u_meas_norm.reshape(n_sensors, n_sensors)
            v_grid = v_meas_norm.reshape(n_sensors, n_sensors)
            f_grid = f_meas_norm.reshape(n_sensors, n_sensors)
            meas = np.stack([u_grid, v_grid, f_grid], axis=0).astype(np.float32)
        else:
            meas = np.concatenate([u_meas_norm, v_meas_norm])
    else:
        meas = u_meas_norm
    meas_tensor = torch.from_numpy(meas).float().unsqueeze(0).to(device)
    
    x = x_from_file if (x_from_file is not None and x_from_file.size == Nx) else np.linspace(0.0, 1.0, Nx)
    y = y_from_file if (y_from_file is not None and y_from_file.size == Ny) else np.linspace(0.0, 1.0, Ny)
    t = t_from_file if (t_from_file is not None and t_from_file.size == Nt) else np.linspace(0.0, 1.0, Nt)
    X, Y, _ = np.meshgrid(x, y, t, indexing='ij')
    
    print("  Generating predictions for test sample...")
    rows_data = []
    for t_val in time_instants:
        t_idx = int(np.argmin(np.abs(t - t_val)))
        t_actual = float(t[t_idx])
        print(f"  Requested t={t_val:.3f} -> nearest t[{t_idx}]={t_actual:.6f}")
        coords_np = np.stack([
            X[:, :, t_idx].flatten('F'),
            Y[:, :, t_idx].flatten('F'),
            np.full((Nx * Ny,), t_actual)
        ], axis=1)

        u_pred, v_pred = _run_deeponet_inference(deeponet, meas_tensor, coords_np, Nx, Ny, device, dual)
        u_gt = u_fom[:, :, t_idx, 0]
        rows_data.append({'kind': 'u', 't_val': t_val, 'gt': u_gt, 'pred': u_pred, 'label': 'u'})
        if dual and v_fom is not None and v_pred is not None:
            v_gt = v_fom[:, :, t_idx, 0]
            rows_data.append({'kind': 'v', 't_val': t_val, 'gt': v_gt, 'pred': v_pred, 'label': 'v'})

    n_rows = len(rows_data)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    u_vals = [arr for row_data in rows_data if row_data['kind'] == 'u' for arr in (row_data['gt'], row_data['pred'])]
    u_limits = (min(arr.min() for arr in u_vals), max(arr.max() for arr in u_vals))

    v_vals = [arr for row_data in rows_data if row_data['kind'] == 'v' for arr in (row_data['gt'], row_data['pred'])]
    v_limits = None
    if v_vals:
        v_limits = (min(arr.min() for arr in v_vals), max(arr.max() for arr in v_vals))

    error_u_vals = [np.abs(row_data['pred'] - row_data['gt']).max() for row_data in rows_data if row_data['kind'] == 'u']
    error_max_u = max(max(error_u_vals), 1e-12)

    error_v_vals = [np.abs(row_data['pred'] - row_data['gt']).max() for row_data in rows_data if row_data['kind'] == 'v']
    error_max_v = max(max(error_v_vals), 1e-12) if error_v_vals else None

    u_rows = []
    v_rows = []
    for row_idx, row_data in enumerate(rows_data):
        if row_data['kind'] == 'u':
            value_limits = u_limits
            error_max = error_max_u
            u_rows.append(row_idx)
        else:
            value_limits = v_limits
            error_max = error_max_v
            v_rows.append(row_idx)

        l2_val = _plot_field_row(
            axes[row_idx],
            row_data['gt'],
            row_data['pred'],
            row_data['label'],
            row_data['t_val'],
            field_kind=row_data['kind'],
            value_limits=value_limits,
            error_max=error_max,
        )
        print(f"  t={row_data['t_val']:.2f}  {row_data['kind']}: L2={l2_val:.4e}")
    
    plt.suptitle(f'DeepONet Test: Out-of-Range Sample ({problem_type})', fontsize=14, y=0.998)
    plt.tight_layout(rect=[0.0, 0.0, 0.84, 0.97])

    cbar_x = 0.865
    cbar_w = 0.012
    cbar_h = 0.18
    cbar_gap = 0.03
    cbar_top = 0.93
    cbar_slot = 0

    if u_rows:
        sm_u = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=u_limits[0], vmax=u_limits[1]), cmap='seismic')
        sm_u.set_array([])
        y1 = cbar_top - cbar_slot * (cbar_h + cbar_gap)
        cax_u = fig.add_axes([cbar_x, y1 - cbar_h, cbar_w, cbar_h])
        cb_u = fig.colorbar(sm_u, cax=cax_u)
        cb_u.set_label('Deformation (u)', fontsize=9)
        cbar_slot += 1

    if v_rows and v_limits is not None:
        sm_v = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=v_limits[0], vmax=v_limits[1]), cmap='viridis')
        sm_v.set_array([])
        y1 = cbar_top - cbar_slot * (cbar_h + cbar_gap)
        cax_v = fig.add_axes([cbar_x, y1 - cbar_h, cbar_w, cbar_h])
        cb_v = fig.colorbar(sm_v, cax=cax_v)
        cb_v.set_label('Velocity (v)', fontsize=9)
        cbar_slot += 1

    sm_err_u = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0.0, vmax=error_max_u), cmap='YlOrRd')
    sm_err_u.set_array([])
    y1 = cbar_top - cbar_slot * (cbar_h + cbar_gap)
    cax_err_u = fig.add_axes([cbar_x, y1 - cbar_h, cbar_w, cbar_h])
    cb_err_u = fig.colorbar(sm_err_u, cax=cax_err_u)
    cb_err_u.set_label('Abs Error (u)', fontsize=9)
    cbar_slot += 1

    if error_max_v is not None:
        sm_err_v = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0.0, vmax=error_max_v), cmap='YlOrRd')
        sm_err_v.set_array([])
        y1 = cbar_top - cbar_slot * (cbar_h + cbar_gap)
        cax_err_v = fig.add_axes([cbar_x, y1 - cbar_h, cbar_w, cbar_h])
        cb_err_v = fig.colorbar(sm_err_v, cax=cax_err_v)
        cb_err_v.set_label('Abs Error (v)', fontsize=9)

    save_path = os.path.join(output_dir, f'validation_deeponet_test_{problem_type}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: validation_deeponet_test_{problem_type}.png")
    print("  ✓ DeepONet test complete")
