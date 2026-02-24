"""
Validation and Visualization Functions

Generates comparison plots and validation metrics for DeepONet models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path

from .models import MLP, DeepONet


def plot_validation_basic(
    output_dir: str,
    u_fom: np.ndarray = None,
    svd_data: dict = None,
    data_dir: str = None,
    models_dir: str = None,
    device: torch.device = None,
    n_samples_plot: int = 3,
    cfg: dict = None,
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
    
    # ========================================================================
    # 1. SVD RECONSTRUCTION ERROR
    # ========================================================================
    print("\n1. Computing SVD reconstruction...")
    
    U_basis = svd_data['basis']  # (Nx*Ny*Nt, n_modes)
    Sigma = svd_data['singular_values']  # (n_modes,)
    VT = svd_data['coefficients']  # (n_modes, N_samples)
    
    print(f"  U_basis shape: {U_basis.shape}")
    print(f"  Sigma shape: {Sigma.shape}")
    print(f"  VT shape: {VT.shape}")
    
    # Reconstruct: U_recon = U_basis @ diag(Sigma) @ VT
    U_recon = U_basis @ np.diag(Sigma) @ VT  # (Nx*Ny*Nt, N_samples)
    
    # Reshape back to original format
    u_svd = U_recon.reshape(Nx, Ny, Nt, N_samples, order='F')
    
    svd_error = np.mean((u_fom - u_svd) ** 2)
    print(f"  SVD MSE: {svd_error:.6e}")
    
    # Get number of modes from SVD data
    n_modes = Sigma.shape[0]
    
    # ========================================================================
    # 2. PLOT COMPARISONS
    # ========================================================================
    print("\n2. Generating plots...")
    
    if cfg.svd.visualize:
        # Plot: GT vs SVD Reconstruction vs Error
        # Rows: different (time, sample) combinations
        # Columns: GT, SVD Recon, Error
        t_indices = [0, Nt//2, Nt-1]
        t_labels = ['t=0', f't={Nt//2}', f't={Nt-1}']
        
        n_rows = len(t_indices) * n_samples_plot
        n_cols = 3
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        
        row_idx = 0
        for t_i, (t_idx, t_label) in enumerate(zip(t_indices, t_labels)):
            for s_i in range(min(n_samples_plot, N_samples)):
                # Get data for this (time, sample) pair
                gt = u_fom[:, :, t_idx, s_i]
                svd = u_svd[:, :, t_idx, s_i]
                error = np.abs(gt - svd)
                
                # Find common vmin/vmax for GT and SVD
                vmin = min(gt.min(), svd.min())
                vmax = max(gt.max(), svd.max())
                
                # Column 0: Ground Truth
                im0 = axes[row_idx, 0].imshow(gt, cmap='seismic', origin='lower', vmin=vmin, vmax=vmax)
                axes[row_idx, 0].set_title(f'{t_label} Sample {s_i} - GT', fontsize=10)
                axes[row_idx, 0].set_xticks([])
                axes[row_idx, 0].set_yticks([])
                plt.colorbar(im0, ax=axes[row_idx, 0], fraction=0.046)
                
                # Column 1: SVD Reconstruction
                im1 = axes[row_idx, 1].imshow(svd, cmap='seismic', origin='lower', vmin=vmin, vmax=vmax)
                axes[row_idx, 1].set_title(f'{t_label} Sample {s_i} - SVD', fontsize=10)
                axes[row_idx, 1].set_xticks([])
                axes[row_idx, 1].set_yticks([])
                plt.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046)
                
                # Column 2: Error
                im2 = axes[row_idx, 2].imshow(error, cmap='hot', origin='lower')
                axes[row_idx, 2].set_title(f'{t_label} Sample {s_i} - Error', fontsize=10)
                axes[row_idx, 2].set_xticks([])
                axes[row_idx, 2].set_yticks([])
                cbar = plt.colorbar(im2, ax=axes[row_idx, 2], fraction=0.046)
                cbar.set_label('|Error|', fontsize=8)
                
                row_idx += 1
        
        plt.suptitle('SVD Reconstruction Validation\n(Rows: time instant × sample, Columns: GT | SVD | Error)', 
                    fontsize=12, y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'validation_svd_reconstruction.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: validation_svd_reconstruction.png")
    
    # ========================================================================
    # SECTION 3: BRANCH VALIDATION
    # ========================================================================
    if cfg.training.branch_n_epochs > 0:
        plot_branch_validation(output_dir, u_fom, svd_data, models_dir, device)
    
    # ========================================================================
    # SECTION 4: TRUNK VALIDATION
    # ========================================================================
    if cfg.training.trunk_n_epochs > 0:
        plot_trunk_validation(output_dir, svd_data, models_dir, device)
    
    # ========================================================================
    # SECTION 5: DEEPONET VALIDATION
    # ========================================================================
    plot_deeponet_validation(output_dir, u_fom, svd_data, models_dir, device)
    plot_deeponet_test(output_dir, data_dir, models_dir, device)
        
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"SVD Reconstruction MSE: {svd_error:.6e}")
    print("=" * 70)
    print("✓ Validation plots complete")
    print("=" * 70)


def plot_branch_validation(
    output_dir: str,
    u_fom: np.ndarray,
    svd_data: dict,
    models_dir: str,
    device: torch.device = None,
):
    """
    Compare branch network coefficient predictions with SVD coefficients.
    
    Args:
        output_dir: Directory to save plots
        u_fom: Ground truth data (Nx, Ny, Nt, N_samples)
        svd_data: SVD decomposition dict
        models_dir: Path to models directory
        device: Torch device
    """
    
    print("\n" + "=" * 70)
    print("BRANCH VALIDATION: Comparing Coefficients with SVD")
    print("=" * 70)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    output_dir = Path(output_dir)
    branch_checkpoint = output_dir / "branch_svd_free_evolution.pth"
    
    if not branch_checkpoint.exists():
        print(f"⚠ Warning: Branch checkpoint not found: {branch_checkpoint}")
        print("  Skipping branch validation.")
        return
    
    # Load branch model
    print("Loading branch model...")
    ckpt = torch.load(branch_checkpoint, map_location=device, weights_only=False)
    config = ckpt['config']
    state_dict = ckpt['model_state_dict']
    
    # Infer dimensions from state_dict
    first_layer_weight = state_dict['net.0.weight']
    input_dim = first_layer_weight.shape[1]
    hidden_dim = first_layer_weight.shape[0]
    
    last_layer_key = [k for k in state_dict.keys() if k.startswith('net.') and k.endswith('.weight')][-1]
    last_layer_weight = state_dict[last_layer_key]
    output_dim = last_layer_weight.shape[0]
    
    n_layers = len([k for k in state_dict.keys() if k.endswith('.weight')])
    
    print(f"  Detected model architecture:")
    print(f"    Input dim: {input_dim}")
    print(f"    Hidden dim: {hidden_dim}")
    print(f"    Output dim (n_modes): {output_dim}")
    print(f"    N layers: {n_layers}")
    
    # Reconstruct branch model
    branch = MLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_layers=n_layers
    ).to(device)
    
    branch.load_state_dict(state_dict)
    branch.eval()
    
    # Extract initial conditions from u_fom
    Nx, Ny, Nt, N_samples = u_fom.shape
    
    # Extract sensors matching training procedure
    n_sensors_inferred = int(np.sqrt(input_dim))
    sensor_x_indices = np.linspace(0, Nx - 1, n_sensors_inferred, dtype=int)
    sensor_y_indices = np.linspace(0, Ny - 1, n_sensors_inferred, dtype=int)
    
    ic_sensors = []
    for s in range(N_samples):
        u_ic = u_fom[:, :, 0, s]
        sensors = [u_ic[si, sj] for si in sensor_x_indices for sj in sensor_y_indices]
        ic_sensors.append(sensors)
    
    ic_sensors = np.array(ic_sensors)
    
    # Normalize (same as training)
    ic_min = ic_sensors.min(axis=0, keepdims=True)
    ic_max = ic_sensors.max(axis=0, keepdims=True)
    ic_range = ic_max - ic_min
    ic_norm = 2 * (ic_sensors - ic_min) / (ic_range + 1e-10) - 1
    
    ic_tensor = torch.tensor(ic_norm, dtype=torch.float32).to(device)
    
    # Get predictions
    print("Generating predictions...")
    with torch.no_grad():
        coeffs_pred_norm = branch(ic_tensor).cpu().numpy()  # (N_samples, n_modes)
    
    # Get true coefficients from SVD
    VT = svd_data['coefficients'][:output_dim, :]  # (n_modes, N_samples)
    Sigma = svd_data['singular_values'][:output_dim]
    coeffs_true = (Sigma[:, None] * VT).T  # (N_samples, n_modes)
    
    # Normalize true coeffs (same as training)
    coeff_min = coeffs_true.min()
    coeff_max = coeffs_true.max()
    coeff_range = coeff_max - coeff_min
    coeffs_true_norm = 2 * (coeffs_true - coeff_min) / (coeff_range + 1e-10) - 1
    
    branch_error = np.mean((coeffs_true_norm - coeffs_pred_norm) ** 2)
    print(f"  Branch coefficient MSE (normalized): {branch_error:.6e}")
    
    # Create comparison plots
    print("  Generating comparison plot...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 2a: Coefficient scatter (first few modes)
    ax = axes[0, 0]
    modes_to_plot = min(3, output_dim)
    for mode_idx in range(modes_to_plot):
        ax.scatter(coeffs_true_norm[:, mode_idx], coeffs_pred_norm[:, mode_idx], 
                  alpha=0.6, s=30, label=f'Mode {mode_idx}')
    
    # Perfect prediction line
    coeff_range_plot = [coeffs_true_norm[:, :modes_to_plot].min(), coeffs_true_norm[:, :modes_to_plot].max()]
    ax.plot(coeff_range_plot, coeff_range_plot, 'k--', linewidth=2, label='Perfect')
    
    ax.set_xlabel('SVD Coefficients (True, normalized)', fontsize=11)
    ax.set_ylabel('Branch Predictions (normalized)', fontsize=11)
    ax.set_title('Branch vs SVD Coefficients', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 2b: Error by mode
    ax = axes[0, 1]
    mode_errors = np.mean((coeffs_true_norm - coeffs_pred_norm) ** 2, axis=0)
    ax.bar(range(output_dim), mode_errors, color='steelblue', alpha=0.7)
    ax.set_xlabel('Mode Index', fontsize=11)
    ax.set_ylabel('MSE', fontsize=11)
    ax.set_title('Branch Error by Mode', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2c: Coefficient histogram
    ax = axes[1, 0]
    ax.hist(coeffs_true_norm[:, 0], bins=30, alpha=0.6, label='SVD (Mode 0)', color='blue')
    ax.hist(coeffs_pred_norm[:, 0], bins=30, alpha=0.6, label='Branch (Mode 0)', color='red')
    ax.set_xlabel('Coefficient Value (normalized)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Coefficient Distribution (Mode 0)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 2d: Relative error by sample
    ax = axes[1, 1]
    sample_errors = np.mean((coeffs_true_norm - coeffs_pred_norm) ** 2, axis=1)
    ax.plot(sample_errors, 'o-', color='steelblue', markersize=4, linewidth=1.5)
    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('MSE', fontsize=11)
    ax.set_title('Branch Error by Sample', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'validation_branch_coefficients.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: validation_branch_coefficients.png")
    print(f"  ✓ Branch validation complete")


def plot_trunk_validation(
    output_dir: str,
    svd_data: dict,
    models_dir: str,
    device: torch.device = None,
    time_instant: float = 0.33,
):
    """
    Compare trunk network predictions with SVD modes at a given time instant.
    
    Args:
        output_dir: Directory to save plots
        svd_data: SVD decomposition dict
        models_dir: Path to models directory
        device: Torch device
        time_instant: Time instant for comparison (0.0 to 1.0)
    """
    
    print("\n" + "=" * 70)
    print("TRUNK VALIDATION: Comparing with SVD Modes")
    print("=" * 70)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    output_dir = Path(output_dir)
    trunk_checkpoint = output_dir / "trunk_svd_free_evolution.pth"
    
    if not trunk_checkpoint.exists():
        print(f"⚠ Warning: Trunk checkpoint not found: {trunk_checkpoint}")
        print("  Skipping trunk validation.")
        return
    
    # Load trunk model
    print("Loading trunk model...")
    ckpt = torch.load(trunk_checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    config = ckpt.get('config', {})
    
    # Infer dimensions from state_dict
    # First layer: input_dim -> hidden_dim
    first_layer_weight = state_dict['net.0.weight']
    input_dim = first_layer_weight.shape[1]  # Should be 3 (x, y, t)
    trunk_hidden_dim = first_layer_weight.shape[0]
    
    # Last layer: hidden_dim -> output_dim (n_modes)
    last_keys = [k for k in state_dict.keys() if k.endswith('.weight')]
    last_layer_weight = state_dict[last_keys[-1]]
    n_modes = last_layer_weight.shape[0]
    
    # Count layers
    trunk_n_layers = len([k for k in state_dict.keys() if k.endswith('.weight')])
    
    print(f"  Inferred: input_dim={input_dim}, hidden_dim={trunk_hidden_dim}, n_modes={n_modes}, n_layers={trunk_n_layers}")
    
    # Create trunk model
    trunk = MLP(3, trunk_hidden_dim, n_modes, trunk_n_layers).to(device)
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
    
    # Normalization used during training: targets were normalized to [-1, 1]
    # We need to compute the actual range of the SVD basis that was used during training
    targets_min = U_basis_spatial.min()
    targets_max = U_basis_spatial.max()
    targets_range = targets_max - targets_min
    
    # Build coordinate grid
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    t = np.linspace(0, 1, Nt)
    X, Y, T = np.meshgrid(x, y, t, indexing='ij')
    
    # Select time instant
    t_idx = int(time_instant * (Nt - 1))
    t_actual = t[t_idx]
    print(f"  Time instant: {time_instant:.2f} (index {t_idx}, actual value {t_actual:.6f})")
    
    # Get trunk predictions at this time
    coords_t = np.stack([
        X[:, :, t_idx].flatten('F'),
        Y[:, :, t_idx].flatten('F'),
        np.full((Nx * Ny,), t_actual)
    ], axis=1)
    
    coords_tensor = torch.from_numpy(coords_t).float().to(device)
    
    with torch.no_grad():
        trunk_out_norm = trunk(coords_tensor).cpu().numpy()  # (Nx*Ny, n_modes)
    
    # Reshape to spatial: (Nx, Ny, n_modes)
    trunk_modes_norm = trunk_out_norm.reshape(Nx, Ny, n_modes, order='F')
    
    # Denormalize trunk outputs
    trunk_modes = (trunk_modes_norm + 1) * targets_range / 2 + targets_min
    
    print(f"  Trunk predictions shape: {trunk_modes.shape}")
    print(f"  Trunk value range: [{trunk_modes.min():.6f}, {trunk_modes.max():.6f}]")
    print(f"  SVD modes value range: [{modes_svd[:, :, t_idx, :n_modes].min():.6f}, {modes_svd[:, :, t_idx, :n_modes].max():.6f}]")
    
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
            ax_trunk.set_title(f'Trunk Mode {mode_idx}', fontsize=9)
            ax_trunk.set_xticks([])
            ax_trunk.set_yticks([])
            
            # SVD mode (right)
            svd_mode = modes_svd[:, :, t_idx, mode_idx]
            ax_svd = axes[row, 2*pair + 1]
            ax_svd.imshow(svd_mode, cmap='seismic', origin='lower',
                         vmin=vmin_global, vmax=vmax_global)
            ax_svd.set_title(f'SVD Mode {mode_idx}', fontsize=9)
            ax_svd.set_xticks([])
            ax_svd.set_yticks([])
            
            mode_idx += 1
    
    # Add colorbar
    fig.subplots_adjust(right=0.92, hspace=0.3, wspace=0.2)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Mode Value')
    
    plt.suptitle(f'Trunk Network vs SVD Modes Comparison (t={t_actual:.2f})\\n'
                 f'Left: Trunk Prediction | Right: SVD Ground Truth',
                 fontsize=14, y=0.995)
    
    save_path = os.path.join(output_dir, f'validation_trunk_vs_svd_modes.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: validation_trunk_vs_svd_modes.png")
    
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
    print("  ✓ Trunk validation complete")


def plot_deeponet_validation(
    output_dir: str,
    u_fom: np.ndarray,
    svd_data: dict,
    models_dir: str,
    device: torch.device = None,
    sample_idx: int = 0,
    time_instants: list = [0.0, 0.5, 1.0],
):
    """
    Validate full DeepONet predictions against ground truth.
    
    Args:
        output_dir: Directory to save plots
        u_fom: Ground truth data (Nx, Ny, Nt, N_samples)
        svd_data: SVD decomposition dict
        models_dir: Path to models directory
        device: Torch device
        sample_idx: Sample index to validate
        time_instants: List of time instants (0.0 to 1.0)
    """
    
    print("\\n" + "=" * 70)
    print("DEEPONET VALIDATION: Full Reconstruction vs Ground Truth")
    print("=" * 70)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    output_dir = Path(output_dir)
    deeponet_checkpoint = output_dir / "deeponet_free_evolution.pth"
    
    if not deeponet_checkpoint.exists():
        print(f"⚠ Warning: DeepONet checkpoint not found: {deeponet_checkpoint}")
        print("  Skipping DeepONet validation.")
        return
    
    # Load DeepONet model
    print("Loading DeepONet model...")
    deeponet_ckpt = torch.load(deeponet_checkpoint, map_location=device, weights_only=False)
    deeponet_state = deeponet_ckpt.get('model_state_dict', deeponet_ckpt)
    
    # Infer trunk dimensions from state dict
    trunk_first_layer = deeponet_state['trunk.net.0.weight']
    trunk_hidden_dim = trunk_first_layer.shape[0]
    trunk_last_keys = [k for k in deeponet_state.keys() if k.startswith('trunk.') and k.endswith('.weight')]
    n_modes = deeponet_state[trunk_last_keys[-1]].shape[0]
    trunk_n_layers = len(trunk_last_keys)
    
    # Infer branch dimensions from state dict
    branch_first_layer = deeponet_state['branch.net.0.weight']
    branch_input_dim = branch_first_layer.shape[1]
    branch_hidden_dim = branch_first_layer.shape[0]
    branch_last_keys = [k for k in deeponet_state.keys() if k.startswith('branch.') and k.endswith('.weight')]
    branch_output_dim = deeponet_state[branch_last_keys[-1]].shape[0]
    branch_n_layers = len(branch_last_keys)
    
    # Create trunk and branch networks
    trunk_net = MLP(3, trunk_hidden_dim, n_modes, trunk_n_layers).to(device)
    branch_net = MLP(branch_input_dim, branch_hidden_dim, branch_output_dim, branch_n_layers).to(device)
    
    # Create DeepONet model
    deeponet = DeepONet(trunk_net, branch_net).to(device)
    deeponet.load_state_dict(deeponet_state)
    deeponet.eval()
    
    print(f"  DeepONet loaded: {n_modes} modes")
    
    # Get grid dimensions
    grid_info = svd_data['grid_info']
    Nx, Ny, Nt = grid_info.astype(int)
    
    # Extract IC sensors
    n_sensors_inferred = int(np.sqrt(branch_input_dim))
    sensor_x_indices = np.linspace(0, Nx - 1, n_sensors_inferred, dtype=int)
    sensor_y_indices = np.linspace(0, Ny - 1, n_sensors_inferred, dtype=int)
    
    N_samples = u_fom.shape[3]
    
    u_ic = u_fom[:, :, 0, sample_idx]
    sensors = [u_ic[si, sj] for si in sensor_x_indices for sj in sensor_y_indices]
    ic_sensors = np.array(sensors)
    
    # No normalization - use raw IC values
    ic_tensor = torch.from_numpy(ic_sensors).float().unsqueeze(0).to(device)
    
    # Build coordinate grid
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    t = np.linspace(0, 1, Nt)
    X, Y, T = np.meshgrid(x, y, t, indexing='ij')
    
    # Generate predictions at time instants
    print(f"  Generating predictions for sample {sample_idx}...")
    preds = []
    gts = []
    
    for t_val in time_instants:
        t_idx = int(t_val * (Nt - 1))
        t_actual = t[t_idx]
        
        coords_t = np.stack([
            X[:, :, t_idx].flatten('F'),
            Y[:, :, t_idx].flatten('F'),
            np.full((Nx * Ny,), t_actual)
        ], axis=1)
        
        coords_tensor = torch.from_numpy(coords_t).float().to(device)
        ic_batch = ic_tensor.repeat(coords_tensor.shape[0], 1)
        
        # Use DeepONet forward method directly - no denormalization
        with torch.no_grad():
            u_pred = deeponet(ic_batch, coords_tensor).cpu().numpy()  # (Nx*Ny,)
        
        # Reshape
        u_pred = u_pred.reshape(Nx, Ny, order='F')
        u_gt = u_fom[:, :, t_idx, sample_idx]
        
        preds.append(u_pred)
        gts.append(u_gt)
    
    # Plot GT vs Pred vs Error
    fig, axes = plt.subplots(len(time_instants), 3, figsize=(15, 4 * len(time_instants)))
    if len(time_instants) == 1:
        axes = axes[np.newaxis, :]
    
    for i, t_val in enumerate(time_instants):
        gt = gts[i]
        pred = preds[i]
        err = np.abs(pred - gt)
        vmin = min(gt.min(), pred.min())
        vmax = max(gt.max(), pred.max())
        
        # Ground truth
        im0 = axes[i, 0].imshow(gt, cmap='seismic', origin='lower', vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f"Ground Truth (t={t_val:.2f})", fontsize=11)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)
        
        # Prediction
        im1 = axes[i, 1].imshow(pred, cmap='seismic', origin='lower', vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f"DeepONet Prediction (t={t_val:.2f})", fontsize=11)
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)
        
        # Error
        im2 = axes[i, 2].imshow(err, cmap='hot', origin='lower')
        axes[i, 2].set_title(f"Absolute Error (t={t_val:.2f})", fontsize=11)
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)
        
        # Compute metrics
        l2_error = np.linalg.norm(pred - gt) / np.linalg.norm(gt)
        max_error = err.max()
        print(f"  t={t_val:.2f}: L2 Error = {l2_error:.6e}, Max Error = {max_error:.6e}")
    
    plt.suptitle(f'DeepONet Validation (Sample {sample_idx})', fontsize=14, y=0.995)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'validation_deeponet_sample{sample_idx}.png')
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
):
    """
    Test DeepONet predictions against the out-of-range single-sample dataset.
    
    Args:
        output_dir: Directory to save plots
        data_dir: Path to data directory (contains free_evolution_test.mat)
        models_dir: Path to models directory
        device: Torch device
        time_instants: List of time instants (0.0 to 1.0)
    """
    
    print("\n" + "=" * 70)
    print("DEEPONET TEST: Out-of-Range Single Sample")
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
    
    # Load test data
    import h5py
    mat_file = data_dir / "free_evolution_test.mat"
    if not mat_file.exists():
        print(f"⚠ Warning: Test file not found: {mat_file}")
        print("  Skipping DeepONet test.")
        return
    
    with h5py.File(mat_file, 'r') as f:
        u_fom = np.array(f['U_data']).T
    
    Nx, Ny, Nt, N_samples = u_fom.shape
    if N_samples != 1:
        print(f"  Warning: Expected a single test sample, got {N_samples}.")
    
    # Load DeepONet model
    output_dir = Path(output_dir)
    deeponet_checkpoint = output_dir / "deeponet_free_evolution.pth"
    if not deeponet_checkpoint.exists():
        print(f"⚠ Warning: DeepONet checkpoint not found: {deeponet_checkpoint}")
        print("  Skipping DeepONet test.")
        return
    
    print("Loading DeepONet model...")
    deeponet_ckpt = torch.load(deeponet_checkpoint, map_location=device, weights_only=False)
    deeponet_state = deeponet_ckpt.get('model_state_dict', deeponet_ckpt)
    
    # Infer trunk dimensions from state dict
    trunk_first_layer = deeponet_state['trunk.net.0.weight']
    trunk_hidden_dim = trunk_first_layer.shape[0]
    trunk_last_keys = [k for k in deeponet_state.keys() if k.startswith('trunk.') and k.endswith('.weight')]
    n_modes = deeponet_state[trunk_last_keys[-1]].shape[0]
    trunk_n_layers = len(trunk_last_keys)
    
    # Infer branch dimensions from state dict
    branch_first_layer = deeponet_state['branch.net.0.weight']
    branch_input_dim = branch_first_layer.shape[1]
    branch_hidden_dim = branch_first_layer.shape[0]
    branch_last_keys = [k for k in deeponet_state.keys() if k.startswith('branch.') and k.endswith('.weight')]
    branch_output_dim = deeponet_state[branch_last_keys[-1]].shape[0]
    branch_n_layers = len(branch_last_keys)
    
    # Create trunk and branch networks
    trunk_net = MLP(3, trunk_hidden_dim, n_modes, trunk_n_layers).to(device)
    branch_net = MLP(branch_input_dim, branch_hidden_dim, branch_output_dim, branch_n_layers).to(device)
    
    # Create DeepONet model
    deeponet = DeepONet(trunk_net, branch_net).to(device)
    deeponet.load_state_dict(deeponet_state)
    deeponet.eval()
    
    print(f"  DeepONet loaded: {n_modes} modes")
    
    # Extract IC sensors
    n_sensors_inferred = int(np.sqrt(branch_input_dim))
    sensor_x_indices = np.linspace(0, Nx - 1, n_sensors_inferred, dtype=int)
    sensor_y_indices = np.linspace(0, Ny - 1, n_sensors_inferred, dtype=int)
    
    u_ic = u_fom[:, :, 0, 0]
    sensors = [u_ic[si, sj] for si in sensor_x_indices for sj in sensor_y_indices]
    ic_sensors = np.array(sensors)
    
    # No normalization - use raw IC values
    ic_tensor = torch.from_numpy(ic_sensors).float().unsqueeze(0).to(device)
    
    # Build coordinate grid
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    t = np.linspace(0, 1, Nt)
    X, Y, T = np.meshgrid(x, y, t, indexing='ij')
    
    # Generate predictions at time instants
    print("  Generating predictions for test sample...")
    preds = []
    gts = []
    
    for t_val in time_instants:
        t_idx = int(t_val * (Nt - 1))
        t_actual = t[t_idx]
        
        coords_t = np.stack([
            X[:, :, t_idx].flatten('F'),
            Y[:, :, t_idx].flatten('F'),
            np.full((Nx * Ny,), t_actual)
        ], axis=1)
        
        coords_tensor = torch.from_numpy(coords_t).float().to(device)
        ic_batch = ic_tensor.repeat(coords_tensor.shape[0], 1)
        
        # Use DeepONet forward method directly - no denormalization
        with torch.no_grad():
            u_pred = deeponet(ic_batch, coords_tensor).cpu().numpy()  # (Nx*Ny,)
        
        # Reshape
        u_pred = u_pred.reshape(Nx, Ny, order='F')
        u_gt = u_fom[:, :, t_idx, 0]
        
        preds.append(u_pred)
        gts.append(u_gt)
    
    # Plot GT vs Pred vs Error
    fig, axes = plt.subplots(len(time_instants), 3, figsize=(15, 4 * len(time_instants)))
    if len(time_instants) == 1:
        axes = axes[np.newaxis, :]
    
    for i, t_val in enumerate(time_instants):
        gt = gts[i]
        pred = preds[i]
        err = np.abs(pred - gt)
        vmin = min(gt.min(), pred.min())
        vmax = max(gt.max(), pred.max())
        
        # Ground truth
        im0 = axes[i, 0].imshow(gt, cmap='seismic', origin='lower', vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f"Ground Truth (t={t_val:.2f})", fontsize=11)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)
        
        # Prediction
        im1 = axes[i, 1].imshow(pred, cmap='seismic', origin='lower', vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f"DeepONet Prediction (t={t_val:.2f})", fontsize=11)
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)
        
        # Error
        im2 = axes[i, 2].imshow(err, cmap='hot', origin='lower')
        axes[i, 2].set_title(f"Absolute Error (t={t_val:.2f})", fontsize=11)
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)
        
        # Compute metrics
        l2_error = np.linalg.norm(pred - gt) / np.linalg.norm(gt)
        max_error = err.max()
        print(f"  t={t_val:.2f}: L2 Error = {l2_error:.6e}, Max Error = {max_error:.6e}")
    
    plt.suptitle('DeepONet Test: Out-of-Range Sample', fontsize=14, y=0.995)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'validation_deeponet_test.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: validation_deeponet_test.png")
    print("  ✓ DeepONet test complete")
