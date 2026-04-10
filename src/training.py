"""
DeepONet Training Functions

Provides modular training functions for:
- Trunk network (on SVD basis)
- Branch network (on SVD coefficients)
- Joint DeepONet fine-tuning
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime

from .models import MLP, DualHeadMLP, DualHeadSensorBranch, DeepONet


def _scale_to_ominus1_1(values: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    """Scale values into [-1, 1] using externally supplied bounds."""

    return 2 * (values - min_value) / (max_value - min_value + 1e-10) - 1


def _get_summary_range(summary: dict, field_name: str) -> tuple[float, float]:
    """Read a min/max range from cfg.svd.magnitude_summary."""

    try:
        min_value = float(summary[field_name]['min'])
        max_value = float(summary[field_name]['max'])
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"svd_magnitude_summary must include {field_name}.min and {field_name}.max"
        ) from exc

    if max_value <= min_value:
        raise ValueError(
            f"Invalid range in svd_magnitude_summary for {field_name}: "
            f"min={min_value}, max={max_value}"
        )

    return min_value, max_value


def train_trunk(
    config: dict,
    svd_data: dict,
    device: torch.device,
    output_dir: str = None,
    models_dir: str = None,
    problem_type: str = 'free_evolution',
) -> dict:
    """Train trunk network on SVD basis functions.

    For free_evolution the trunk is a DualHeadMLP trained jointly on u-basis
    and v-basis (both required in svd_data).
    For constant_force the trunk remains a single-output MLP trained on u-basis.
    """
    
    print("\n" + "=" * 70)
    print("TRAINING TRUNK NETWORK (SVD Basis)")
    print("=" * 70)
    
    U_basis = svd_data['basis']
    grid_info = svd_data['grid_info']
    Nx, Ny, Nt = grid_info.astype(int)
    n_space_time = U_basis.shape[0]
    
    n_modes = config['n_modes']
    targets_u_raw = U_basis[:, :n_modes]

    dual = ('basis_v' in svd_data)
    if dual:
        V_basis = svd_data['basis_v']
        targets_v_raw = V_basis[:, :n_modes]
    
    # Generate coordinates
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    t = np.linspace(0, 1, Nt)
    X, Y, T = np.meshgrid(x, y, t, indexing='ij')
    coords = np.stack([X.flatten('F'), Y.flatten('F'), T.flatten('F')], axis=1)
    
    # Sample points for training (if max_trunk_points specified)
    max_trunk_points = config.get('max_trunk_points', None)
    if max_trunk_points and n_space_time > max_trunk_points:
        print(f"Sampling {max_trunk_points} points from {n_space_time} total space-time points")
        sample_idx = np.random.choice(n_space_time, max_trunk_points, replace=False)
        coords = coords[sample_idx]
        targets_u_raw = targets_u_raw[sample_idx]
        if dual:
            targets_v_raw = targets_v_raw[sample_idx]
    else:
        if max_trunk_points:
            print(f"Using all {n_space_time} space-time points (≤ max_trunk_points={max_trunk_points})")
        else:
            print(f"Using all {n_space_time} space-time points")

    if dual:
        # Physical target training for both fields
        targets_u = targets_u_raw
        targets_v = targets_v_raw
    else:
        # Keep legacy normalization for non-dual trunk paths
        u_min = targets_u_raw.min()
        u_max = targets_u_raw.max()
        u_range = u_max - u_min
        targets_u = 2 * (targets_u_raw - u_min) / u_range - 1
    
    # Train/test split
    n_total = coords.shape[0]
    n_train = int(n_total * config['train_test_split'])
    indices = np.random.permutation(n_total)
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    
    coords_train = torch.from_numpy(coords[train_idx]).float().to(device)
    coords_test  = torch.from_numpy(coords[test_idx]).float().to(device)
    tu_train = torch.from_numpy(targets_u[train_idx]).float().to(device)
    tu_test  = torch.from_numpy(targets_u[test_idx]).float().to(device)
    if dual:
        tv_train = torch.from_numpy(targets_v[train_idx]).float().to(device)
        tv_test  = torch.from_numpy(targets_v[test_idx]).float().to(device)

    u_output_scale = 1.0
    v_output_scale = 1.0
    if dual:
        summary = config.get('svd_magnitude_summary', None)
        if summary is None:
            raise ValueError(
                "Missing 'svd_magnitude_summary' in trunk config. "
                "Run SVD analysis and propagate cfg.svd.magnitude_summary into trunk config."
            )
        try:
            u_output_scale = float(summary['svd_basis_u']['max'])
            v_output_scale = float(summary['svd_basis_v']['max'])
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "svd_magnitude_summary must include svd_basis_u.max and svd_basis_v.max"
            ) from exc

        if u_output_scale <= 0 or v_output_scale <= 0:
            raise ValueError(
                f"Invalid output scales from svd_magnitude_summary: "
                f"u={u_output_scale}, v={v_output_scale}. Expected positive values."
            )

        print(f"  Trunk output scales from SVD summary: u={u_output_scale:.6e}, v={v_output_scale:.6e}")
        print(
            f"  Physical target ranges: "
            f"u=[{targets_u.min():.6e}, {targets_u.max():.6e}], "
            f"v=[{targets_v.min():.6e}, {targets_v.max():.6e}]"
        )
    
    # Build model
    if dual:
        trunk = DualHeadMLP(
            3,
            config['trunk_hidden_dim'],
            n_modes,
            config['trunk_n_layers'],
            u_output_scale=u_output_scale,
            v_output_scale=v_output_scale,
        ).to(device)
        print(f"  Using DualHeadMLP trunk (shared backbone, two heads of size {n_modes})")
    else:
        trunk = MLP(3, config['trunk_hidden_dim'], n_modes, config['trunk_n_layers']).to(device)
        print(f"  Using MLP trunk (single head, {n_modes} modes)")

    optimizer = optim.Adam(trunk.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=False)
    criterion = nn.MSELoss()
    
    # Training loop
    if dual:
        train_dataset = TensorDataset(coords_train, tu_train, tv_train)
    else:
        train_dataset = TensorDataset(coords_train, tu_train)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    train_losses, test_losses = [], []
    best_test_loss = float('inf')
    best_state = None
    
    print(f"Training: {len(train_idx)} samples, Testing: {len(test_idx)} samples")
    
    for epoch in range(config['trunk_n_epochs']):
        trunk.train()
        train_loss_epoch = 0.0

        if dual:
            for coords_batch, tu_batch, tv_batch in train_loader:
                optimizer.zero_grad()
                pred_u, pred_v = trunk(coords_batch)
                loss = criterion(pred_u, tu_batch) + criterion(pred_v, tv_batch)
                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item()
        else:
            for coords_batch, tu_batch in train_loader:
                optimizer.zero_grad()
                pred = trunk(coords_batch)
                loss = criterion(pred, tu_batch)
                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item()
        
        train_loss_epoch /= len(train_loader)
        train_losses.append(train_loss_epoch)
        
        trunk.eval()
        with torch.no_grad():
            if dual:
                pred_u_test, pred_v_test = trunk(coords_test)
                test_loss = (criterion(pred_u_test, tu_test) + criterion(pred_v_test, tv_test)).item()
            else:
                pred_test = trunk(coords_test)
                test_loss = criterion(pred_test, tu_test).item()
        test_losses.append(test_loss)
        scheduler.step(test_loss)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = {k: v.cpu().clone() for k, v in trunk.state_dict().items()}
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d}/{config['trunk_n_epochs']} | Train: {train_loss_epoch:.6e} | Test: {test_loss:.6e}")
    
    trunk.load_state_dict(best_state)
    trunk.eval()
    
    # Save checkpoint to models directory (fallback to output directory)
    ckpt_dir = models_dir if models_dir is not None else output_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'trunk_svd_{problem_type}.pth')
    torch.save({
        'model_state_dict': best_state,
        'config': config,
        'dual': dual,
        'u_output_scale': float(u_output_scale) if dual else None,
        'v_output_scale': float(v_output_scale) if dual else None,
        'trunk_output_scaled': bool(dual),
    }, ckpt_path)
    print(f"\n✓ Saved: {ckpt_path}")
    
    # Plot training curves
    if output_dir:
        plt.figure(figsize=(8, 5))
        plt.semilogy(train_losses, label='Train')
        plt.semilogy(test_losses, label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Trunk Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'trunk_training_curves.png'), dpi=150)
        plt.close()
    
    print("✓ Trunk training complete")
    print("=" * 70)
    
    return {'model': trunk, 'train_losses': train_losses, 'test_losses': test_losses}


def train_branch(
    config: dict,
    u_fom: np.ndarray,
    svd_data: dict,
    device: torch.device,
    output_dir: str = None,
    models_dir: str = None,
    problem_type: str = 'free_evolution',
    v_fom: np.ndarray = None,
    f_fom: np.ndarray = None,
) -> dict:
    """Train dual-head sensor branch on IC/force sensors -> (sigma*coeff_u, sigma*coeff_v)."""
    
    print("\n" + "=" * 70)
    print("TRAINING BRANCH NETWORK (Sensors → SVD Coefficients)")
    print("=" * 70)
    
    # Extract dimensions
    Nx, Ny, Nt, N_samples = u_fom.shape
    n_modes = config['n_modes']
    n_sensors = config['n_sensors']

    if v_fom is None or 'coefficients_v' not in svd_data:
        raise ValueError(
            "Dual branch training requires both v_fom and coefficients_v in svd_data"
        )

    use_force_channel = (problem_type == 'constant_force')
    if use_force_channel and f_fom is None:
        raise ValueError("constant_force branch training requires f_fom for the third branch channel")

    summary = config.get('svd_magnitude_summary', None)
    if summary is None:
        raise ValueError("Missing svd_magnitude_summary in branch config")

    raw_u_min, raw_u_max = _get_summary_range(summary, 'raw_u')
    raw_v_min, raw_v_max = _get_summary_range(summary, 'raw_v')
    if use_force_channel:
        raw_f_min, raw_f_max = _get_summary_range(summary, 'raw_f')
    else:
        raw_f_min, raw_f_max = 0.0, 1.0

    u_output_scale = float(summary['sigma_coeff_u']['max'])
    v_output_scale = float(summary['sigma_coeff_v']['max'])
    if u_output_scale <= 0 or v_output_scale <= 0:
        raise ValueError(f"Invalid branch output scales u={u_output_scale}, v={v_output_scale}")
    
    # Sample training samples if max_branch_samples specified
    max_branch_samples = config.get('max_branch_samples', None)
    if max_branch_samples and N_samples > max_branch_samples:
        print(f"Sampling {max_branch_samples} samples from {N_samples} total samples")
        sample_indices = np.random.choice(N_samples, max_branch_samples, replace=False)
        u_fom_subset = u_fom[:, :, :, sample_indices]
        v_fom_subset = v_fom[:, :, :, sample_indices]
        f_fom_subset = f_fom[:, :, sample_indices] if use_force_channel else None
        N_samples_train = max_branch_samples
    else:
        if max_branch_samples:
            print(f"Using all {N_samples} samples (≤ max_branch_samples={max_branch_samples})")
        else:
            print(f"Using all {N_samples} samples")
        u_fom_subset = u_fom
        v_fom_subset = v_fom
        f_fom_subset = f_fom if use_force_channel else None
        N_samples_train = N_samples
        sample_indices = None
    
    sensor_x_indices = np.linspace(0, Nx - 1, n_sensors, dtype=int)
    sensor_y_indices = np.linspace(0, Ny - 1, n_sensors, dtype=int)
    
    # Build u0 sensor measurements
    u_sensors = []
    for s in range(N_samples_train):
        u_ic = u_fom_subset[:, :, 0, s]
        sensors = [u_ic[si, sj] for si in sensor_x_indices for sj in sensor_y_indices]
        u_sensors.append(sensors)
    u_sensors = np.array(u_sensors)  # (N_samples_train, n_sensors^2)

    # Build v0 sensor measurements
    v_sensors = []
    for s in range(N_samples_train):
        v_ic = v_fom_subset[:, :, 0, s]
        sensors = [v_ic[si, sj] for si in sensor_x_indices for sj in sensor_y_indices]
        v_sensors.append(sensors)
    v_sensors = np.array(v_sensors)  # (N_samples_train, n_sensors^2)

    # Normalize sensors with global raw field bounds from cfg.svd.magnitude_summary
    u_s_norm = _scale_to_ominus1_1(u_sensors, raw_u_min, raw_u_max)
    v_s_norm = _scale_to_ominus1_1(v_sensors, raw_v_min, raw_v_max)

    # Base channels: u and v sensor grids
    u_grid = u_s_norm.reshape(N_samples_train, n_sensors, n_sensors)
    v_grid = v_s_norm.reshape(N_samples_train, n_sensors, n_sensors)

    if use_force_channel:
        f_sensors = []
        for s in range(N_samples_train):
            f_field = f_fom_subset[:, :, s]
            sensors = [f_field[si, sj] for si in sensor_x_indices for sj in sensor_y_indices]
            f_sensors.append(sensors)
        f_sensors = np.array(f_sensors)
        f_s_norm = _scale_to_ominus1_1(f_sensors, raw_f_min, raw_f_max)
        f_grid = f_s_norm.reshape(N_samples_train, n_sensors, n_sensors)
        ic_norm = np.stack([u_grid, v_grid, f_grid], axis=1)
        input_channels = 3
    else:
        f_s_norm = None
        ic_norm = np.stack([u_grid, v_grid], axis=1)
        input_channels = 2

    print(
        f"  Branch input raw ranges from cfg: "
        f"u=[{raw_u_min:.6e}, {raw_u_max:.6e}], v=[{raw_v_min:.6e}, {raw_v_max:.6e}]"
        + (f", f=[{raw_f_min:.6e}, {raw_f_max:.6e}]" if use_force_channel else "")
    )
    print(
        f"  Branch normalized input ranges: "
        f"u=[{u_s_norm.min():.6e}, {u_s_norm.max():.6e}], "
        f"v=[{v_s_norm.min():.6e}, {v_s_norm.max():.6e}]"
        + (f", f=[{f_s_norm.min():.6e}, {f_s_norm.max():.6e}]" if use_force_channel else "")
    )
    print(
        f"  Dual-field branch: input tensor = {ic_norm.shape} "
        f"(N, {input_channels}, {n_sensors}, {n_sensors})"
    )
    
    # SVD coefficients as targets (u)
    VT_u = svd_data['coefficients'][:n_modes, :]
    if sample_indices is not None:
        VT_u = VT_u[:, sample_indices]
    Sigma_u = svd_data['singular_values'][:n_modes]
    target_u = (Sigma_u[:, None] * VT_u).T  # (N_samples_train, n_modes)
    target_u_norm = target_u

    VT_v = svd_data['coefficients_v'][:n_modes, :]
    if sample_indices is not None:
        VT_v = VT_v[:, sample_indices]
    Sigma_v = svd_data['singular_values_v'][:n_modes]
    target_v = (Sigma_v[:, None] * VT_v).T  # (N_samples_train, n_modes)
    target_v_norm = target_v
    
    # Train/test split
    indices = np.random.permutation(N_samples_train)
    train_size = int(N_samples_train * config['train_test_split'])
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    
    ic_train   = torch.from_numpy(ic_norm[train_idx]).float().to(device)
    ic_test    = torch.from_numpy(ic_norm[test_idx]).float().to(device)
    tu_train   = torch.from_numpy(target_u_norm[train_idx]).float().to(device)
    tu_test    = torch.from_numpy(target_u_norm[test_idx]).float().to(device)
    tv_train = torch.from_numpy(target_v_norm[train_idx]).float().to(device)
    tv_test  = torch.from_numpy(target_v_norm[test_idx]).float().to(device)

    print(f"  Branch output scales from SVD summary: u={u_output_scale:.6e}, v={v_output_scale:.6e}")
    print(
        f"  Physical target ranges: "
        f"u=[{target_u_norm.min():.6e}, {target_u_norm.max():.6e}], "
        f"v=[{target_v_norm.min():.6e}, {target_v_norm.max():.6e}]"
    )
    
    # Build model
    input_scale = 1.0
    branch = DualHeadSensorBranch(
        n_sensors=n_sensors,
        hidden_dim=config['branch_hidden_dim'],
        n_modes=n_modes,
        n_layers=config['branch_n_layers'],
        input_scale=input_scale,
        input_channels=input_channels,
        u_output_scale=u_output_scale,
        v_output_scale=v_output_scale,
    ).to(device)

    optimizer = optim.Adam(branch.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=False)
    criterion = nn.MSELoss()
    
    # Training loop
    train_dataset = TensorDataset(ic_train, tu_train, tv_train)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    train_losses, test_losses = [], []
    best_test_loss = float('inf')
    best_state = None
    
    print(f"Training: {len(train_idx)} samples, Testing: {len(test_idx)} samples")
    
    for epoch in range(config['branch_n_epochs']):
        branch.train()
        train_loss_epoch = 0.0

        for ic_batch, tu_batch, tv_batch in train_loader:
            optimizer.zero_grad()
            pred_u, pred_v = branch(ic_batch)
            loss = criterion(pred_u, tu_batch) + criterion(pred_v, tv_batch)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()
        
        train_loss_epoch /= len(train_loader)
        train_losses.append(train_loss_epoch)
        
        branch.eval()
        with torch.no_grad():
            pu_test, pv_test = branch(ic_test)
            test_loss = (criterion(pu_test, tu_test) + criterion(pv_test, tv_test)).item()
        test_losses.append(test_loss)
        scheduler.step(test_loss)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = {k: v.cpu().clone() for k, v in branch.state_dict().items()}
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d}/{config['branch_n_epochs']} | Train: {train_loss_epoch:.6e} | Test: {test_loss:.6e}")
    
    branch.load_state_dict(best_state)
    branch.eval()
    
    # Save checkpoint to models directory (fallback to output directory)
    ckpt_dir = models_dir if models_dir is not None else output_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'branch_svd_{problem_type}.pth')
    torch.save({
        'model_state_dict': best_state,
        'config': config,
        'input_scale': input_scale,
        'dual': True,
        'branch_output_scaled': True,
        'u_output_scale': float(u_output_scale),
        'v_output_scale': float(v_output_scale),
        'n_sensors': int(n_sensors),
        'n_input_channels': int(input_channels),
        'uses_sensor_encoder': True,
        'input_normalization': {
            'raw_u_min': float(raw_u_min),
            'raw_u_max': float(raw_u_max),
            'raw_v_min': float(raw_v_min),
            'raw_v_max': float(raw_v_max),
            'raw_f_min': float(raw_f_min) if use_force_channel else None,
            'raw_f_max': float(raw_f_max) if use_force_channel else None,
        },
    }, ckpt_path)
    print(f"\n✓ Saved: {ckpt_path}")
    
    # Plot training curves
    if output_dir:
        plt.figure(figsize=(8, 5))
        plt.semilogy(train_losses, label='Train')
        plt.semilogy(test_losses, label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Branch Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'branch_training_curves.png'), dpi=150)
        plt.close()
    
    print("✓ Branch training complete")
    print("=" * 70)
    
    return {'model': branch, 'train_losses': train_losses, 'test_losses': test_losses}


def train_deeponet_joint(
    config: dict,
    u_fom: np.ndarray,
    svd_data: dict,
    trunk_pretrained_path: str = None,
    branch_pretrained_path: str = None,
    device: torch.device = None,
    output_dir: str = None,
    models_dir: str = None,
    problem_type: str = 'free_evolution',
    f_data: np.ndarray = None,
    v_fom: np.ndarray = None,
) -> dict:
    """Joint training of DeepONet on ground truth data.
    
    Trains the complete DeepONet (trunk + branch) on raw field data,
    optionally initializing from pretrained trunk and branch networks.
    
    Args:
        problem_type: 'free_evolution' or 'constant_force'
        f_data:       Force field data (Nx, Ny, N_samples), required for 'constant_force'
        v_fom:        Velocity field (Nx, Ny, Nt, N_samples), used for 'free_evolution'
    """
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "=" * 70)
    print(f"TRAINING DEEPONET (Joint Trunk + Branch - {problem_type.upper()})")
    print("=" * 70)
    
    # Validate inputs
    if problem_type == 'constant_force' and f_data is None:
        raise ValueError("f_data must be provided for 'constant_force' problem")
    
    dual = (v_fom is not None) and ('basis_v' in svd_data)
    use_force_channel = (problem_type == 'constant_force') and dual

    # Extract data dimensions
    Nx, Ny, Nt, N_samples = u_fom.shape
    n_modes = config['n_modes']
    n_sensors = config['n_sensors']
    
    # Sample training samples if max_deeponet_samples specified
    max_deeponet_samples = config.get('max_deeponet_samples', None)
    if max_deeponet_samples and N_samples > max_deeponet_samples:
        print(f"Sampling {max_deeponet_samples} samples from {N_samples} total samples")
        sample_indices = np.random.choice(N_samples, max_deeponet_samples, replace=False)
        u_fom = u_fom[:, :, :, sample_indices]
        if v_fom is not None:
            v_fom = v_fom[:, :, :, sample_indices]
        if f_data is not None:
            f_data = f_data[:, :, sample_indices]
        N_samples = max_deeponet_samples
    else:
        if max_deeponet_samples:
            print(f"Using all {N_samples} samples (≤ max_deeponet_samples={max_deeponet_samples})")
        else:
            print(f"Using all {N_samples} samples")
    
    # ------------------------------------------------------------------ #
    # Build trunk                                                          #
    # ------------------------------------------------------------------ #
    if dual:
        trunk = DualHeadMLP(3, config['trunk_hidden_dim'], n_modes, config['trunk_n_layers']).to(device)
    else:
        trunk = MLP(3, config['trunk_hidden_dim'], n_modes, config['trunk_n_layers']).to(device)

    if trunk_pretrained_path and os.path.exists(trunk_pretrained_path):
        print(f"Loading pretrained trunk from: {trunk_pretrained_path}")
        trunk_ckpt = torch.load(trunk_pretrained_path, map_location=device, weights_only=False)
        trunk_state = trunk_ckpt.get('model_state_dict', trunk_ckpt)
        trunk.load_state_dict(trunk_state)
        print("  ✓ Trunk loaded")
    else:
        print("  Initializing trunk from scratch")
    
    # ------------------------------------------------------------------ #
    # Build branch                                                         #
    # ------------------------------------------------------------------ #
    input_scale = 0.1 if problem_type == 'constant_force' else 1.0
    if dual:
        input_channels = 3 if use_force_channel else 2
        measurement_dim = input_channels * n_sensors * n_sensors
        branch = DualHeadSensorBranch(
            n_sensors=n_sensors,
            hidden_dim=config['branch_hidden_dim'],
            n_modes=n_modes,
            n_layers=config['branch_n_layers'],
            input_scale=input_scale,
            input_channels=input_channels,
        ).to(device)
    else:
        measurement_dim = n_sensors * n_sensors
        branch = MLP(measurement_dim, config['branch_hidden_dim'], n_modes,
                     config['branch_n_layers'], input_scale=input_scale).to(device)

    if branch_pretrained_path and os.path.exists(branch_pretrained_path):
        print(f"Loading pretrained branch from: {branch_pretrained_path}")
        branch_ckpt = torch.load(branch_pretrained_path, map_location=device, weights_only=False)
        branch_state = branch_ckpt.get('model_state_dict', branch_ckpt)
        branch.load_state_dict(branch_state)
        print("  ✓ Branch loaded")
    else:
        print("  Initializing branch from scratch")
    
    # Create DeepONet
    if dual:
        deeponet = DeepONet(
            trunk=trunk,
            branch_ic=branch,
            problem_type='free_evolution',
            wave_speed=config.get('wave_speed', 1.0),
            damping=config.get('damping', 1.0)
        ).to(device)
    else:
        deeponet = DeepONet(
            trunk=trunk,
            branch_force=branch,
            problem_type='constant_force',
            wave_speed=config.get('wave_speed', 1.0),
            damping=config.get('damping', 1.0)
        ).to(device)
    
    print(f"\nDeepONet architecture:")
    model_type = "DualHeadMLP" if dual else "MLP"
    print(f"  Trunk  ({model_type}): 3 → {config['trunk_hidden_dim']} ({config['trunk_n_layers']} layers) → {n_modes} × {'2 heads' if dual else '1 head'}")
    branch_type = "DualHeadSensorBranch" if dual else model_type
    print(f"  Branch ({branch_type}): {measurement_dim} → {config['branch_hidden_dim']} ({config['branch_n_layers']} layers) → {n_modes} × {'2 heads' if dual else '1 head'}")
    print(f"  Problem type: {problem_type}")
    
    # ------------------------------------------------------------------ #
    # Prepare training data                                                #
    # ------------------------------------------------------------------ #
    print("\nPreparing training data...")
    
    sensor_x_indices = np.linspace(0, Nx - 1, n_sensors, dtype=int)
    sensor_y_indices = np.linspace(0, Ny - 1, n_sensors, dtype=int)
    
    if dual:
        u_sensors = []
        for s in range(N_samples):
            u_ic = u_fom[:, :, 0, s]
            u_sensors.append([u_ic[si, sj] for si in sensor_x_indices for sj in sensor_y_indices])
        u_sensors = np.array(u_sensors)
        u_s_min, u_s_max = u_sensors.min(), u_sensors.max()
        u_s_norm = 2 * (u_sensors - u_s_min) / (u_s_max - u_s_min + 1e-10) - 1

        v_sensors = []
        for s in range(N_samples):
            v_ic = v_fom[:, :, 0, s]
            v_sensors.append([v_ic[si, sj] for si in sensor_x_indices for sj in sensor_y_indices])
        v_sensors = np.array(v_sensors)
        v_s_min, v_s_max = v_sensors.min(), v_sensors.max()
        v_s_norm = 2 * (v_sensors - v_s_min) / (v_s_max - v_s_min + 1e-10) - 1

        u_grid = u_s_norm.reshape(N_samples, n_sensors, n_sensors)
        v_grid = v_s_norm.reshape(N_samples, n_sensors, n_sensors)
        if use_force_channel:
            force_sensors = []
            for s in range(N_samples):
                f_field = f_data[:, :, s]
                force_sensors.append([f_field[si, sj] for si in sensor_x_indices for sj in sensor_y_indices])
            force_sensors = np.array(force_sensors)
            f_s_min, f_s_max = force_sensors.min(), force_sensors.max()
            f_s_norm = 2 * (force_sensors - f_s_min) / (f_s_max - f_s_min + 1e-10) - 1
            f_grid = f_s_norm.reshape(N_samples, n_sensors, n_sensors)
            measurements = np.stack([u_grid, v_grid, f_grid], axis=1)
        else:
            f_s_min = f_s_max = 0.0
            measurements = np.stack([u_grid, v_grid], axis=1)

        measurements_min = measurements.min()
        measurements_max = measurements.max()
    else:  # constant_force
        meas_list = []
        for s in range(N_samples):
            f_field = f_data[:, :, s]
            meas_list.append([f_field[si, sj] for si in sensor_x_indices for sj in sensor_y_indices])
        measurements = np.array(meas_list)
        measurements_min = measurements.min()
        measurements_max = measurements.max()
        measurements_range = measurements_max - measurements_min
        measurements = 2 * (measurements - measurements_min) / (measurements_range + 1e-10) - 1
    
    # Generate coordinate grid
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    t_vals = np.linspace(0, 1, Nt)
    X, Y, T = np.meshgrid(x, y, t_vals, indexing='ij')
    coords_all = np.stack([X.flatten('F'), Y.flatten('F'), T.flatten('F')], axis=1)
    
    # Normalize u field
    u_min = u_fom.min()
    u_max = u_fom.max()
    u_fom_norm = 2 * (u_fom - u_min) / (u_max - u_min + 1e-10) - 1
    u_targets_all = u_fom_norm.reshape(-1, N_samples, order='F')  # (Nx*Ny*Nt, N_samples)

    # Normalize v field
    if dual:
        v_min = v_fom.min()
        v_max = v_fom.max()
        v_fom_norm = 2 * (v_fom - v_min) / (v_max - v_min + 1e-10) - 1
        v_targets_all = v_fom_norm.reshape(-1, N_samples, order='F')
    
    # Sample points per sample
    n_points_per_sample = config.get('n_points_per_sample', 1000)
    
    coords_list, meas_list_out, u_tgt_list = [], [], []
    v_tgt_list = [] if dual else None

    for s in range(N_samples):
        n_pts = min(n_points_per_sample, coords_all.shape[0])
        pt_indices = np.random.choice(coords_all.shape[0], n_pts, replace=False)
        
        coords_list.append(coords_all[pt_indices])
        if dual:
            meas_list_out.append(np.repeat(measurements[s][None, ...], n_pts, axis=0))
        else:
            meas_list_out.append(np.tile(measurements[s], (n_pts, 1)))
        u_tgt_list.append(u_targets_all[pt_indices, s])
        if dual:
            v_tgt_list.append(v_targets_all[pt_indices, s])
    
    coords_train_np = np.vstack(coords_list)
    meas_train_np   = np.concatenate(meas_list_out, axis=0)
    u_tgt_np        = np.hstack(u_tgt_list)
    
    print(f"  Training points: {len(u_tgt_np)}")
    if dual:
        print(f"  Measurement shape: {measurements.shape[1:]} (channels, sensors, sensors)")
    else:
        print(f"  Measurement dim: {measurements.shape[1]}")
    print(f"  u target range: [{u_fom.min():.4f}, {u_fom.max():.4f}]")
    if dual:
        v_tgt_np = np.hstack(v_tgt_list)
        print(f"  v target range: [{v_fom.min():.4f}, {v_fom.max():.4f}]")
    
    # Convert to tensors
    coords_t  = torch.from_numpy(coords_train_np).float().to(device)
    meas_t    = torch.from_numpy(meas_train_np).float().to(device)
    u_tgt_t   = torch.from_numpy(u_tgt_np).float().unsqueeze(1).to(device)
    if dual:
        v_tgt_t = torch.from_numpy(v_tgt_np).float().unsqueeze(1).to(device)
    
    # Train/test split
    n_total = len(u_tgt_np)
    n_train = int(n_total * config.get('train_test_split', 0.8))
    perm    = torch.randperm(n_total)
    tr_idx, te_idx = perm[:n_train], perm[n_train:]
    
    if dual:
        train_dataset = TensorDataset(meas_t[tr_idx], coords_t[tr_idx],
                                      u_tgt_t[tr_idx], v_tgt_t[tr_idx])
        test_meas    = meas_t[te_idx]
        test_coords  = coords_t[te_idx]
        test_u_tgt   = u_tgt_t[te_idx]
        test_v_tgt   = v_tgt_t[te_idx]
    else:
        train_dataset = TensorDataset(meas_t[tr_idx], coords_t[tr_idx], u_tgt_t[tr_idx])
        test_meas   = meas_t[te_idx]
        test_coords = coords_t[te_idx]
        test_u_tgt  = u_tgt_t[te_idx]
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    optimizer = optim.Adam(deeponet.parameters(), lr=config.get('learning_rate', 1e-3))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=False)
    criterion = nn.MSELoss()
    
    train_losses, test_losses = [], []
    best_test_loss = float('inf')
    best_state = None
    
    print(f"\nTraining: {len(tr_idx)} points, Testing: {len(te_idx)} points")
    print("Starting training...\n")
    
    for epoch in range(config['deeponet_n_epochs']):
        deeponet.train()
        train_loss_epoch = 0.0
        
        if dual:
            for meas_b, coords_b, u_b, v_b in train_loader:
                optimizer.zero_grad()
                u_pred, v_pred = deeponet(meas_b, coords_b)
                loss = criterion(u_pred.unsqueeze(1), u_b) + criterion(v_pred.unsqueeze(1), v_b)
                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item()
        else:
            for meas_b, coords_b, u_b in train_loader:
                optimizer.zero_grad()
                pred = deeponet(meas_b, coords_b).unsqueeze(1)
                loss = criterion(pred, u_b)
                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item()
        
        train_loss_epoch /= len(train_loader)
        train_losses.append(train_loss_epoch)
        
        deeponet.eval()
        with torch.no_grad():
            if dual:
                u_pred_te, v_pred_te = deeponet(test_meas, test_coords)
                test_loss = (criterion(u_pred_te.unsqueeze(1), test_u_tgt) +
                             criterion(v_pred_te.unsqueeze(1), test_v_tgt)).item()
            else:
                pred_te = deeponet(test_meas, test_coords).unsqueeze(1)
                test_loss = criterion(pred_te, test_u_tgt).item()
        test_losses.append(test_loss)
        scheduler.step(test_loss)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = {k: v.cpu().clone() for k, v in deeponet.state_dict().items()}
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d}/{config['deeponet_n_epochs']} | Train: {train_loss_epoch:.6e} | Test: {test_loss:.6e}")
    
    # Load best model
    deeponet.load_state_dict(best_state)
    deeponet.eval()
    
    # Save checkpoint
    os.makedirs(output_dir, exist_ok=True)
    problem_suffix = problem_type
    ckpt_path = os.path.join(output_dir, f'deeponet_{problem_suffix}.pth')
    torch.save({
        'model_state_dict': best_state,
        'config': config,
        'input_scale': input_scale,
        'dual': dual,
        'n_input_channels': int(input_channels) if dual else 1,
        'normalization': {
            'u_min': u_min, 'u_max': u_max,
            'v_min': float(v_min) if dual else None,
            'v_max': float(v_max) if dual else None,
            'measurements_min': measurements_min,
            'measurements_max': measurements_max,
            'force_measurements_min': float(f_s_min) if (dual and use_force_channel) else None,
            'force_measurements_max': float(f_s_max) if (dual and use_force_channel) else None,
        }
    }, ckpt_path)
    print(f"\n✓ Saved: {ckpt_path}")
    
    # Plot training curves
    if output_dir:
        plt.figure(figsize=(8, 5))
        plt.semilogy(train_losses, label='Train')
        plt.semilogy(test_losses, label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title(f'DeepONet Joint Training Loss ({problem_type})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'deeponet_training_curves.png'), dpi=150)
        plt.close()
    
    print("✓ DeepONet joint training complete")
    print("=" * 70)
    
    return {'model': deeponet, 'train_losses': train_losses, 'test_losses': test_losses}
