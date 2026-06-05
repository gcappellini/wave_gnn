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


def _scale_tensor_to_ominus1_1(values: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:
    """Scale a tensor into [-1, 1] using externally supplied bounds."""

    return 2.0 * (values - min_value) / (max_value - min_value + 1e-10) - 1.0


def _sample_force_at_coords(force_grid: torch.Tensor, coords_xy: torch.Tensor) -> torch.Tensor:
    """Sample a static force grid at (x, y) using bilinear interpolation on [0, 1]^2."""

    if force_grid.ndim != 2:
        raise ValueError(f"force_grid must have shape (Nx, Ny), got {tuple(force_grid.shape)}")
    if coords_xy.ndim != 2 or coords_xy.shape[1] < 2:
        raise ValueError(f"coords_xy must have shape (N, >=2), got {tuple(coords_xy.shape)}")

    nx, ny = force_grid.shape
    x = coords_xy[:, 0].detach().clamp(0.0, 1.0)
    y = coords_xy[:, 1].detach().clamp(0.0, 1.0)

    gx = x * (nx - 1)
    gy = y * (ny - 1)

    x0 = torch.floor(gx).long()
    y0 = torch.floor(gy).long()
    x1 = torch.clamp(x0 + 1, max=nx - 1)
    y1 = torch.clamp(y0 + 1, max=ny - 1)

    wx = gx - x0.float()
    wy = gy - y0.float()

    f00 = force_grid[x0, y0]
    f10 = force_grid[x1, y0]
    f01 = force_grid[x0, y1]
    f11 = force_grid[x1, y1]

    return (
        (1.0 - wx) * (1.0 - wy) * f00
        + wx * (1.0 - wy) * f10
        + (1.0 - wx) * wy * f01
        + wx * wy * f11
    )


def _apply_deeponet_branch_input_noise(
    measurements: torch.Tensor,
    noise_enabled: bool,
    noise_std: float,
    apply_noise_to_force: bool,
    clip_min,
    clip_max,
) -> torch.Tensor:
    """Apply optional Gaussian noise to normalized branch inputs."""

    if not noise_enabled or noise_std <= 0.0:
        return measurements

    noise = torch.randn_like(measurements) * noise_std
    if measurements.ndim == 4 and measurements.shape[1] > 2 and not apply_noise_to_force:
        noise[:, 2:, :, :] = 0.0
    elif measurements.ndim == 2 and not apply_noise_to_force:
        noise.zero_()

    noisy_measurements = measurements + noise
    if clip_min is not None or clip_max is not None:
        clamp_min = clip_min if clip_min is not None else float('-inf')
        clamp_max = clip_max if clip_max is not None else float('inf')
        noisy_measurements = torch.clamp(noisy_measurements, min=clamp_min, max=clamp_max)

    return noisy_measurements


def _resolve_sample_split_ids(config: dict, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Resolve train/test sample ids from split file or fallback random split."""

    split_override = config.get('sample_split_indices_override', None)
    if split_override is not None:
        train_sample_ids = np.array(split_override.get('train', []), dtype=np.int64).reshape(-1)
        test_sample_ids = np.array(split_override.get('test', []), dtype=np.int64).reshape(-1)
        if train_sample_ids.size == 0 or test_sample_ids.size == 0:
            raise ValueError("sample_split_indices_override must include non-empty train/test arrays")
        if train_sample_ids.min() < 0 or test_sample_ids.min() < 0:
            raise ValueError("sample_split_indices_override cannot include negative indices")
        if train_sample_ids.max() >= n_samples or test_sample_ids.max() >= n_samples:
            raise ValueError(
                f"sample_split_indices_override out of bounds for N_samples={n_samples}"
            )
        return train_sample_ids, test_sample_ids

    split_file = config.get('sample_split_file', None)
    if split_file:
        split_path = os.path.expanduser(str(split_file))
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"sample_split_file not found: {split_path}")

        split_data = np.load(split_path, allow_pickle=True)
        train_key = 'train_indices' if 'train_indices' in split_data else 'train_idx'
        test_key = 'test_indices' if 'test_indices' in split_data else 'test_idx'
        if train_key not in split_data or test_key not in split_data:
            raise ValueError(
                f"Invalid split file {split_path}: expected train_indices/test_indices"
            )

        train_sample_ids = np.array(split_data[train_key], dtype=np.int64).reshape(-1)
        test_sample_ids = np.array(split_data[test_key], dtype=np.int64).reshape(-1)
        if train_sample_ids.size == 0 or test_sample_ids.size == 0:
            raise ValueError(f"Invalid split file {split_path}: empty train or test indices")
        if train_sample_ids.min() < 0 or test_sample_ids.min() < 0:
            raise ValueError(f"Invalid split file {split_path}: negative sample indices")
        if train_sample_ids.max() >= n_samples or test_sample_ids.max() >= n_samples:
            raise ValueError(
                f"Invalid split file {split_path}: indices out of bounds for N_samples={n_samples}"
            )

        print(f"  Using fixed sample split from file: {split_path}")
        print(f"  Samples train/test: {len(train_sample_ids)}/{len(test_sample_ids)}")
        return train_sample_ids, test_sample_ids

    train_split = float(config.get('train_test_split', 0.8))
    if n_samples > 1:
        n_train_samples = int(n_samples * train_split)
        n_train_samples = min(max(n_train_samples, 1), n_samples - 1)
    else:
        n_train_samples = 1

    sample_perm = np.random.permutation(n_samples)
    train_sample_ids = sample_perm[:n_train_samples]
    test_sample_ids = sample_perm[n_train_samples:] if n_samples > 1 else sample_perm[:1]
    if len(test_sample_ids) == 0:
        test_sample_ids = train_sample_ids[:1]

    return train_sample_ids, test_sample_ids


def _build_dual_rollout_measurement_tensor(
    u_field: torch.Tensor,
    v_field: torch.Tensor,
    sample_idx: int,
    sensor_x_indices: torch.Tensor,
    sensor_y_indices: torch.Tensor,
    raw_u_min: float,
    raw_u_max: float,
    raw_v_min: float,
    raw_v_max: float,
    input_channels: int,
    force_grids: torch.Tensor = None,
) -> torch.Tensor:
    """Build a normalized dual-head branch input from the current rollout state."""

    u_grid = u_field.index_select(0, sensor_x_indices).index_select(1, sensor_y_indices)
    v_grid = v_field.index_select(0, sensor_x_indices).index_select(1, sensor_y_indices)
    u_grid_norm = _scale_tensor_to_ominus1_1(u_grid, raw_u_min, raw_u_max)
    v_grid_norm = _scale_tensor_to_ominus1_1(v_grid, raw_v_min, raw_v_max)

    if input_channels == 3:
        if force_grids is None:
            raise ValueError("3-channel rollout training requires normalized force sensor grids")
        meas = torch.stack([u_grid_norm, v_grid_norm, force_grids[sample_idx]], dim=0)
    else:
        meas = torch.stack([u_grid_norm, v_grid_norm], dim=0)

    return meas.unsqueeze(0)


def _predict_dual_rollout_field(
    deeponet: DeepONet,
    measurement_tensor: torch.Tensor,
    coords_tensor: torch.Tensor,
    nx: int,
    ny: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict a full u/v field for one rollout step using chunked inference."""

    u_chunks = []
    v_chunks = []
    n_points = coords_tensor.shape[0]

    for start_idx in range(0, n_points, chunk_size):
        end_idx = min(start_idx + chunk_size, n_points)
        coords_chunk = coords_tensor[start_idx:end_idx]
        meas_chunk = measurement_tensor.expand(end_idx - start_idx, -1, -1, -1)
        u_chunk, v_chunk = deeponet(meas_chunk, coords_chunk)
        u_chunks.append(u_chunk)
        v_chunks.append(v_chunk)

    u_pred = torch.cat(u_chunks, dim=0).reshape(ny, nx).transpose(0, 1).contiguous()
    v_pred = torch.cat(v_chunks, dim=0).reshape(ny, nx).transpose(0, 1).contiguous()
    return u_pred, v_pred


def _resolve_rollout_schedule(t_vals: np.ndarray, rollout_config: dict) -> tuple[int, int, float, int, list[float]]:
    """Resolve rollout horizon and dt on the native time grid."""

    horizon = int(rollout_config.get('horizon', 1) or 1)
    if horizon <= 0:
        raise ValueError(f"deeponet_rollout.horizon must be positive, got {horizon}")

    step_stride = int(rollout_config.get('step_stride', 1) or 1)
    if step_stride <= 0:
        raise ValueError(f"deeponet_rollout.step_stride must be positive, got {step_stride}")

    dt_override = rollout_config.get('dt', None)
    if dt_override is not None:
        dt_override = float(dt_override)
        step_deltas = t_vals - float(t_vals[0])
        candidate_idx = int(np.argmin(np.abs(step_deltas - dt_override)))
        candidate_dt = float(step_deltas[candidate_idx])
        if candidate_idx <= 0 or not np.isclose(candidate_dt, dt_override, atol=1e-8, rtol=1e-6):
            raise ValueError(
                "deeponet_rollout.dt must match an integer multiple of the stored time grid. "
                f"Requested {dt_override}, nearest available is {candidate_dt}."
            )
        step_stride = candidate_idx

    if step_stride >= len(t_vals):
        raise ValueError(
            f"deeponet_rollout.step_stride={step_stride} is too large for Nt={len(t_vals)}"
        )

    rollout_dt = float(t_vals[step_stride] - t_vals[0])
    max_start_idx = len(t_vals) - 1 - horizon * step_stride
    if max_start_idx < 0:
        raise ValueError(
            f"Rollout horizon={horizon} with step_stride={step_stride} exceeds Nt={len(t_vals)}"
        )

    step_weights = rollout_config.get('step_weights', None)
    if step_weights is None:
        rollout_weights = [1.0] * horizon
    else:
        rollout_weights = [float(weight) for weight in step_weights]
        if len(rollout_weights) != horizon:
            raise ValueError(
                f"deeponet_rollout.step_weights must have length {horizon}, got {len(rollout_weights)}"
            )

    return horizon, step_stride, rollout_dt, max_start_idx, rollout_weights


def _compute_dual_rollout_batch_loss(
    deeponet: DeepONet,
    sample_indices,
    start_indices,
    u_fom: np.ndarray,
    v_fom: np.ndarray,
    sensor_x_indices: torch.Tensor,
    sensor_y_indices: torch.Tensor,
    raw_u_min: float,
    raw_u_max: float,
    raw_v_min: float,
    raw_v_max: float,
    input_channels: int,
    coords_rollout: torch.Tensor,
    rollout_weights: list[float],
    step_stride: int,
    criterion,
    device: torch.device,
    nx: int,
    ny: int,
    chunk_size: int,
    noise_enabled: bool,
    noise_std: float,
    apply_noise_to_force: bool,
    clip_min,
    clip_max,
    force_grids: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute multi-step rollout loss for a batch of sample trajectories."""

    batch_loss_u = torch.zeros((), device=device)
    batch_loss_v = torch.zeros((), device=device)
    total_weight = sum(rollout_weights)

    for sample_idx, start_idx in zip(sample_indices, start_indices):
        current_u = torch.from_numpy(u_fom[:, :, start_idx, sample_idx]).float().to(device)
        current_v = torch.from_numpy(v_fom[:, :, start_idx, sample_idx]).float().to(device)

        for step_number, step_weight in enumerate(rollout_weights, start=1):
            target_idx = start_idx + step_number * step_stride

            meas_tensor = _build_dual_rollout_measurement_tensor(
                u_field=current_u,
                v_field=current_v,
                sample_idx=sample_idx,
                sensor_x_indices=sensor_x_indices,
                sensor_y_indices=sensor_y_indices,
                raw_u_min=raw_u_min,
                raw_u_max=raw_u_max,
                raw_v_min=raw_v_min,
                raw_v_max=raw_v_max,
                input_channels=input_channels,
                force_grids=force_grids,
            )
            meas_tensor = _apply_deeponet_branch_input_noise(
                meas_tensor,
                noise_enabled=noise_enabled,
                noise_std=noise_std,
                apply_noise_to_force=apply_noise_to_force,
                clip_min=clip_min,
                clip_max=clip_max,
            )

            u_pred, v_pred = _predict_dual_rollout_field(
                deeponet=deeponet,
                measurement_tensor=meas_tensor,
                coords_tensor=coords_rollout,
                nx=nx,
                ny=ny,
                chunk_size=chunk_size,
            )
            u_gt = torch.from_numpy(u_fom[:, :, target_idx, sample_idx]).float().to(device)
            v_gt = torch.from_numpy(v_fom[:, :, target_idx, sample_idx]).float().to(device)

            batch_loss_u = batch_loss_u + step_weight * criterion(u_pred, u_gt)
            batch_loss_v = batch_loss_v + step_weight * criterion(v_pred, v_gt)

            current_u = u_pred
            current_v = v_pred

    normalizer = max(1, len(sample_indices)) * total_weight
    return batch_loss_u / normalizer, batch_loss_v / normalizer


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
        
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < 5e-6:
            print(f"\nEarly stopping at epoch {epoch+1}: LR={current_lr:.3e} < 5e-6")
            break
    
    trunk.load_state_dict(best_state)
    trunk.eval()
    
    # Save checkpoint to output checkpoints directory.
    ckpt_dir = os.path.join(output_dir, 'checkpoints') if output_dir is not None else models_dir
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

    use_force_channel = (problem_type in {'constant_force', 'merged'})
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
        
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < 5e-6:
            print(f"\nEarly stopping at epoch {epoch+1}: LR={current_lr:.3e} < 5e-6")
            break
    
    branch.load_state_dict(best_state)
    branch.eval()
    
    # Save checkpoint to output checkpoints directory.
    ckpt_dir = os.path.join(output_dir, 'checkpoints') if output_dir is not None else models_dir
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
    deeponet_pretrained_path: str = None,
    device: torch.device = None,
    output_dir: str = None,
    models_dir: str = None,
    problem_type: str = 'free_evolution',
    f_data: np.ndarray = None,
    v_fom: np.ndarray = None,
) -> dict:
    """Joint training of DeepONet on ground truth data.
    
    Trains the complete DeepONet (trunk + branch) on raw field data,
    optionally initializing from pretrained trunk/branch or an existing
    full DeepONet checkpoint.
    
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
    if problem_type in {'constant_force', 'merged'} and f_data is None:
        raise ValueError("f_data must be provided for force-driven problems ('constant_force'/'merged')")
    
    dual = (v_fom is not None) and ('basis_v' in svd_data)
    use_force_channel = (problem_type in {'constant_force', 'merged'}) and dual

    # Extract data dimensions
    Nx, Ny, Nt, N_samples = u_fom.shape
    n_modes = config['n_modes']
    n_sensors = config['n_sensors']
    
    # Sample training samples if max_deeponet_samples specified
    max_deeponet_samples = config.get('max_deeponet_samples', None)
    if max_deeponet_samples and N_samples > max_deeponet_samples:
        print(f"Sampling {max_deeponet_samples} samples from {N_samples} total samples")

        split_file_for_sampling = config.get('sample_split_file', None)
        if split_file_for_sampling:
            # Preserve train/test integrity under subsampling by sampling within each split.
            train_ids_full, test_ids_full = _resolve_sample_split_ids(config, N_samples)
            train_ratio = len(train_ids_full) / float(N_samples)

            n_train_target = int(round(max_deeponet_samples * train_ratio))
            n_train_target = min(max(1, n_train_target), len(train_ids_full))
            n_test_target = max_deeponet_samples - n_train_target
            n_test_target = min(max(1, n_test_target), len(test_ids_full))

            # Adjust if rounding/clamping changed total.
            total_target = n_train_target + n_test_target
            if total_target < max_deeponet_samples:
                remaining = max_deeponet_samples - total_target
                train_room = len(train_ids_full) - n_train_target
                add_train = min(train_room, remaining)
                n_train_target += add_train
                remaining -= add_train
                if remaining > 0:
                    test_room = len(test_ids_full) - n_test_target
                    n_test_target += min(test_room, remaining)
            elif total_target > max_deeponet_samples:
                overflow = total_target - max_deeponet_samples
                reduce_from_test = min(max(0, n_test_target - 1), overflow)
                n_test_target -= reduce_from_test
                overflow -= reduce_from_test
                if overflow > 0:
                    n_train_target -= min(max(0, n_train_target - 1), overflow)

            train_sub = np.random.choice(train_ids_full, n_train_target, replace=False)
            test_sub = np.random.choice(test_ids_full, n_test_target, replace=False)
            sample_indices = np.sort(np.concatenate([train_sub, test_sub]).astype(np.int64))

            global_to_local = {int(g_idx): int(l_idx) for l_idx, g_idx in enumerate(sample_indices.tolist())}
            train_local = np.array([global_to_local[int(g_idx)] for g_idx in train_sub], dtype=np.int64)
            test_local = np.array([global_to_local[int(g_idx)] for g_idx in test_sub], dtype=np.int64)
            config['sample_split_indices_override'] = {
                'train': train_local.tolist(),
                'test': test_local.tolist(),
            }
            print(
                f"  Split-aware subsampling applied: train/test={len(train_local)}/{len(test_local)} "
                f"(from split file)"
            )
        else:
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
    input_scale = 1.0
    branch_input_norm = None
    if branch_pretrained_path and os.path.exists(branch_pretrained_path):
        try:
            branch_ckpt_meta = torch.load(branch_pretrained_path, map_location='cpu', weights_only=False)
            input_scale = float(branch_ckpt_meta.get('input_scale', 1.0) or 1.0)
            branch_input_norm = branch_ckpt_meta.get('input_normalization', None)
        except Exception as exc:
            print(f"Warning: failed to read branch checkpoint metadata for normalization/scales: {exc}")
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

    if deeponet_pretrained_path and os.path.exists(deeponet_pretrained_path):
        print(f"Loading pretrained DeepONet from: {deeponet_pretrained_path}")
        deeponet_ckpt = torch.load(deeponet_pretrained_path, map_location=device, weights_only=False)
        deeponet_state = deeponet_ckpt.get('model_state_dict', deeponet_ckpt)
        deeponet.load_state_dict(deeponet_state)
        print("  ✓ DeepONet loaded")
    elif deeponet_pretrained_path:
        print(f"  Warning: requested DeepONet checkpoint not found: {deeponet_pretrained_path}")
        print("  Continuing with trunk/branch initialization.")
    
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

        v_sensors = []
        for s in range(N_samples):
            v_ic = v_fom[:, :, 0, s]
            v_sensors.append([v_ic[si, sj] for si in sensor_x_indices for sj in sensor_y_indices])
        v_sensors = np.array(v_sensors)

        # Keep branch input normalization consistent with branch pretraining when possible.
        if branch_input_norm is not None:
            raw_u_min = float(branch_input_norm.get('raw_u_min', u_sensors.min()))
            raw_u_max = float(branch_input_norm.get('raw_u_max', u_sensors.max()))
            raw_v_min = float(branch_input_norm.get('raw_v_min', v_sensors.min()))
            raw_v_max = float(branch_input_norm.get('raw_v_max', v_sensors.max()))
            if use_force_channel:
                raw_f_min = float(branch_input_norm.get('raw_f_min', float(np.min(f_data))))
                raw_f_max = float(branch_input_norm.get('raw_f_max', float(np.max(f_data))))
            else:
                raw_f_min, raw_f_max = 0.0, 1.0
            print("  Using branch checkpoint input_normalization for DeepONet branch inputs")
        else:
            raw_u_min, raw_u_max = float(u_sensors.min()), float(u_sensors.max())
            raw_v_min, raw_v_max = float(v_sensors.min()), float(v_sensors.max())
            if use_force_channel:
                raw_f_min, raw_f_max = float(np.min(f_data)), float(np.max(f_data))
            else:
                raw_f_min, raw_f_max = 0.0, 1.0
            print("  Warning: branch checkpoint input_normalization missing; using current-run sensor min/max")

        u_s_norm = _scale_to_ominus1_1(u_sensors, raw_u_min, raw_u_max)
        v_s_norm = _scale_to_ominus1_1(v_sensors, raw_v_min, raw_v_max)

        u_grid = u_s_norm.reshape(N_samples, n_sensors, n_sensors)
        v_grid = v_s_norm.reshape(N_samples, n_sensors, n_sensors)
        if use_force_channel:
            force_sensors = []
            for s in range(N_samples):
                f_field = f_data[:, :, s]
                force_sensors.append([f_field[si, sj] for si in sensor_x_indices for sj in sensor_y_indices])
            force_sensors = np.array(force_sensors)
            f_s_min, f_s_max = float(force_sensors.min()), float(force_sensors.max())
            f_s_norm = _scale_to_ominus1_1(force_sensors, raw_f_min, raw_f_max)
            f_grid = f_s_norm.reshape(N_samples, n_sensors, n_sensors)
            measurements = np.stack([u_grid, v_grid, f_grid], axis=1)
        else:
            f_s_min = f_s_max = 0.0
            measurements = np.stack([u_grid, v_grid], axis=1)

        print(
            f"  Branch input normalization ranges used in DeepONet: "
            f"u=[{raw_u_min:.6e}, {raw_u_max:.6e}], v=[{raw_v_min:.6e}, {raw_v_max:.6e}]"
            + (f", f=[{raw_f_min:.6e}, {raw_f_max:.6e}]" if use_force_channel else "")
        )
        measurements_min = measurements.min()
        measurements_max = measurements.max()
    else:  # constant_force
        meas_list = []
        for s in range(N_samples):
            f_field = f_data[:, :, s]
            meas_list.append([f_field[si, sj] for si in sensor_x_indices for sj in sensor_y_indices])
        measurements = np.array(meas_list)
        if branch_input_norm is not None:
            raw_f_min = float(branch_input_norm.get('raw_f_min', measurements.min()))
            raw_f_max = float(branch_input_norm.get('raw_f_max', measurements.max()))
            print("  Using branch checkpoint input_normalization for DeepONet force inputs")
        else:
            raw_f_min = float(measurements.min())
            raw_f_max = float(measurements.max())
            print("  Warning: branch checkpoint input_normalization missing; using current-run force min/max")

        measurements = _scale_to_ominus1_1(measurements, raw_f_min, raw_f_max)
        measurements_min = measurements.min()
        measurements_max = measurements.max()
        f_s_min, f_s_max = float(np.min(meas_list)), float(np.max(meas_list))
    
    rollout_config = config.get('deeponet_rollout', {}) or {}
    rollout_enabled = bool(rollout_config.get('enabled', False))
    if rollout_enabled and not dual:
        raise ValueError("deeponet_rollout is only supported for dual-head DeepONet training")

    # Generate coordinate grid
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    t_vals = np.linspace(0, 1, Nt)
    X, Y, T = np.meshgrid(x, y, t_vals, indexing='ij')
    coords_all = np.stack([X.flatten('F'), Y.flatten('F'), T.flatten('F')], axis=1)
    
    # DeepONet trains on raw physical fields from simulation data.
    u_min = u_fom.min()
    u_max = u_fom.max()
    u_targets_all = u_fom.reshape(-1, N_samples, order='F')  # (Nx*Ny*Nt, N_samples)

    # Normalize v field
    if dual:
        v_min = v_fom.min()
        v_max = v_fom.max()
        v_targets_all = v_fom.reshape(-1, N_samples, order='F')
    
    train_sample_ids, test_sample_ids = _resolve_sample_split_ids(config, N_samples)
    force_fields_t = torch.from_numpy(f_data).float().to(device) if use_force_channel else None
    if dual and rollout_enabled:
        horizon, step_stride, rollout_dt, max_start_idx, rollout_weights = _resolve_rollout_schedule(
            t_vals,
            rollout_config,
        )

        rollout_batch_size = int(rollout_config.get('batch_size', min(4, max(1, len(train_sample_ids)))) or 1)
        rollout_batch_size = max(1, rollout_batch_size)
        rollout_chunk_size = int(rollout_config.get('chunk_size', Nx * Ny) or (Nx * Ny))
        rollout_chunk_size = max(1, rollout_chunk_size)
        eval_max_samples = rollout_config.get('eval_max_samples', None)
        eval_start_idx = int(rollout_config.get('eval_start_index', 0) or 0)
        if eval_start_idx < 0 or eval_start_idx > max_start_idx:
            raise ValueError(
                f"deeponet_rollout.eval_start_index must be in [0, {max_start_idx}], got {eval_start_idx}"
            )

        coords_rollout_np = np.stack([
            X[:, :, 0].flatten('F'),
            Y[:, :, 0].flatten('F'),
            np.full((Nx * Ny,), rollout_dt, dtype=np.float32),
        ], axis=1)
        coords_rollout = torch.from_numpy(coords_rollout_np).float().to(device)
        sensor_x_t = torch.as_tensor(sensor_x_indices, dtype=torch.long, device=device)
        sensor_y_t = torch.as_tensor(sensor_y_indices, dtype=torch.long, device=device)
        force_grids_t = torch.from_numpy(f_grid).float().to(device) if use_force_channel else None

        train_dataset = TensorDataset(torch.from_numpy(train_sample_ids).long())
        train_loader = DataLoader(train_dataset, batch_size=rollout_batch_size, shuffle=True)

        print(f"  Rollout training enabled: horizon={horizon}, step_stride={step_stride}, dt={rollout_dt:.6e}")
        print(f"  Rollout train/test samples: {len(train_sample_ids)}/{len(test_sample_ids)}")
        print(f"  Rollout batch size: {rollout_batch_size}, chunk size: {rollout_chunk_size}")
        print(f"  Measurement shape: {measurements.shape[1:]} (channels, sensors, sensors)")
        print(f"  u target range: [{u_fom.min():.4f}, {u_fom.max():.4f}]")
        print(f"  v target range: [{v_fom.min():.4f}, {v_fom.max():.4f}]")
    else:
        # Sample points per sample for one-step supervision.
        n_points_per_sample = config.get('n_points_per_sample', config.get('points_per_sample', 1000))

        def _build_point_supervision(sample_ids):
            coords_list, meas_list_out, u_tgt_list = [], [], []
            v_tgt_list_local = [] if dual else None
            f_tgt_list_local = [] if (dual and use_force_channel) else None

            if dual and use_force_channel:
                # Force is static in time for each sample: index it on (x, y) only.
                n_xy_local = Nx * Ny
                f_targets_all_local = f_data.reshape(n_xy_local, N_samples, order='F')

            for s in sample_ids:
                n_pts = min(n_points_per_sample, coords_all.shape[0])
                pt_indices = np.random.choice(coords_all.shape[0], n_pts, replace=False)

                coords_list.append(coords_all[pt_indices])
                if dual:
                    meas_list_out.append(np.repeat(measurements[s][None, ...], n_pts, axis=0))
                else:
                    meas_list_out.append(np.tile(measurements[s], (n_pts, 1)))
                u_tgt_list.append(u_targets_all[pt_indices, s])
                if dual:
                    v_tgt_list_local.append(v_targets_all[pt_indices, s])
                    if use_force_channel:
                        spatial_indices = pt_indices % n_xy_local
                        f_tgt_list_local.append(f_targets_all_local[spatial_indices, s])

            coords_np = np.vstack(coords_list)
            meas_np = np.concatenate(meas_list_out, axis=0)
            u_np = np.hstack(u_tgt_list)

            out = {
                'coords': coords_np,
                'meas': meas_np,
                'u': u_np,
            }
            if dual:
                out['v'] = np.hstack(v_tgt_list_local)
                if use_force_channel:
                    out['f'] = np.hstack(f_tgt_list_local)
            return out

        train_data = _build_point_supervision(train_sample_ids)
        test_data = _build_point_supervision(test_sample_ids)

        coords_train_np = train_data['coords']
        meas_train_np = train_data['meas']
        u_tgt_np = train_data['u']

        print(f"  Training points: {len(u_tgt_np)}")
        if dual:
            print(f"  Measurement shape: {measurements.shape[1:]} (channels, sensors, sensors)")
        else:
            print(f"  Measurement dim: {measurements.shape[1]}")
        print(f"  u target range: [{u_fom.min():.4f}, {u_fom.max():.4f}]")
        if dual:
            v_tgt_np = train_data['v']
            print(f"  v target range: [{v_fom.min():.4f}, {v_fom.max():.4f}]")

        # Convert to tensors
        coords_t = torch.from_numpy(coords_train_np).float().to(device)
        meas_t = torch.from_numpy(meas_train_np).float().to(device)
        u_tgt_t = torch.from_numpy(u_tgt_np).float().unsqueeze(1).to(device)
        test_coords = torch.from_numpy(test_data['coords']).float().to(device)
        test_meas = torch.from_numpy(test_data['meas']).float().to(device)
        test_u_tgt = torch.from_numpy(test_data['u']).float().unsqueeze(1).to(device)
        if dual:
            v_tgt_np = train_data['v']
            v_tgt_t = torch.from_numpy(v_tgt_np).float().unsqueeze(1).to(device)
            test_v_tgt = torch.from_numpy(test_data['v']).float().unsqueeze(1).to(device)
            if use_force_channel:
                f_tgt_np = train_data['f']
                f_tgt_t = torch.from_numpy(f_tgt_np).float().unsqueeze(1).to(device)

        if dual:
            if use_force_channel:
                train_dataset = TensorDataset(
                    meas_t,
                    coords_t,
                    u_tgt_t,
                    v_tgt_t,
                    f_tgt_t,
                )
            else:
                train_dataset = TensorDataset(meas_t, coords_t, u_tgt_t, v_tgt_t)
        else:
            train_dataset = TensorDataset(meas_t, coords_t, u_tgt_t)

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    optimizer = optim.Adam(deeponet.parameters(), lr=config.get('learning_rate', 1e-3))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=False)
    criterion = nn.MSELoss()

    # Optional Gaussian noise on branch inputs (applied only during DeepONet training).
    # Noise is injected after input scaling/normalization because meas_t is already normalized.
    noise_config_new = config.get('deeponet_branch_input_noise', None)
    noise_config_legacy = config.get('branch_input_noise', None)
    noise_config = noise_config_new if noise_config_new is not None else (noise_config_legacy or {})
    noise_enabled = bool(noise_config.get('enabled', False))
    noise_std = float(noise_config.get('std', 0.0))
    apply_noise_to_force = bool(noise_config.get('apply_to_force', True))
    clip_min = noise_config.get('clip_min', None)
    clip_max = noise_config.get('clip_max', None)

    if noise_config_new is None and noise_config_legacy is not None:
        print("  Note: 'branch_input_noise' is deprecated for DeepONet training; use 'deeponet_branch_input_noise'.")

    if noise_enabled and noise_std > 0.0:
        if dual:
            affected_channels = 'all channels'
            if use_force_channel and not apply_noise_to_force:
                affected_channels = 'u/v channels only'
        else:
            affected_channels = 'force channel only' if apply_noise_to_force else 'none (force excluded)'

        print(
            f"  DeepONet branch-input Gaussian noise enabled: std={noise_std:.6e}, "
            f"channels={affected_channels}, clip=[{clip_min}, {clip_max}]"
        )
    else:
        print("  DeepONet branch-input Gaussian noise disabled")

    # Physics (PINN) losses on dual-head model:
    #   r1 = u_t - v
    #   r2 = v_t + damping*v - c^2*(u_xx + u_yy) - f(x,y)
    physics_loss_enabled = bool(config.get('use_pinn_loss', False))
    physics_loss_weight = float(config.get('pde_loss_weight', 0.1))
    physics_loss_weight_uv = float(config.get('pde_loss_weight_uv', 1.0))
    physics_loss_weight_wave = float(config.get('pde_loss_weight_wave', 1.0))
    wave_speed = float(config.get('wave_speed', 1.0))
    damping_coeff = float(config.get('damping', 1.0))
    n_collocation_per_batch = int(config.get('n_collocation_per_batch', 256) or 256)
    pde_weight_start_factor = float(config.get('pde_loss_weight_start_factor', 0.01))
    pde_weight_end_factor = float(config.get('pde_loss_weight_end_factor', 1.0))
    pde_weight_warmup_epochs = int(config.get('pde_loss_weight_warmup_epochs', 0) or 0)
    pde_weight_schedule = str(config.get('pde_loss_weight_schedule', 'linear')).lower()
    rollout_pi_use_all_samples = bool(config.get('rollout_pi_use_all_samples', True))
    if physics_loss_enabled:
        if not dual:
            print("  Warning: physics loss (∂u/∂t = v) requires dual-head model; disabling.")
            physics_loss_enabled = False
        elif rollout_enabled:
            print(
                f"  Physics loss enabled (rollout mode): weight={physics_loss_weight}, "
                f"uv_weight={physics_loss_weight_uv}, wave_weight={physics_loss_weight_wave}, "
                f"c={wave_speed}, damping={damping_coeff}, "
                f"{n_collocation_per_batch} collocation pts/step at fixed t=rollout_dt"
            )
            print(
                f"  Rollout PI settings: use_all_samples={rollout_pi_use_all_samples}, "
                f"pde_weight_schedule={pde_weight_schedule}, start_factor={pde_weight_start_factor}, "
                f"end_factor={pde_weight_end_factor}, warmup_epochs={pde_weight_warmup_epochs}"
            )
        else:
            print(
                f"  Physics loss enabled: weight={physics_loss_weight}, "
                f"uv_weight={physics_loss_weight_uv}, wave_weight={physics_loss_weight_wave}, "
                f"c={wave_speed}, damping={damping_coeff}, "
                f"evaluated on training-batch coordinates via autograd"
            )
            print(
                f"  PI weight schedule: {pde_weight_schedule}, start_factor={pde_weight_start_factor}, "
                f"end_factor={pde_weight_end_factor}, warmup_epochs={pde_weight_warmup_epochs}"
            )
    else:
        print("  Physics loss disabled")

    train_losses, test_losses = [], []
    best_test_loss = float('inf')
    best_state = None
    log_every = int(config.get('deeponet_log_every', 10) or 10)
    if log_every <= 0:
        log_every = 10
    
    if dual and rollout_enabled:
        print(f"\nTraining: {len(train_sample_ids)} samples, Testing: {len(test_sample_ids)} samples")
    else:
        print(f"\nTraining: {len(train_data['u'])} points, Testing: {len(test_data['u'])} points")
    print(f"Logging DeepONet losses every {log_every} epoch(s)")
    print("Starting training...\n")
    
    for epoch in range(config['deeponet_n_epochs']):
        deeponet.train()
        current_physics_weight = 0.0
        if physics_loss_enabled:
            if pde_weight_warmup_epochs > 0 and epoch < pde_weight_warmup_epochs:
                if pde_weight_warmup_epochs == 1:
                    progress = 1.0
                else:
                    progress = epoch / float(max(1, pde_weight_warmup_epochs - 1))
                if pde_weight_schedule == 'cosine':
                    factor = pde_weight_start_factor + (
                        pde_weight_end_factor - pde_weight_start_factor
                    ) * 0.5 * (1.0 - np.cos(np.pi * progress))
                else:
                    factor = pde_weight_start_factor + (
                        pde_weight_end_factor - pde_weight_start_factor
                    ) * progress
            else:
                factor = pde_weight_end_factor
            current_physics_weight = physics_loss_weight * factor

        train_loss_epoch = 0.0
        train_loss_u_epoch = 0.0
        train_loss_v_epoch = 0.0
        train_loss_phys_epoch = 0.0
        train_loss_phys_uv_epoch = 0.0
        train_loss_phys_wave_epoch = 0.0

        if dual and rollout_enabled:
            for (sample_b,) in train_loader:
                optimizer.zero_grad()
                batch_sample_ids = sample_b.cpu().numpy().tolist()
                batch_start_indices = np.random.randint(0, max_start_idx + 1, size=len(batch_sample_ids))

                loss_u, loss_v = _compute_dual_rollout_batch_loss(
                    deeponet=deeponet,
                    sample_indices=batch_sample_ids,
                    start_indices=batch_start_indices,
                    u_fom=u_fom,
                    v_fom=v_fom,
                    sensor_x_indices=sensor_x_t,
                    sensor_y_indices=sensor_y_t,
                    raw_u_min=raw_u_min,
                    raw_u_max=raw_u_max,
                    raw_v_min=raw_v_min,
                    raw_v_max=raw_v_max,
                    input_channels=input_channels,
                    coords_rollout=coords_rollout,
                    rollout_weights=rollout_weights,
                    step_stride=step_stride,
                    criterion=criterion,
                    device=device,
                    nx=Nx,
                    ny=Ny,
                    chunk_size=rollout_chunk_size,
                    noise_enabled=noise_enabled,
                    noise_std=noise_std,
                    apply_noise_to_force=apply_noise_to_force,
                    clip_min=clip_min,
                    clip_max=clip_max,
                    force_grids=force_grids_t,
                )
                loss = loss_u + loss_v
                if physics_loss_enabled:
                    # PI in rollout mode should match one-step operator training:
                    # use state at sampled rollout start and evaluate at fixed t=rollout_dt.
                    sample_pairs = list(zip(batch_sample_ids, batch_start_indices))
                    if not rollout_pi_use_all_samples:
                        sample_pairs = sample_pairs[:1]

                    loss_phys_uv_vals = []
                    loss_phys_wave_vals = []
                    n_coll = n_collocation_per_batch
                    for s_phys, start_idx_phys in sample_pairs:
                        u_state = torch.from_numpy(u_fom[:, :, start_idx_phys, s_phys]).float().to(device)
                        v_state = torch.from_numpy(v_fom[:, :, start_idx_phys, s_phys]).float().to(device)
                        meas_coll = _build_dual_rollout_measurement_tensor(
                            u_field=u_state,
                            v_field=v_state,
                            sample_idx=s_phys,
                            sensor_x_indices=sensor_x_t,
                            sensor_y_indices=sensor_y_t,
                            raw_u_min=raw_u_min,
                            raw_u_max=raw_u_max,
                            raw_v_min=raw_v_min,
                            raw_v_max=raw_v_max,
                            input_channels=input_channels,
                            force_grids=force_grids_t,
                        )

                        coords_coll = torch.stack([
                            torch.rand(n_coll, device=device),
                            torch.rand(n_coll, device=device),
                            torch.full((n_coll,), rollout_dt, device=device),
                        ], dim=1).requires_grad_(True)
                        u_coll, v_coll = deeponet(
                            meas_coll.expand(n_coll, -1, -1, -1), coords_coll
                        )
                        grad_u_coll = torch.autograd.grad(
                            u_coll.sum(), coords_coll, create_graph=True
                        )[0]
                        grad_v_coll = torch.autograd.grad(
                            v_coll.sum(), coords_coll, create_graph=True
                        )[0]

                        du_dt_coll = grad_u_coll[:, 2]
                        dv_dt_coll = grad_v_coll[:, 2]
                        du_dx_coll = grad_u_coll[:, 0]
                        du_dy_coll = grad_u_coll[:, 1]
                        d2u_dx2_coll = torch.autograd.grad(
                            du_dx_coll.sum(), coords_coll, create_graph=True
                        )[0][:, 0]
                        d2u_dy2_coll = torch.autograd.grad(
                            du_dy_coll.sum(), coords_coll, create_graph=True
                        )[0][:, 1]

                        if use_force_channel and force_fields_t is not None:
                            force_grid = force_fields_t[:, :, s_phys]
                            force_coll = _sample_force_at_coords(force_grid, coords_coll)
                        else:
                            force_coll = torch.zeros_like(v_coll)

                        wave_res_coll = (
                            dv_dt_coll
                            + damping_coeff * v_coll
                            - (wave_speed ** 2) * (d2u_dx2_coll + d2u_dy2_coll)
                            - force_coll
                        )

                        loss_phys_uv_vals.append(criterion(du_dt_coll, v_coll))
                        loss_phys_wave_vals.append(
                            criterion(wave_res_coll, torch.zeros_like(wave_res_coll))
                        )

                    loss_phys_uv = torch.stack(loss_phys_uv_vals).mean()
                    loss_phys_wave = torch.stack(loss_phys_wave_vals).mean()
                    loss_phys = (
                        physics_loss_weight_uv * loss_phys_uv
                        + physics_loss_weight_wave * loss_phys_wave
                    )
                    loss = loss + current_physics_weight * loss_phys
                    train_loss_phys_epoch += loss_phys.item()
                    train_loss_phys_uv_epoch += loss_phys_uv.item()
                    train_loss_phys_wave_epoch += loss_phys_wave.item()
                loss.backward()
                optimizer.step()

                train_loss_epoch += loss.item()
                train_loss_u_epoch += loss_u.item()
                train_loss_v_epoch += loss_v.item()
        elif dual:
            for batch in train_loader:
                optimizer.zero_grad()

                if use_force_channel:
                    meas_b, coords_b, u_b, v_b, f_b = batch
                else:
                    meas_b, coords_b, u_b, v_b = batch
                    f_b = None

                noisy_meas_b = _apply_deeponet_branch_input_noise(
                    meas_b,
                    noise_enabled=noise_enabled,
                    noise_std=noise_std,
                    apply_noise_to_force=apply_noise_to_force,
                    clip_min=clip_min,
                    clip_max=clip_max,
                )

                if physics_loss_enabled:
                    # Enable autograd on the time coordinate so ∂u/∂t can be computed.
                    coords_phys = coords_b.detach().requires_grad_(True)
                    u_pred, v_pred = deeponet(noisy_meas_b, coords_phys)
                else:
                    coords_phys = None
                    u_pred, v_pred = deeponet(noisy_meas_b, coords_b)
                loss_u = criterion(u_pred.unsqueeze(1), u_b)
                loss_v = criterion(v_pred.unsqueeze(1), v_b)
                loss = loss_u + loss_v
                if physics_loss_enabled:
                    grad_u = torch.autograd.grad(
                        u_pred.sum(), coords_phys, create_graph=True
                    )[0]
                    grad_v = torch.autograd.grad(
                        v_pred.sum(), coords_phys, create_graph=True
                    )[0]

                    du_dt = grad_u[:, 2]
                    dv_dt = grad_v[:, 2]
                    du_dx = grad_u[:, 0]
                    du_dy = grad_u[:, 1]
                    d2u_dx2 = torch.autograd.grad(
                        du_dx.sum(), coords_phys, create_graph=True
                    )[0][:, 0]
                    d2u_dy2 = torch.autograd.grad(
                        du_dy.sum(), coords_phys, create_graph=True
                    )[0][:, 1]

                    if use_force_channel and f_b is not None:
                        force_vals = f_b.squeeze(1)
                    else:
                        force_vals = torch.zeros_like(v_pred)

                    wave_res = (
                        dv_dt
                        + damping_coeff * v_pred
                        - (wave_speed ** 2) * (d2u_dx2 + d2u_dy2)
                        - force_vals
                    )
                    loss_phys_uv = criterion(du_dt, v_pred)
                    loss_phys_wave = criterion(wave_res, torch.zeros_like(wave_res))
                    loss_phys = (
                        physics_loss_weight_uv * loss_phys_uv
                        + physics_loss_weight_wave * loss_phys_wave
                    )
                    loss = loss + current_physics_weight * loss_phys
                    train_loss_phys_epoch += loss_phys.item()
                    train_loss_phys_uv_epoch += loss_phys_uv.item()
                    train_loss_phys_wave_epoch += loss_phys_wave.item()
                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item()
                train_loss_u_epoch += loss_u.item()
                train_loss_v_epoch += loss_v.item()
        else:
            for meas_b, coords_b, u_b in train_loader:
                optimizer.zero_grad()

                noisy_meas_b = _apply_deeponet_branch_input_noise(
                    meas_b,
                    noise_enabled=noise_enabled,
                    noise_std=noise_std,
                    apply_noise_to_force=apply_noise_to_force,
                    clip_min=clip_min,
                    clip_max=clip_max,
                )

                pred = deeponet(noisy_meas_b, coords_b).unsqueeze(1)
                loss_u = criterion(pred, u_b)
                loss = loss_u
                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item()
                train_loss_u_epoch += loss_u.item()
        
        train_loss_epoch /= len(train_loader)
        train_losses.append(train_loss_epoch)
        
        deeponet.eval()
        with torch.no_grad():
            if dual and rollout_enabled:
                eval_sample_ids = np.array(test_sample_ids, copy=True)
                if eval_max_samples is not None and len(eval_sample_ids) > int(eval_max_samples):
                    eval_sample_ids = eval_sample_ids[:int(eval_max_samples)]

                test_loss = 0.0
                test_loss_u_total = 0.0
                test_loss_v_total = 0.0
                n_eval_batches = 0
                for start_offset in range(0, len(eval_sample_ids), rollout_batch_size):
                    batch_eval_ids = eval_sample_ids[start_offset:start_offset + rollout_batch_size].tolist()
                    batch_eval_starts = [eval_start_idx] * len(batch_eval_ids)
                    test_loss_u_batch, test_loss_v_batch = _compute_dual_rollout_batch_loss(
                        deeponet=deeponet,
                        sample_indices=batch_eval_ids,
                        start_indices=batch_eval_starts,
                        u_fom=u_fom,
                        v_fom=v_fom,
                        sensor_x_indices=sensor_x_t,
                        sensor_y_indices=sensor_y_t,
                        raw_u_min=raw_u_min,
                        raw_u_max=raw_u_max,
                        raw_v_min=raw_v_min,
                        raw_v_max=raw_v_max,
                        input_channels=input_channels,
                        coords_rollout=coords_rollout,
                        rollout_weights=rollout_weights,
                        step_stride=step_stride,
                        criterion=criterion,
                        device=device,
                        nx=Nx,
                        ny=Ny,
                        chunk_size=rollout_chunk_size,
                        noise_enabled=False,
                        noise_std=0.0,
                        apply_noise_to_force=apply_noise_to_force,
                        clip_min=None,
                        clip_max=None,
                        force_grids=force_grids_t,
                    )
                    test_loss_u_total += test_loss_u_batch.item()
                    test_loss_v_total += test_loss_v_batch.item()
                    test_loss += test_loss_u_batch.item() + test_loss_v_batch.item()
                    n_eval_batches += 1

                test_loss_u = test_loss_u_total / max(1, n_eval_batches)
                test_loss_v = test_loss_v_total / max(1, n_eval_batches)
                test_loss = test_loss / max(1, n_eval_batches)
            elif dual:
                u_pred_te, v_pred_te = deeponet(test_meas, test_coords)
                test_loss_u = criterion(u_pred_te.unsqueeze(1), test_u_tgt)
                test_loss_v = criterion(v_pred_te.unsqueeze(1), test_v_tgt)
                test_loss = (test_loss_u + test_loss_v).item()
            else:
                pred_te = deeponet(test_meas, test_coords).unsqueeze(1)
                test_loss_u = criterion(pred_te, test_u_tgt)
                test_loss = test_loss_u.item()
        test_losses.append(test_loss)
        scheduler.step(test_loss)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = {k: v.cpu().clone() for k, v in deeponet.state_dict().items()}
        
        current_lr = optimizer.param_groups[0]['lr']
        if (epoch + 1) % log_every == 0 or epoch == 0 or (epoch + 1) == config['deeponet_n_epochs']:
            if dual:
                avg_train_u = train_loss_u_epoch / len(train_loader)
                avg_train_v = train_loss_v_epoch / len(train_loader)
                test_loss_u_value = test_loss_u if isinstance(test_loss_u, float) else test_loss_u.item()
                test_loss_v_value = test_loss_v if isinstance(test_loss_v, float) else test_loss_v.item()
                phys_str = ""
                if physics_loss_enabled:
                    avg_train_phys = train_loss_phys_epoch / len(train_loader)
                    avg_train_phys_uv = train_loss_phys_uv_epoch / len(train_loader)
                    avg_train_phys_wave = train_loss_phys_wave_epoch / len(train_loader)
                    phys_str = (
                        f" | PhysW: {current_physics_weight:.3e} | Phys(total/r1/r2): {avg_train_phys:.6e}/"
                        f"{avg_train_phys_uv:.6e}/{avg_train_phys_wave:.6e}"
                    )
                print(
                    f"Epoch {epoch+1:4d}/{config['deeponet_n_epochs']} | "
                    f"Train(total/u/v): {train_loss_epoch:.6e}/{avg_train_u:.6e}/{avg_train_v:.6e} | "
                    f"Test(total/u/v): {test_loss:.6e}/{test_loss_u_value:.6e}/{test_loss_v_value:.6e} | "
                    f"LR: {current_lr:.3e}{phys_str}"
                )
            else:
                print(
                    f"Epoch {epoch+1:4d}/{config['deeponet_n_epochs']} | "
                    f"Train: {train_loss_epoch:.6e} | Test: {test_loss:.6e} | LR: {current_lr:.3e}"
                )
        
        if current_lr < 5e-6:
            print(f"\nEarly stopping at epoch {epoch+1}: LR={current_lr:.3e} < 5e-6")
            break
    
    # Load best model
    deeponet.load_state_dict(best_state)
    deeponet.eval()
    
    # Save checkpoint to output checkpoints directory.
    ckpt_dir = os.path.join(output_dir, 'checkpoints') if output_dir is not None else models_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    problem_suffix = problem_type
    ckpt_path = os.path.join(ckpt_dir, f'deeponet_{problem_suffix}.pth')
    torch.save({
        'model_state_dict': best_state,
        'config': config,
        'input_scale': input_scale,
        'dual': dual,
        'n_input_channels': int(input_channels) if dual else 1,
        'normalization': {
            'raw_u_min': float(raw_u_min) if dual else None,
            'raw_u_max': float(raw_u_max) if dual else None,
            'raw_v_min': float(raw_v_min) if dual else None,
            'raw_v_max': float(raw_v_max) if dual else None,
            'raw_f_min': float(raw_f_min),
            'raw_f_max': float(raw_f_max),
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
