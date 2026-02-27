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

from .models import MLP, DeepONet


def train_trunk(
    config: dict,
    svd_data: dict,
    device: torch.device,
    output_dir: str = None,
    models_dir: str = None,
    problem_type: str = 'free_evolution',
) -> dict:
    """Train trunk network on SVD basis functions."""
    
    print("\n" + "=" * 70)
    print("TRAINING TRUNK NETWORK (SVD Basis)")
    print("=" * 70)
    
    U_basis = svd_data['basis']
    grid_info = svd_data['grid_info']
    Nx, Ny, Nt = grid_info.astype(int)
    n_space_time = U_basis.shape[0]
    
    n_modes = config['n_modes']
    targets_raw = U_basis[:, :n_modes]
    
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
        targets_raw = targets_raw[sample_idx]
    else:
        if max_trunk_points:
            print(f"Using all {n_space_time} space-time points (≤ max_trunk_points={max_trunk_points})")
    
        else:
            print(f"Using all {n_space_time} space-time points")
    
    # Normalize
    targets_min = targets_raw.min()
    targets_max = targets_raw.max()
    targets_range = targets_max - targets_min
    targets_normalized = 2 * (targets_raw - targets_min) / targets_range - 1
    
    # Train/test split
    n_total = coords.shape[0]
    n_train = int(n_total * config['train_test_split'])
    indices = np.random.permutation(n_total)
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    
    coords_train = torch.from_numpy(coords[train_idx]).float().to(device)
    targets_train = torch.from_numpy(targets_normalized[train_idx]).float().to(device)
    coords_test = torch.from_numpy(coords[test_idx]).float().to(device)
    targets_test = torch.from_numpy(targets_normalized[test_idx]).float().to(device)
    
    # Build model
    trunk = MLP(3, config['trunk_hidden_dim'], n_modes, config['trunk_n_layers']).to(device)
    optimizer = optim.Adam(trunk.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=False)
    criterion = nn.MSELoss()
    
    # Training loop
    train_dataset = TensorDataset(coords_train, targets_train)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    train_losses, test_losses = [], []
    best_test_loss = float('inf')
    best_state = None
    
    print(f"Training: {len(train_idx)} samples, Testing: {len(test_idx)} samples")
    
    for epoch in range(config['trunk_n_epochs']):
        trunk.train()
        train_loss_epoch = 0.0
        for coords_batch, targets_batch in train_loader:
            optimizer.zero_grad()
            pred = trunk(coords_batch)
            loss = criterion(pred, targets_batch)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()
        
        train_loss_epoch /= len(train_loader)
        train_losses.append(train_loss_epoch)
        
        trunk.eval()
        with torch.no_grad():
            pred_test = trunk(coords_test)
            test_loss = criterion(pred_test, targets_test).item()
        test_losses.append(test_loss)
        scheduler.step(test_loss)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = {k: v.cpu().clone() for k, v in trunk.state_dict().items()}
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d}/{config['trunk_n_epochs']} | Train: {train_loss_epoch:.6e} | Test: {test_loss:.6e}")
    
    trunk.load_state_dict(best_state)
    trunk.eval()
    
    # Save checkpoint to output directory
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f'trunk_svd_{problem_type}.pth')
    torch.save({
        'model_state_dict': best_state,
        'config': config,
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
) -> dict:
    """Train branch network on IC → SVD coefficients mapping."""
    
    print("\n" + "=" * 70)
    print("TRAINING BRANCH NETWORK (IC → SVD Coefficients)")
    print("=" * 70)
    
    # Extract IC sensors
    Nx, Ny, Nt, N_samples = u_fom.shape
    n_modes = config['n_modes']
    n_sensors = config['n_sensors']
    
    # Sample training samples if max_branch_samples specified
    max_branch_samples = config.get('max_branch_samples', None)
    if max_branch_samples and N_samples > max_branch_samples:
        print(f"Sampling {max_branch_samples} samples from {N_samples} total samples")
        sample_indices = np.random.choice(N_samples, max_branch_samples, replace=False)
        u_fom_subset = u_fom[:, :, :, sample_indices]
        N_samples_train = max_branch_samples
    else:
        if max_branch_samples:
            print(f"Using all {N_samples} samples (≤ max_branch_samples={max_branch_samples})")
        else:
            print(f"Using all {N_samples} samples")
        u_fom_subset = u_fom
        N_samples_train = N_samples
    
    sensor_x_indices = np.linspace(0, Nx - 1, n_sensors, dtype=int)
    sensor_y_indices = np.linspace(0, Ny - 1, n_sensors, dtype=int)
    
    ic_sensors = []
    for s in range(N_samples_train):
        u_ic = u_fom_subset[:, :, 0, s]
        sensors = [u_ic[si, sj] for si in sensor_x_indices for sj in sensor_y_indices]
        ic_sensors.append(sensors)
    
    ic_sensors = np.array(ic_sensors)
    ic_min = ic_sensors.min(axis=0, keepdims=True)
    ic_max = ic_sensors.max(axis=0, keepdims=True)
    ic_range = ic_max - ic_min
    ic_norm = 2 * (ic_sensors - ic_min) / (ic_range + 1e-10) - 1
    
    # Get SVD coefficients (ground truth for branch)
    VT = svd_data['coefficients'][:n_modes, :]  # (n_modes, N_samples_total)
    if max_branch_samples and N_samples > max_branch_samples:
        VT = VT[:, sample_indices]  # Select same samples
    Sigma = svd_data['singular_values'][:n_modes]
    target_coeffs = (Sigma[:, None] * VT).T  # (N_samples_train, n_modes)
    
    # Normalize coefficients
    coeff_min = target_coeffs.min()
    coeff_max = target_coeffs.max()
    coeff_range = coeff_max - coeff_min
    target_norm = 2 * (target_coeffs - coeff_min) / (coeff_range + 1e-10) - 1
    
    # Train/test split
    indices = np.random.permutation(N_samples_train)
    train_size = int(N_samples_train * config['train_test_split'])
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    
    ic_train = torch.from_numpy(ic_norm[train_idx]).float().to(device)
    target_train = torch.from_numpy(target_norm[train_idx]).float().to(device)
    ic_test = torch.from_numpy(ic_norm[test_idx]).float().to(device)
    target_test = torch.from_numpy(target_norm[test_idx]).float().to(device)
    
    # Build model
    ic_dim = n_sensors * n_sensors
    input_scale = 0.1 if problem_type == 'constant_force' else 1.0
    branch = MLP(ic_dim, config['branch_hidden_dim'], n_modes, config['branch_n_layers'], input_scale=input_scale).to(device)
    optimizer = optim.Adam(branch.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=False)
    criterion = nn.MSELoss()
    
    # Training loop
    train_dataset = TensorDataset(ic_train, target_train)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    train_losses, test_losses = [], []
    best_test_loss = float('inf')
    best_state = None
    
    print(f"Training: {len(train_idx)} samples, Testing: {len(test_idx)} samples")
    
    for epoch in range(config['branch_n_epochs']):
        branch.train()
        train_loss_epoch = 0.0
        for ic_batch, target_batch in train_loader:
            optimizer.zero_grad()
            pred = branch(ic_batch)
            loss = criterion(pred, target_batch)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()
        
        train_loss_epoch /= len(train_loader)
        train_losses.append(train_loss_epoch)
        
        branch.eval()
        with torch.no_grad():
            pred_test = branch(ic_test)
            test_loss = criterion(pred_test, target_test).item()
        test_losses.append(test_loss)
        scheduler.step(test_loss)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = {k: v.cpu().clone() for k, v in branch.state_dict().items()}
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d}/{config['branch_n_epochs']} | Train: {train_loss_epoch:.6e} | Test: {test_loss:.6e}")
    
    branch.load_state_dict(best_state)
    branch.eval()
    
    # Save checkpoint to output directory
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f'branch_svd_{problem_type}.pth')
    torch.save({
        'model_state_dict': best_state,
        'config': config,
        'input_scale': input_scale,
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
) -> dict:
    """Joint training of DeepONet on ground truth data.
    
    Trains the complete DeepONet (trunk + branch) on raw field data,
    optionally initializing from pretrained trunk and branch networks.
    
    Args:
        problem_type: 'free_evolution' or 'constant_force'
        f_data: Force field data (Nx, Ny, N_samples), required only for 'constant_force'
    """
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "=" * 70)
    print(f"TRAINING DEEPONET (Joint Trunk + Branch - {problem_type.upper()})")
    print("=" * 70)
    
    # Validate inputs
    if problem_type == 'constant_force' and f_data is None:
        raise ValueError("f_data must be provided for 'constant_force' problem")
    
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
        if f_data is not None:
            f_data = f_data[:, :, sample_indices]
        N_samples = max_deeponet_samples
    else:
        if max_deeponet_samples:
            print(f"Using all {N_samples} samples (≤ max_deeponet_samples={max_deeponet_samples})")
        else:
            print(f"Using all {N_samples} samples")
    
    # Initialize trunk network
    trunk = MLP(3, config['trunk_hidden_dim'], n_modes, config['trunk_n_layers']).to(device)
    if trunk_pretrained_path and os.path.exists(trunk_pretrained_path):
        print(f"Loading pretrained trunk from: {trunk_pretrained_path}")
        trunk_ckpt = torch.load(trunk_pretrained_path, map_location=device, weights_only=False)
        trunk_state = trunk_ckpt.get('model_state_dict', trunk_ckpt)
        trunk.load_state_dict(trunk_state)
        print("  ✓ Trunk loaded")
    else:
        print("  Initializing trunk from scratch")
    
    # Initialize branch network
    measurement_dim = n_sensors * n_sensors
    input_scale = 0.1 if problem_type == 'constant_force' else 1.0
    branch = MLP(measurement_dim, config['branch_hidden_dim'], n_modes, config['branch_n_layers'], input_scale=input_scale).to(device)
    if branch_pretrained_path and os.path.exists(branch_pretrained_path):
        print(f"Loading pretrained branch from: {branch_pretrained_path}")
        branch_ckpt = torch.load(branch_pretrained_path, map_location=device, weights_only=False)
        branch_state = branch_ckpt.get('model_state_dict', branch_ckpt)
        branch.load_state_dict(branch_state)
        print("  ✓ Branch loaded")
    else:
        print("  Initializing branch from scratch")
    
    # Create DeepONet with problem-specific branch
    if problem_type == 'free_evolution':
        deeponet = DeepONet(
            trunk=trunk,
            branch_ic=branch,
            problem_type='free_evolution',
            wave_speed=config.get('wave_speed', 1.0),
            damping=config.get('damping', 1.0)
        ).to(device)
    else:  # constant_force
        deeponet = DeepONet(
            trunk=trunk,
            branch_force=branch,
            problem_type='constant_force',
            wave_speed=config.get('wave_speed', 1.0),
            damping=config.get('damping', 1.0)
        ).to(device)
    
    print(f"\nDeepONet architecture:")
    print(f"  Trunk: 3 → {config['trunk_hidden_dim']} ({config['trunk_n_layers']} layers) → {n_modes}")
    print(f"  Branch: {measurement_dim} → {config['branch_hidden_dim']} ({config['branch_n_layers']} layers) → {n_modes}")
    print(f"  Problem type: {problem_type}")
    
    # Prepare training data
    print("\nPreparing training data...")
    
    # Extract measurements based on problem type
    sensor_x_indices = np.linspace(0, Nx - 1, n_sensors, dtype=int)
    sensor_y_indices = np.linspace(0, Ny - 1, n_sensors, dtype=int)
    
    if problem_type == 'free_evolution':
        # Extract IC measurements from initial condition
        measurements = []
        for s in range(N_samples):
            u_ic = u_fom[:, :, 0, s]
            sensors = [u_ic[si, sj] for si in sensor_x_indices for sj in sensor_y_indices]
            measurements.append(sensors)
        measurements = np.array(measurements)  # (N_samples, n_sensors^2)
        measurements_min = measurements.min()
        measurements_max = measurements.max()
    else:  # constant_force
        # Extract force measurements
        measurements = []
        for s in range(N_samples):
            f_field = f_data[:, :, s]
            sensors = [f_field[si, sj] for si in sensor_x_indices for sj in sensor_y_indices]
            measurements.append(sensors)
        measurements = np.array(measurements)  # (N_samples, n_sensors^2)
        measurements_min = measurements.min()
        measurements_max = measurements.max()
    
    # Normalize measurements
    measurements_range = measurements_max - measurements_min
    measurements_norm = 2 * (measurements - measurements_min) / (measurements_range + 1e-10) - 1
    
    # Generate coordinate grid
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    t_vals = np.linspace(0, 1, Nt)
    X, Y, T = np.meshgrid(x, y, t_vals, indexing='ij')
    coords_all = np.stack([X.flatten('F'), Y.flatten('F'), T.flatten('F')], axis=1)  # (Nx*Ny*Nt, 3)
    
    # Normalize field values
    u_min = u_fom.min()
    u_max = u_fom.max()
    u_range = u_max - u_min
    u_fom_norm = 2 * (u_fom - u_min) / (u_range + 1e-10) - 1
    u_targets_all = u_fom_norm.reshape(-1, N_samples, order='F')  # (Nx*Ny*Nt, N_samples)
    
    # Sample points for training (to avoid memory issues)
    n_points_per_sample = config.get('n_points_per_sample', 1000)
    n_total_points = min(n_points_per_sample * N_samples, coords_all.shape[0] * N_samples)
    
    # Create training dataset
    coords_list = []
    meas_list = []
    targets_list = []
    
    for s in range(N_samples):
        # Sample random space-time points for this sample
        n_pts = min(n_points_per_sample, coords_all.shape[0])
        pt_indices = np.random.choice(coords_all.shape[0], n_pts, replace=False)
        
        coords_sample = coords_all[pt_indices]  # (n_pts, 3)
        meas_sample = np.tile(measurements_norm[s], (n_pts, 1))  # (n_pts, n_sensors^2)
        targets_sample = u_targets_all[pt_indices, s]  # (n_pts,)
        
        coords_list.append(coords_sample)
        meas_list.append(meas_sample)
        targets_list.append(targets_sample)
    
    coords_train = np.vstack(coords_list)
    meas_train = np.vstack(meas_list)
    targets_train = np.hstack(targets_list)
    
    print(f"  Training points: {len(targets_train)}")
    print(f"  Measurement range: [{measurements.min():.6f}, {measurements.max():.6f}]")
    print(f"  Target range: [{u_fom.min():.6f}, {u_fom.max():.6f}]")
    
    # Convert to tensors
    coords_tensor = torch.from_numpy(coords_train).float().to(device)
    meas_tensor = torch.from_numpy(meas_train).float().to(device)
    targets_tensor = torch.from_numpy(targets_train).float().unsqueeze(1).to(device)
    
    # Train/test split
    n_total = len(targets_train)
    n_train = int(n_total * config.get('train_test_split', 0.8))
    indices = torch.randperm(n_total)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    train_dataset = TensorDataset(meas_tensor[train_idx], coords_tensor[train_idx], targets_tensor[train_idx])
    test_meas = meas_tensor[test_idx]
    test_coords = coords_tensor[test_idx]
    test_targets = targets_tensor[test_idx]
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Setup optimizer
    optimizer = optim.Adam(deeponet.parameters(), lr=config.get('learning_rate', 1e-3))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=False)
    criterion = nn.MSELoss()
    
    # Training loop
    train_losses, test_losses = [], []
    best_test_loss = float('inf')
    best_state = None
    
    print(f"\nTraining: {len(train_idx)} points, Testing: {len(test_idx)} points")
    print("Starting training...\n")
    
    for epoch in range(config['deeponet_n_epochs']):
        deeponet.train()
        train_loss_epoch = 0.0
        
        for meas_batch, coords_batch, targets_batch in train_loader:
            optimizer.zero_grad()
            pred = deeponet(meas_batch, coords_batch).unsqueeze(1)  # (batch, 1)
            loss = criterion(pred, targets_batch)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()
        
        train_loss_epoch /= len(train_loader)
        train_losses.append(train_loss_epoch)
        
        # Validation
        deeponet.eval()
        with torch.no_grad():
            pred_test = deeponet(test_meas, test_coords).unsqueeze(1)
            test_loss = criterion(pred_test, test_targets).item()
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
    
    # Save checkpoint to output directory
    os.makedirs(output_dir, exist_ok=True)
    problem_suffix = 'free_evolution' if problem_type == 'free_evolution' else 'constant_force'
    ckpt_path = os.path.join(output_dir, f'deeponet_{problem_suffix}.pth')
    torch.save({
        'model_state_dict': best_state,
        'trunk_state_dict': {k: v for k, v in best_state.items() if k.startswith('trunk.')},
        'branch_state_dict': {k: v for k, v in best_state.items() if k.startswith('branch.')},
        'config': config,
        'input_scale': input_scale,
        'normalization': {
            'measurements_min': measurements_min,
            'measurements_max': measurements_max,
            'u_min': u_min,
            'u_max': u_max,
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
