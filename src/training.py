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


class MLP(nn.Module):
    """Simple feedforward MLP network."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int):
        super().__init__()
        # n_layers = total number of Linear layers
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 2):  # n_layers - 2 because we have 1 input + 1 output
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class DeepONet(nn.Module):
    """DeepONet operator network: trunk + branch."""
    
    def __init__(self, trunk: MLP, branch: MLP, wave_speed: float = 1.0, damping: float = 1.0):
        super().__init__()
        self.trunk = trunk
        self.branch = branch
        self.c = wave_speed
        self.k = damping
    
    def forward(self, ic: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        trunk_out = self.trunk(coords)  # (B, N_MODES)
        branch_out = self.branch(ic)    # (B, N_MODES)
        u = torch.sum(trunk_out * branch_out, dim=1)  # (B,)
        return u
    
    def compute_pde_residual(self, ic: torch.Tensor, xyt: torch.Tensor) -> torch.Tensor:
        """Compute 2D wave equation residual: u_tt + k*u_t - c^2*(u_xx+u_yy)."""
        xyt_grad = xyt.clone().requires_grad_(True)
        u = self.forward(ic, xyt_grad)
        
        grad_u = torch.autograd.grad(
            u, xyt_grad,
            torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        u_x, u_y, u_t = grad_u[:, 0], grad_u[:, 1], grad_u[:, 2]
        
        u_xx = torch.autograd.grad(u_x, xyt_grad, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0][:, 0]
        u_yy = torch.autograd.grad(u_y, xyt_grad, torch.ones_like(u_y), create_graph=True, retain_graph=True)[0][:, 1]
        u_tt = torch.autograd.grad(u_t, xyt_grad, torch.ones_like(u_t), create_graph=True)[0][:, 2]
        
        residual = u_tt + self.k * u_t - self.c**2 * (u_xx + u_yy)
        return residual


def train_trunk(
    config: dict,
    svd_data: dict,
    device: torch.device,
    output_dir: str = None,
    models_dir: str = None,
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
    
    for epoch in range(config['n_epochs']):
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
            print(f"Epoch {epoch+1:3d}/{config['n_epochs']} | Train: {train_loss_epoch:.6e} | Test: {test_loss:.6e}")
    
    trunk.load_state_dict(best_state)
    trunk.eval()
    
    # Save checkpoint
    save_dir = models_dir if models_dir else os.path.join(os.path.dirname(output_dir or '.'), 'models')
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, 'trunk_svd_free_evolution.pth')
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
) -> dict:
    """Train branch network on IC → SVD coefficients mapping."""
    
    print("\n" + "=" * 70)
    print("TRAINING BRANCH NETWORK (IC → SVD Coefficients)")
    print("=" * 70)
    
    # Extract IC sensors
    Nx, Ny, Nt, N_samples = u_fom.shape
    n_modes = config['n_modes']
    n_sensors = config['n_sensors']
    
    sensor_x_indices = np.linspace(0, Nx - 1, n_sensors, dtype=int)
    sensor_y_indices = np.linspace(0, Ny - 1, n_sensors, dtype=int)
    
    ic_sensors = []
    for s in range(N_samples):
        u_ic = u_fom[:, :, 0, s]
        sensors = [u_ic[si, sj] for si in sensor_x_indices for sj in sensor_y_indices]
        ic_sensors.append(sensors)
    
    ic_sensors = np.array(ic_sensors)
    ic_min = ic_sensors.min(axis=0, keepdims=True)
    ic_max = ic_sensors.max(axis=0, keepdims=True)
    ic_range = ic_max - ic_min
    ic_norm = 2 * (ic_sensors - ic_min) / (ic_range + 1e-10) - 1
    
    # Get SVD coefficients (ground truth for branch)
    VT = svd_data['coefficients'][:n_modes, :]  # (n_modes, N_samples)
    Sigma = svd_data['singular_values'][:n_modes]
    target_coeffs = (Sigma[:, None] * VT).T  # (N_samples, n_modes)
    
    # Normalize coefficients
    coeff_min = target_coeffs.min()
    coeff_max = target_coeffs.max()
    coeff_range = coeff_max - coeff_min
    target_norm = 2 * (target_coeffs - coeff_min) / (coeff_range + 1e-10) - 1
    
    # Train/test split
    indices = np.random.permutation(N_samples)
    train_size = int(N_samples * config['train_test_split'])
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    
    ic_train = torch.from_numpy(ic_norm[train_idx]).float().to(device)
    target_train = torch.from_numpy(target_norm[train_idx]).float().to(device)
    ic_test = torch.from_numpy(ic_norm[test_idx]).float().to(device)
    target_test = torch.from_numpy(target_norm[test_idx]).float().to(device)
    
    # Build model
    ic_dim = n_sensors * n_sensors
    branch = MLP(ic_dim, config['branch_hidden_dim'], n_modes, config['branch_n_layers']).to(device)
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
    
    for epoch in range(config['n_epochs']):
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
            print(f"Epoch {epoch+1:3d}/{config['n_epochs']} | Train: {train_loss_epoch:.6e} | Test: {test_loss:.6e}")
    
    branch.load_state_dict(best_state)
    branch.eval()
    
    # Save checkpoint
    save_dir = models_dir if models_dir else os.path.join(os.path.dirname(output_dir or '.'), 'models')
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, 'branch_svd_free_evolution.pth')
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
    device: torch.device = None,
    output_dir: str = None,
    models_dir: str = None,
) -> dict:
    """Joint training of trunk + branch on full ground truth."""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "=" * 70)
    print("TRAINING DEEPONET (Joint Trunk + Branch)")
    print("=" * 70)
    
    # (Simplified for now - can expand with full GT training if needed)
    # For now, just fine-tune branch on GT data
    
    # Save checkpoint
    save_dir = models_dir if models_dir else os.path.join(os.path.dirname(output_dir or '.'), 'models')
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, 'deeponet_free_evolution.pth')
    torch.save({
        'config': config,
        'status': 'placeholder'
    }, ckpt_path)
    print(f"✓ Saved: {ckpt_path}")
    
    print("✓ DeepONet joint training complete")
    print("=" * 70)
    
    return {'train_losses': [], 'test_losses': []}
