"""
Simplified Trunk Network Training on SVD Basis Functions
- Supervised learning: Direct MSE on SVD basis values
- No FFT, no physics loss yet
- Designed for 18 SVD modes with [-0.003, 0.003] range
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
N_MODES = 18
TRUNK_HIDDEN_DIM = 128
TRUNK_N_LAYERS = 4
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
N_EPOCHS = 500
TRAIN_TEST_SPLIT = 0.8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")
print(f"Config: {N_MODES} modes, hidden_dim={TRUNK_HIDDEN_DIM}, n_layers={TRUNK_N_LAYERS}")

# ============================================================
# 1. LOAD SVD DATA
# ============================================================
print("\n1. Loading SVD data...")
svd_data = np.load(os.path.join(SCRIPT_DIR, 'data/svd_basis_data.npy'), allow_pickle=True).item()
U_basis = svd_data['basis']  # (Nx*Ny*Nt, p)
Sigma = svd_data['singular_values']
grid_info = svd_data['grid_info']

Nx, Ny, Nt = grid_info.astype(int)
n_space_time = U_basis.shape[0]
print(f"Grid: {Nx} x {Ny} x {Nt} = {n_space_time} space-time points")
print(f"SVD basis available: {U_basis.shape[1]} modes")
print(f"Using first {N_MODES} modes")

# Extract first N_MODES
targets_raw = U_basis[:, :N_MODES]  # (n_space_time, N_MODES)
print(f"Target data shape: {targets_raw.shape}")
print(f"Target range: [{targets_raw.min():.6e}, {targets_raw.max():.6e}]")

# ============================================================
# 2. GENERATE COORDINATES (x, y, t)
# ============================================================
print("\n2. Generating coordinates...")
# Must match the Fortran order used in process_data_svd.py
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
t = np.linspace(0, 1, Nt)

# Create meshgrid with ij indexing, flatten with Fortran order
X, Y, T = np.meshgrid(x, y, t, indexing='ij')
coords = np.stack([X.flatten('F'), Y.flatten('F'), T.flatten('F')], axis=1)
print(f"Coordinates shape: {coords.shape}")
print(f"x range: [{x.min():.3f}, {x.max():.3f}]")
print(f"y range: [{y.min():.3f}, {y.max():.3f}]")
print(f"t range: [{t.min():.3f}, {t.max():.3f}]")

# ============================================================
# 3. NORMALIZE TARGETS TO [-1, 1]
# ============================================================
print("\n3. Normalizing targets...")
# Store normalization for later denormalization
targets_min = targets_raw.min()
targets_max = targets_raw.max()
targets_range = targets_max - targets_min

targets_normalized = 2 * (targets_raw - targets_min) / targets_range - 1
print(f"Normalized range: [{targets_normalized.min():.6f}, {targets_normalized.max():.6f}]")

# ============================================================
# 4. TRAIN/TEST SPLIT
# ============================================================
print("\n4. Creating train/test split...")
n_total = coords.shape[0]
n_train = int(n_total * TRAIN_TEST_SPLIT)
indices = np.random.permutation(n_total)
train_idx = indices[:n_train]
test_idx = indices[n_train:]

coords_train = torch.from_numpy(coords[train_idx]).float().to(DEVICE)
targets_train = torch.from_numpy(targets_normalized[train_idx]).float().to(DEVICE)
coords_test = torch.from_numpy(coords[test_idx]).float().to(DEVICE)
targets_test = torch.from_numpy(targets_normalized[test_idx]).float().to(DEVICE)

print(f"Train set: {coords_train.shape[0]} samples")
print(f"Test set: {coords_test.shape[0]} samples")

# Create DataLoader for training
train_dataset = TensorDataset(coords_train, targets_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ============================================================
# 5. BUILD SIMPLE MLP TRUNK
# ============================================================
print("\n5. Building MLP Trunk network...")

class SimpleTrunk(nn.Module):
    """Simple MLP: (x,y,t) -> [18 basis values]"""
    def __init__(self, hidden_dim, output_dim, n_layers):
        super().__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(3, hidden_dim))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

trunk = SimpleTrunk(
    hidden_dim=TRUNK_HIDDEN_DIM,
    output_dim=N_MODES,
    n_layers=TRUNK_N_LAYERS
).to(DEVICE)

n_params = sum(p.numel() for p in trunk.parameters())
print(f"Network: {n_params:,} parameters")

# ============================================================
# 6. TRAINING SETUP
# ============================================================
print("\n6. Setting up training...")
criterion = nn.MSELoss()
optimizer = optim.Adam(trunk.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=50, verbose=True
)

train_losses = []
test_losses = []
best_test_loss = float('inf')
best_trunk_state = None

# ============================================================
# 7. TRAINING LOOP
# ============================================================
print("\n7. Training...")
print("=" * 70)

for epoch in range(N_EPOCHS):
    # Training phase
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
    
    # Test phase
    trunk.eval()
    with torch.no_grad():
        pred_test = trunk(coords_test)
        test_loss = criterion(pred_test, targets_test).item()
    test_losses.append(test_loss)
    
    # LR scheduling
    scheduler.step(test_loss)
    
    # Track best model
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_trunk_state = {k: v.cpu().clone() for k, v in trunk.state_dict().items()}
    
    # Logging
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:3d}/{N_EPOCHS} | Train Loss: {train_loss_epoch:.6e} | Test Loss: {test_loss:.6e}")

print("=" * 70)
print(f"Training complete. Best test loss: {best_test_loss:.6e}")

# Load best model
trunk.load_state_dict(best_trunk_state)
trunk.eval()

# ============================================================
# 8. SAVE TRAINED TRUNK
# ============================================================
print("\n8. Saving trained trunk...")
checkpoint = {
    'model_state_dict': best_trunk_state,
    'hyperparams': {
        'hidden_dim': TRUNK_HIDDEN_DIM,
        'n_layers': TRUNK_N_LAYERS,
        'n_modes': N_MODES,
        'learning_rate': LEARNING_RATE,
    },
    'normalization': {
        'min': targets_min.item() if isinstance(targets_min, np.ndarray) else targets_min,
        'max': targets_max.item() if isinstance(targets_max, np.ndarray) else targets_max,
    },
    'grid_info': {'Nx': Nx, 'Ny': Ny, 'Nt': Nt},
}
torch.save(checkpoint, os.path.join(SCRIPT_DIR, 'data/trunk_svd_simple.pth'))
print("✓ Saved to data/trunk_svd_simple.pth")

# ============================================================
# 9. VISUALIZATION
# ============================================================
print("\n9. Creating visualizations...")

# Plot training curves
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(train_losses, label='Train Loss', linewidth=2)
ax.semilogy(test_losses, label='Test Loss', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss (log scale)')
ax.set_title('Training Curves')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'data/trunk_training_curves.png'), dpi=150)
print("✓ Saved training curves to data/trunk_training_curves.png")

# Predict on all data and compare
with torch.no_grad():
    pred_all = trunk(torch.from_numpy(coords).float().to(DEVICE)).cpu().numpy()

# Denormalize predictions
pred_all_denorm = (pred_all + 1) / 2 * targets_range + targets_min

# Visualize first mode at mid-time
t_idx = Nt // 2
layer_size = Nx * Ny
start_idx = t_idx * layer_size
end_idx = start_idx + layer_size

true_mode_0 = targets_raw[start_idx:end_idx, 0].reshape(Nx, Ny, order='F')
pred_mode_0 = pred_all_denorm[start_idx:end_idx, 0].reshape(Nx, Ny, order='F')
error_mode_0 = np.abs(true_mode_0 - pred_mode_0)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

im0 = axes[0].imshow(true_mode_0, cmap='seismic', origin='lower')
axes[0].set_title(f'True Mode 0 (at t={t_idx})')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(pred_mode_0, cmap='seismic', origin='lower')
axes[1].set_title(f'Predicted Mode 0 (at t={t_idx})')
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(error_mode_0, cmap='hot', origin='lower')
axes[2].set_title(f'Absolute Error (max={error_mode_0.max():.2e})')
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'data/trunk_mode_0_comparison.png'), dpi=150)
print("✓ Saved mode comparison to data/trunk_mode_0_comparison.png")

# Test MSE per mode
test_mse_per_mode = np.mean((targets_raw[test_idx] - pred_all_denorm[test_idx])**2, axis=0)

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(range(N_MODES), test_mse_per_mode)
ax.set_xlabel('Mode Index')
ax.set_ylabel('Test MSE (denormalized)')
ax.set_title('Per-Mode Test Error')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'data/trunk_per_mode_error.png'), dpi=150)
print("✓ Saved per-mode error to data/trunk_per_mode_error.png")

# ============================================================
# 10. SUMMARY STATISTICS
# ============================================================
print("\n10. Summary Statistics")
print("=" * 70)
print(f"Final Train Loss: {train_losses[-1]:.6e}")
print(f"Final Test Loss:  {test_losses[-1]:.6e}")
print(f"Best Test Loss:   {best_test_loss:.6e}")
print(f"MSE per mode (denormalized): min={test_mse_per_mode.min():.6e}, max={test_mse_per_mode.max():.6e}")
print(f"Mean per-mode error: {test_mse_per_mode.mean():.6e}")
print("=" * 70)

print("\n✓ Training complete!")
