"""
DeepONet: Supervised Learning with Frozen Trunk + Trainable Branch
- Trunk network: Pre-trained MLP mapping (x,y,t) -> [18 SVD basis values]
- Branch network: MLP mapping initial condition field -> [18 branch coefficients]
- Forward: sum_i T_i(x,y,t) * B_i(IC) for each spatial point
- Loss: MSE on reconstructed solution vs ground truth
- Supervision: All time steps (no restriction to certain time windows)
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
N_SENSORS = 8  # Localized sensors: N_SENSORS x N_SENSORS grid for IC input
BRANCH_HIDDEN_DIM = 128
BRANCH_N_LAYERS = 4
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
N_EPOCHS = 500
TRAIN_TEST_SPLIT = 0.8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")
print(f"Config: {N_MODES} modes, {N_SENSORS}x{N_SENSORS} sensors, branch_hidden_dim={BRANCH_HIDDEN_DIM}, n_layers={BRANCH_N_LAYERS}")

# ============================================================
# 1. LOAD SVD DATA AND GRID INFO
# ============================================================
print("\n1. Loading SVD data...")
svd_data = np.load(os.path.join(SCRIPT_DIR, 'data/svd_basis_data.npy'), allow_pickle=True).item()
U_basis = svd_data['basis']  # (Nx*Ny*Nt, p)
Sigma = svd_data['singular_values']
grid_info = svd_data['grid_info']

Nx, Ny, Nt = grid_info.astype(int)
n_space_time = U_basis.shape[0]
print(f"Grid: {Nx} x {Ny} x {Nt} = {n_space_time} space-time points")
print(f"SVD basis shape: {U_basis.shape}")
print(f"Using first {N_MODES} modes for DeepONet")

# Extract first N_MODES
U_basis_truncated = U_basis[:, :N_MODES]  # (n_space_time, N_MODES)

# ============================================================
# 2. GENERATE COORDINATES (x, y, t)
# ============================================================
print("\n2. Generating coordinates...")
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
t = np.linspace(0, 1, Nt)

# Create meshgrid with ij indexing, flatten with Fortran order
X, Y, T = np.meshgrid(x, y, t, indexing='ij')
coords = np.stack([X.flatten('F'), Y.flatten('F'), T.flatten('F')], axis=1)
print(f"Coordinates shape: {coords.shape}")

# ============================================================
# 3. EXTRACT INITIAL CONDITIONS AT SENSOR LOCATIONS (t=0)
# ============================================================
print("\n3. Extracting initial conditions at sensor locations...")
n_spatial = Nx * Ny

# Sensor locations: uniform grid with N_SENSORS x N_SENSORS points
# Map sensor indices to full spatial grid indices (Fortran order)
sensor_x_indices = np.linspace(0, Nx-1, N_SENSORS, dtype=int)
sensor_y_indices = np.linspace(0, Ny-1, N_SENSORS, dtype=int)

# Extract sensor IC: get raw field values (reconstructed from SVD) at sensor locations
# We reconstruct: u(x,y,0) = sum_i a_i(0) * U_i(x,y)
# At t=0, the coefficients are: a_i(0) = U_basis_truncated[spatial_idx, i]
# We reconstruct the full field and then sample at sensor locations

# Full field reconstruction at t=0 (Nx x Ny grid)
ic_full_field = np.zeros((Nx, Ny))
for mode_idx in range(N_MODES):
    # Get spatial field for this mode at t=0
    mode_spatial_field = U_basis_truncated[:n_spatial, mode_idx].reshape((Nx, Ny), order='F')
    ic_full_field += mode_spatial_field

# Extract values at sensor locations
ic_at_sensors = []  # (N_SENSORS, N_SENSORS) array of raw field values

for si in sensor_x_indices:
    row = []
    for sj in sensor_y_indices:
        row.append(ic_full_field[si, sj])
    ic_at_sensors.append(row)

ic_field_spatial = np.array(ic_at_sensors)  # (N_SENSORS, N_SENSORS)
ic_field_flattened = ic_field_spatial.flatten()  # (N_SENSORS*N_SENSORS,)

print(f"IC shape (spatial grid of sensors): {ic_field_spatial.shape}")
print(f"IC shape (flattened): {ic_field_flattened.shape}")
print(f"IC range: [{ic_field_spatial.min():.6e}, {ic_field_spatial.max():.6e}]")

# ============================================================
# 4. LOAD PRETRAINED TRUNK NETWORK
# ============================================================
print("\n4. Loading pretrained trunk network...")

class SimpleTrunk(nn.Module):
    """Simple MLP: (x,y,t) -> [N_MODES basis values]"""
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

trunk_model = SimpleTrunk(
    hidden_dim=128,
    output_dim=N_MODES,
    n_layers=4
).to(DEVICE)

# Load checkpoint
trunk_checkpoint_path = os.path.join(SCRIPT_DIR, 'data/trunk_svd_simple.pth')
if os.path.exists(trunk_checkpoint_path):
    checkpoint = torch.load(trunk_checkpoint_path, map_location=DEVICE)
    trunk_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded trunk model from {trunk_checkpoint_path}")
else:
    raise FileNotFoundError(f"Trunk checkpoint not found at {trunk_checkpoint_path}")

# Freeze trunk weights
for param in trunk_model.parameters():
    param.requires_grad = False
print("✓ Trunk weights frozen")

# ============================================================
# 5. BUILD BRANCH NETWORK
# ============================================================
print("\n5. Building Branch network...")

class BranchNetwork(nn.Module):
    """MLP: initial condition field -> [N_MODES coefficients]"""
    def __init__(self, ic_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(ic_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        # Output layer (linear, can be positive or negative)
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, ic):
        return self.net(ic)

# IC input dimension: flattened sensor measurements (N_SENSORS*N_SENSORS raw field values)
ic_input_dim = N_SENSORS * N_SENSORS
branch_model = BranchNetwork(
    ic_dim=ic_input_dim,
    hidden_dim=BRANCH_HIDDEN_DIM,
    output_dim=N_MODES,
    n_layers=BRANCH_N_LAYERS
).to(DEVICE)

print(f"Branch input dim (sensor measurements): {ic_input_dim} ({N_SENSORS}x{N_SENSORS})")
print(f"Branch output dim: {N_MODES}")
print(f"Branch parameters: {sum(p.numel() for p in branch_model.parameters())}")

# ============================================================
# 6. PREPARE TRAINING DATA FOR DEEPONET
# ============================================================
print("\n6. Preparing DeepONet training data...")

# For supervised learning, we need pairs of (IC sensor measurements, full solution field)
# IC: raw field values at t=0 at sensor locations (used by branch to infer SVD coefficients)
# Full solution: all spatial points at all time steps (targets)

# Replicate sensor IC for each training sample
ic_batch = np.tile(ic_field_flattened, (n_space_time, 1))  # (n_space_time, N_SENSORS*N_SENSORS)

print(f"IC batch shape (sensor measurements): {ic_batch.shape}")

# Targets: SVD coefficients at all space-time points
targets_raw = U_basis_truncated  # (n_space_time, N_MODES)

# Store normalization for denormalization later
targets_min = targets_raw.min()
targets_max = targets_raw.max()
targets_range = targets_max - targets_min

targets_normalized = 2 * (targets_raw - targets_min) / targets_range - 1
print(f"Targets shape: {targets_normalized.shape}")
print(f"Targets range (normalized): [{targets_normalized.min():.6f}, {targets_normalized.max():.6f}]")

# Save normalization constants for later
normalization = {
    'targets_min': targets_min,
    'targets_max': targets_max,
    'targets_range': targets_range
}

# ============================================================
# 7. TRAIN/TEST SPLIT
# ============================================================
print("\n7. Creating train/test split...")
n_total = n_space_time
n_train = int(n_total * TRAIN_TEST_SPLIT)
indices = np.random.permutation(n_total)
train_idx = indices[:n_train]
test_idx = indices[n_train:]

ic_train = torch.from_numpy(ic_batch[train_idx]).float().to(DEVICE)
targets_train = torch.from_numpy(targets_normalized[train_idx]).float().to(DEVICE)
coords_test = torch.from_numpy(coords[test_idx]).float().to(DEVICE)
ic_test = torch.from_numpy(ic_batch[test_idx]).float().to(DEVICE)
targets_test = torch.from_numpy(targets_normalized[test_idx]).float().to(DEVICE)

print(f"Train set: {ic_train.shape[0]} samples")
print(f"Test set: {ic_test.shape[0]} samples")

# Create DataLoader
coords_train = torch.from_numpy(coords[train_idx]).float().to(DEVICE)
train_dataset = TensorDataset(ic_train, coords_train, targets_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Batch size: {BATCH_SIZE}, ~{len(train_loader)} batches per epoch")

# ============================================================
# 8. DEEPONET MODEL (Branch + Trunk)
# ============================================================
print("\n8. Building DeepONet model...")

class DeepONet(nn.Module):
    """DeepONet: Branch network + frozen Trunk network
    
    For supervised learning, we want to predict SVD coefficients directly.
    Branch(IC) outputs N_MODES coefficients
    Trunk(x,y,t) outputs N_MODES basis values
    The interaction term: branch * trunk element-wise gives predictions for each mode
    """
    def __init__(self, branch, trunk):
        super().__init__()
        self.branch = branch
        self.trunk = trunk
    
    def forward(self, ic, coords):
        """
        Args:
            ic: Initial conditions (batch_size, ic_dim)
            coords: Space-time coordinates (batch_size, 3)
        
        Returns:
            predictions: SVD coefficients (batch_size, N_MODES)
        """
        # Branch: IC -> [N_MODES] branch coefficients
        branch_out = self.branch(ic)  # (batch_size, N_MODES)
        
        # Trunk: coords -> [N_MODES] basis values
        trunk_out = self.trunk(coords)  # (batch_size, N_MODES)
        
        # Element-wise product (Hadamard product)
        # This gives the interaction between branch and trunk
        predictions = branch_out * trunk_out  # (batch_size, N_MODES)
        
        return predictions

deeponet = DeepONet(branch_model, trunk_model).to(DEVICE)

print("✓ DeepONet model ready (branch trainable, trunk frozen)")

# ============================================================
# 9. TRAINING LOOP
# ============================================================
print("\n9. Starting training loop...")

optimizer = optim.Adam(deeponet.branch.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=20, verbose=False
)
loss_fn = nn.MSELoss()

train_losses = []
test_losses = []
best_test_loss = float('inf')
patience = 50
patience_counter = 0

for epoch in range(N_EPOCHS):
    # Training step
    deeponet.train()
    epoch_loss = 0.0
    for ic_batch, coords_batch, targets_batch in train_loader:
        optimizer.zero_grad()
        
        # Forward pass through DeepONet
        pred = deeponet(ic_batch, coords_batch)  # (batch_size, N_MODES)
        
        # Loss: MSE between predicted and true SVD coefficients
        loss = loss_fn(pred, targets_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    train_loss = epoch_loss / len(train_loader)
    train_losses.append(train_loss)
    
    # Test step
    deeponet.eval()
    with torch.no_grad():
        pred_test = deeponet(ic_test, coords_test)
        test_loss = loss_fn(pred_test, targets_test)
        test_losses.append(test_loss.item())
    
    # Learning rate scheduling
    scheduler.step(test_loss)
    
    # Early stopping
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        patience_counter = 0
        best_epoch = epoch
        best_state = deeponet.state_dict()
    else:
        patience_counter += 1
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:3d}/{N_EPOCHS}: "
              f"Train Loss={train_loss:.6e}, Test Loss={test_loss:.6e}")
    
    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch+1} (best: {best_epoch+1})")
        break

# ============================================================
# 10. SAVE CHECKPOINT
# ============================================================
print("\n10. Saving checkpoint...")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_path = os.path.join(SCRIPT_DIR, f'data/deeponet_supervised_{timestamp}.pth')

checkpoint = {
    'epoch': best_epoch,
    'branch_state_dict': deeponet.branch.state_dict(),
    'trunk_state_dict': trunk_model.state_dict(),
    'branch_config': {
        'ic_dim': ic_input_dim,
        'hidden_dim': BRANCH_HIDDEN_DIM,
        'output_dim': N_MODES,
        'n_layers': BRANCH_N_LAYERS
    },
    'trunk_config': {
        'hidden_dim': 128,
        'output_dim': N_MODES,
        'n_layers': 4
    },
    'normalization': normalization,
    'grid_info': grid_info,
    'train_loss': train_losses,
    'test_loss': test_losses,
}

torch.save(checkpoint, checkpoint_path)
print(f"✓ Checkpoint saved to {checkpoint_path}")

# ============================================================
# 11. VISUALIZATION
# ============================================================
print("\n11. Generating training curves...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Training and test loss
axes[0].semilogy(train_losses, label='Train', linewidth=2)
axes[0].semilogy(test_losses, label='Test', linewidth=2)
axes[0].axvline(best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best: {best_epoch}')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('DeepONet Training Curves')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Learning rate (if tracked)
axes[1].text(0.5, 0.5, 'DeepONet with Frozen Trunk\n\n' +
             f'Branch Hidden Dim: {BRANCH_HIDDEN_DIM}\n' +
             f'Branch Layers: {BRANCH_N_LAYERS}\n' +
             f'Best Test Loss: {best_test_loss:.6e}\n' +
             f'Final Epoch: {best_epoch+1}',
             ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1].axis('off')

plt.tight_layout()
output_path = os.path.join(SCRIPT_DIR, 'deeponet_training_curves.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Training curves saved to {output_path}")
plt.close()

print("\n" + "="*60)
print("DeepONet supervised training completed!")
print("="*60)
