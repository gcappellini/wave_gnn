"""
Branch Network Supervised Training (Isolated Test)
- Input: Sensor measurements of IC (initial condition) at t=0
- Output: SVD mode coefficients (from VT matrix)
- Task: Learn mapping IC_sensors → VT coefficients
- No trunk network - pure regression on SVD coefficients
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
N_MODES = 18  # First 18 modes (matching trunk network)
N_SENSORS = 8  # 8x8 sensor grid
BRANCH_HIDDEN_DIM = 128
BRANCH_N_LAYERS = 4
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
N_EPOCHS = 500
TRAIN_TEST_SPLIT = 0.8  # Use 80% for training, 20% for testing
USE_ALL_SAMPLES = True  # Use ALL available samples to avoid overfitting
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")
print(f"Config: {N_MODES} modes, {N_SENSORS}x{N_SENSORS} sensors")
print(f"Training strategy: Use ALL available samples ({int(TRAIN_TEST_SPLIT*100)}% train, {int((1-TRAIN_TEST_SPLIT)*100)}% test)")
print(f"  → More data = better generalization, less overfitting")

# ============================================================
# 1. LOAD SVD DATA
# ============================================================
print("\n1. Loading SVD data...")
svd_data = np.load(os.path.join(SCRIPT_DIR, 'data/svd_basis_data.npy'), allow_pickle=True).item()
U_basis = svd_data['basis']  # (Nx*Ny*Nt, p)
Sigma = svd_data['singular_values']
VT = svd_data['coefficients']  # (p, N_samples) ← Ground truth coefficients!
grid_info = svd_data['grid_info']

Nx, Ny, Nt = grid_info.astype(int)
N_samples = VT.shape[1]

print(f"Grid: {Nx} x {Ny} x {Nt}")
print(f"SVD basis: {U_basis.shape}")
print(f"Available modes: {U_basis.shape[1]}")
print(f"Coefficients (VT): {VT.shape}")
print(f"Number of available samples: {N_samples}")

# Use all samples for training
if USE_ALL_SAMPLES:
    n_train_samples = N_samples
    print(f"✓ Using ALL {n_train_samples} samples (80% train, 20% test split)")

# Extract first N_MODES
U_basis_truncated = U_basis[:, :N_MODES]
VT_truncated = VT[:N_MODES, :]  # (N_MODES, N_samples)

print(f"\nUsing first {N_MODES} modes")
print(f"Truncated basis: {U_basis_truncated.shape}")
print(f"Truncated coefficients: {VT_truncated.shape} (modes × samples)")

# ============================================================
# 2. EXTRACT SENSOR IC FOR EACH SAMPLE
# ============================================================
print("\n2. Extracting sensor IC from all samples...")
n_spatial = Nx * Ny

# Sensor locations
sensor_x_indices = np.linspace(0, Nx-1, N_SENSORS, dtype=int)
sensor_y_indices = np.linspace(0, Ny-1, N_SENSORS, dtype=int)

# For each sample, extract IC at t=0 at sensor locations
ic_sensors_all_samples = []

for sample_idx in range(N_samples):
    # Get full spatial field at t=0 for this sample (reconstructed from SVD)
    # u_sample = U_basis @ (Sigma * VT[:, sample])
    reconstructed = U_basis_truncated @ (Sigma[:N_MODES] * VT_truncated[:N_MODES, sample_idx])
    u_spatial_t0 = reconstructed[:n_spatial]  # First Nx*Ny points are t=0
    u_spatial_t0_grid = u_spatial_t0.reshape(Nx, Ny, order='F')
    
    # Extract at sensor locations
    ic_sensors = []
    for si in sensor_x_indices:
        for sj in sensor_y_indices:
            ic_sensors.append(u_spatial_t0_grid[si, sj])
    
    ic_sensors_all_samples.append(ic_sensors)

ic_sensors_array = np.array(ic_sensors_all_samples)  # (N_samples, N_SENSORS*N_SENSORS)
print(f"IC sensors shape: {ic_sensors_array.shape}")
print(f"IC sensors range: [{ic_sensors_array.min():.6e}, {ic_sensors_array.max():.6e}]")

# ============================================================
# 3. NORMALIZE DATA
# ============================================================
print("\n3. Normalizing data...")

# Normalize IC sensors
ic_min = ic_sensors_array.min(axis=0, keepdims=True)
ic_max = ic_sensors_array.max(axis=0, keepdims=True)
ic_range = ic_max - ic_min
ic_normalized = 2 * (ic_sensors_array - ic_min) / (ic_range + 1e-10) - 1

# Normalize coefficients
coeff_min = VT_truncated.min()
coeff_max = VT_truncated.max()
coeff_range = coeff_max - coeff_min
coeff_normalized = 2 * (VT_truncated.T - coeff_min) / coeff_range - 1  # Transpose to (N_samples, N_MODES)

print(f"IC normalized range: [{ic_normalized.min():.6f}, {ic_normalized.max():.6f}]")
print(f"Coefficients normalized range: [{coeff_normalized.min():.6f}, {coeff_normalized.max():.6f}]")

# Store normalization for later denormalization
normalization = {
    'ic_min': ic_min,
    'ic_max': ic_max,
    'ic_range': ic_range,
    'coeff_min': coeff_min,
    'coeff_max': coeff_max,
    'coeff_range': coeff_range,
}

# ============================================================
# 4. TRAIN/TEST SPLIT
# ============================================================
print("\n4. Creating train/test split...")
n_total = N_samples
n_train = int(n_total * TRAIN_TEST_SPLIT)
indices = np.random.permutation(n_total)
train_idx = indices[:n_train]
test_idx = indices[n_train:]

ic_train = torch.from_numpy(ic_normalized[train_idx]).float().to(DEVICE)
coeff_train = torch.from_numpy(coeff_normalized[train_idx]).float().to(DEVICE)
ic_test = torch.from_numpy(ic_normalized[test_idx]).float().to(DEVICE)
coeff_test = torch.from_numpy(coeff_normalized[test_idx]).float().to(DEVICE)

print(f"Train set: {ic_train.shape[0]} samples ({int(TRAIN_TEST_SPLIT*100)}%)")
print(f"Test set: {ic_test.shape[0]} samples ({int((1-TRAIN_TEST_SPLIT)*100)}%)")
print(f"Total data used: {ic_train.shape[0] + ic_test.shape[0]} / {N_samples} (✓ Using all available)")

# Create DataLoader
train_dataset = TensorDataset(ic_train, coeff_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Batch size: {BATCH_SIZE}, ~{len(train_loader)} batches per epoch")

# ============================================================
# 5. BUILD BRANCH NETWORK
# ============================================================
print("\n5. Building Branch network...")

class BranchNetwork(nn.Module):
    """MLP: sensor IC measurements -> SVD mode coefficients"""
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
        
        # Output layer (linear)
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, ic):
        return self.net(ic)

ic_input_dim = N_SENSORS * N_SENSORS
branch_model = BranchNetwork(
    ic_dim=ic_input_dim,
    hidden_dim=BRANCH_HIDDEN_DIM,
    output_dim=N_MODES,
    n_layers=BRANCH_N_LAYERS
).to(DEVICE)

print(f"Branch input dim: {ic_input_dim} ({N_SENSORS}x{N_SENSORS} sensors)")
print(f"Branch output dim: {N_MODES}")
print(f"Branch parameters: {sum(p.numel() for p in branch_model.parameters())}")

# ============================================================
# 6. TRAINING LOOP
# ============================================================
print("\n6. Starting training loop...")

optimizer = optim.Adam(branch_model.parameters(), lr=LEARNING_RATE)
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
    branch_model.train()
    epoch_loss = 0.0
    for ic_batch, coeff_batch in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        pred_coeff = branch_model(ic_batch)  # (batch_size, N_MODES)
        
        # Loss: MSE between predicted and true SVD coefficients
        loss = loss_fn(pred_coeff, coeff_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    train_loss = epoch_loss / len(train_loader)
    train_losses.append(train_loss)
    
    # Test step
    branch_model.eval()
    with torch.no_grad():
        pred_coeff_test = branch_model(ic_test)
        test_loss = loss_fn(pred_coeff_test, coeff_test)
        test_losses.append(test_loss.item())
    
    # Learning rate scheduling
    scheduler.step(test_loss)
    
    # Early stopping
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        patience_counter = 0
        best_epoch = epoch
        best_state = branch_model.state_dict()
    else:
        patience_counter += 1
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:3d}/{N_EPOCHS}: "
              f"Train Loss={train_loss:.6e}, Test Loss={test_loss:.6e}")
    
    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch+1} (best: {best_epoch+1})")
        break

# ============================================================
# 7. SAVE CHECKPOINT
# ============================================================
print("\n7. Saving checkpoint...")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_path = os.path.join(SCRIPT_DIR, f'data/branch_supervised_{timestamp}.pth')

checkpoint = {
    'epoch': best_epoch,
    'model_state_dict': branch_model.state_dict(),
    'config': {
        'ic_dim': ic_input_dim,
        'hidden_dim': BRANCH_HIDDEN_DIM,
        'output_dim': N_MODES,
        'n_layers': BRANCH_N_LAYERS,
        'n_sensors': N_SENSORS,
    },
    'normalization': normalization,
    'train_loss': train_losses,
    'test_loss': test_losses,
}

torch.save(checkpoint, checkpoint_path)
print(f"✓ Checkpoint saved to {checkpoint_path}")

# ============================================================
# 8. PREDICT ON SAMPLE 0 (FIRST SAMPLE)
# ============================================================
print("\n8. Predicting on sample 0...")

branch_model.eval()
with torch.no_grad():
    # Get sample 0 IC (normalized)
    sample_0_ic = torch.from_numpy(ic_normalized[0:1]).float().to(DEVICE)  # (1, ic_dim)
    sample_0_coeff_gt = VT_truncated[:, 0]  # Ground truth coefficients (unnormalized)
    
    # Predict
    sample_0_pred = branch_model(sample_0_ic)  # (1, N_MODES), normalized
    
    # Denormalize prediction
    sample_0_pred_denorm = (sample_0_pred.cpu().numpy() + 1) * coeff_range / 2 + coeff_min
    sample_0_pred_denorm = sample_0_pred_denorm.squeeze()  # (N_MODES,)
    
    # Save predictions
    sample_0_results = {
        'predicted_coefficients': sample_0_pred_denorm,
        'ground_truth_coefficients': sample_0_coeff_gt,
        'absolute_error': np.abs(sample_0_pred_denorm - sample_0_coeff_gt),
        'relative_error': np.abs(sample_0_pred_denorm - sample_0_coeff_gt) / (np.abs(sample_0_coeff_gt) + 1e-10),
    }
    
    sample_0_path = os.path.join(SCRIPT_DIR, f'data/branch_sample0_predictions_{timestamp}.npz')
    np.savez(sample_0_path, **sample_0_results)
    print(f"✓ Sample 0 predictions saved to {sample_0_path}")
    
    print(f"\nSample 0 coefficient comparison:")
    print(f"{'Mode':<6} {'Ground Truth':<15} {'Predicted':<15} {'Abs Error':<15} {'Rel Error':<12}")
    print("-" * 70)
    for i in range(N_MODES):
        print(f"{i:<6} {sample_0_coeff_gt[i]:<15.6e} {sample_0_pred_denorm[i]:<15.6e} "
              f"{sample_0_results['absolute_error'][i]:<15.6e} {sample_0_results['relative_error'][i]:<12.4f}")

# ============================================================
# 9. EVALUATION & VISUALIZATION (TEST SET)
# ============================================================
print("\n9. Evaluating on test set...")

branch_model.eval()
with torch.no_grad():
    pred_coeff_test_denorm = (pred_coeff_test.cpu().numpy() + 1) * coeff_range / 2 + coeff_min
    coeff_test_denorm = (coeff_test.cpu().numpy() + 1) * coeff_range / 2 + coeff_min

# Compute errors
abs_error = np.abs(coeff_test_denorm - pred_coeff_test_denorm)
rel_error = abs_error / (np.abs(coeff_test_denorm) + 1e-10)

print(f"Absolute error (mean): {abs_error.mean():.6e}")
print(f"Absolute error (std): {abs_error.std():.6e}")
print(f"Relative error (mean): {rel_error.mean():.6f}")
print(f"Per-mode MAE:")
for i in range(N_MODES):
    print(f"  Mode {i:2d}: {abs_error[:, i].mean():.6e}")

# ============================================================
# 10. PLOTS
# ============================================================
print("\n9. Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Training curves
ax1 = axes[0, 0]
ax1.semilogy(train_losses, label='Train', linewidth=2)
ax1.semilogy(test_losses, label='Test', linewidth=2)
ax1.axvline(best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best: {best_epoch}')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE Loss')
ax1.set_title('Training Curves')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Prediction vs Ground Truth for first test sample
ax2 = axes[0, 1]
sample_idx = 0
ax2.plot(coeff_test_denorm[sample_idx], 'o-', label='Ground Truth', linewidth=2, markersize=6)
ax2.plot(pred_coeff_test_denorm[sample_idx], 's--', label='Predicted', linewidth=2, markersize=6)
ax2.set_xlabel('Mode Index')
ax2.set_ylabel('Coefficient Value')
ax2.set_title(f'Prediction vs GT (Test Sample {sample_idx})')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Per-mode absolute error
ax3 = axes[1, 0]
mode_mae = abs_error.mean(axis=0)
ax3.bar(range(N_MODES), mode_mae, color='steelblue')
ax3.set_xlabel('Mode Index')
ax3.set_ylabel('Mean Absolute Error')
ax3.set_title('Per-Mode Error')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Error distribution
ax4 = axes[1, 1]
ax4.hist(abs_error.flatten(), bins=30, color='coral', alpha=0.7, edgecolor='black')
ax4.set_xlabel('Absolute Error')
ax4.set_ylabel('Frequency')
ax4.set_title(f'Error Distribution (mean={abs_error.mean():.6e})')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_path = os.path.join(SCRIPT_DIR, 'branch_supervised_analysis.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Analysis saved to {output_path}")
plt.close()

print("\n" + "="*60)
print("Branch Network Supervised Training Completed!")
print("="*60)
print(f"\nSummary:")
print(f"  ✓ Input: {N_SENSORS}x{N_SENSORS} sensor IC measurements")
print(f"  ✓ Output: {N_MODES} SVD mode coefficients")
print(f"  ✓ Test MSE: {best_test_loss:.6e}")
print(f"  ✓ Test MAE: {abs_error.mean():.6e}")
print(f"\nNext steps:")
print(f"  1. Check if errors are reasonable")
print(f"  2. If good: integrate with trunk network (full DeepONet)")
print(f"  3. If bad: increase branch capacity or adjust training")
print("="*60)
