"""
Train DeepONet (trunk + branch) jointly in supervised fashion on MATLAB data.
- Inputs: IC sensors + (x,y,t)
- Output: u(x,y,t) from MATLAB simulations
- Uses random space-time sampling per sample for efficiency
"""

import os
from datetime import datetime
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# CONFIGURATION
# ============================================================
N_MODES = 18
N_SENSORS = 8
TRUNK_HIDDEN_DIM = 128
TRUNK_N_LAYERS = 4
BRANCH_HIDDEN_DIM = 128
BRANCH_N_LAYERS = 4
BATCH_SIZE = 2048
LEARNING_RATE = 1e-3
N_EPOCHS = 300
TRAIN_TEST_SPLIT = 0.8
POINTS_PER_SAMPLE = 5000  # space-time points sampled per sample
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 70)
print("TRAINING JOINT DEEPONET (SUPERVISED) ON MATLAB DATA")
print("=" * 70)
print(f"Device: {DEVICE}")

# ============================================================
# 1. LOAD MATLAB DATA
# ============================================================
print("\n1. Loading MATLAB data...")
matlab_path = os.path.join(SCRIPT_DIR, 'data/test_cases.mat')
if not os.path.exists(matlab_path):
    raise FileNotFoundError(f"MATLAB data not found: {matlab_path}")

# MATLAB v7.3 files use HDF5 format, need h5py
with h5py.File(matlab_path, 'r') as f:
    u_fom = np.array(f['U_data']).T  # (Nx, Ny, Nt, N_samples)

Nx, Ny, Nt, N_samples = u_fom.shape
n_space_time = Nx * Ny * Nt
print(f"u_fom shape: {u_fom.shape}")
print(f"Grid: Nx={Nx}, Ny={Ny}, Nt={Nt}, total points={n_space_time}")

# ============================================================
# 2. BUILD COORDINATES GRID
# ============================================================
print("\n2. Building coordinate grid...")
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
t = np.linspace(0, 1, Nt)
X, Y, T = np.meshgrid(x, y, t, indexing='ij')
coords = np.stack([X.flatten('F'), Y.flatten('F'), T.flatten('F')], axis=1)  # (Nx*Ny*Nt, 3)

# ============================================================
# 3. EXTRACT IC SENSORS FOR ALL SAMPLES
# ============================================================
print("\n3. Extracting IC sensors...")
sensor_x_indices = np.linspace(0, Nx - 1, N_SENSORS, dtype=int)
sensor_y_indices = np.linspace(0, Ny - 1, N_SENSORS, dtype=int)

ic_sensors = []
for s in range(N_samples):
    u_ic = u_fom[:, :, 0, s]
    sensors = []
    for si in sensor_x_indices:
        for sj in sensor_y_indices:
            sensors.append(u_ic[si, sj])
    ic_sensors.append(sensors)

ic_sensors = np.array(ic_sensors)  # (N_samples, N_SENSORS*N_SENSORS)
ic_dim = ic_sensors.shape[1]
print(f"IC sensors shape: {ic_sensors.shape}")

# Normalize IC sensors to [-1, 1]
ic_min = ic_sensors.min(axis=0, keepdims=True)
ic_max = ic_sensors.max(axis=0, keepdims=True)
ic_range = ic_max - ic_min
ic_norm = 2 * (ic_sensors - ic_min) / (ic_range + 1e-10) - 1

# ============================================================
# 4. BUILD TRAINING DATASET (SUBSAMPLE SPACE-TIME POINTS)
# ============================================================
print("\n4. Building training dataset...")
coords_list = []
ic_list = []
target_list = []

for s in range(N_samples):
    u_flat = u_fom[:, :, :, s].reshape(n_space_time, order='F')
    if POINTS_PER_SAMPLE >= n_space_time:
        idx = np.random.choice(n_space_time, size=POINTS_PER_SAMPLE, replace=True)
    else:
        idx = np.random.choice(n_space_time, size=POINTS_PER_SAMPLE, replace=False)

    coords_list.append(coords[idx])
    ic_list.append(np.repeat(ic_norm[s:s+1, :], len(idx), axis=0))
    target_list.append(u_flat[idx])

coords_all = np.concatenate(coords_list, axis=0)
ic_all = np.concatenate(ic_list, axis=0)
target_all = np.concatenate(target_list, axis=0)

print(f"Dataset size: {coords_all.shape[0]} samples")

# Normalize targets to [-1, 1]
u_min = target_all.min()
u_max = target_all.max()
u_range = u_max - u_min
target_norm = 2 * (target_all - u_min) / (u_range + 1e-10) - 1

# ============================================================
# 5. TRAIN/TEST SPLIT
# ============================================================
print("\n5. Train/test split...")
N_total = coords_all.shape[0]
indices = np.random.permutation(N_total)
train_size = int(N_total * TRAIN_TEST_SPLIT)
train_idx = indices[:train_size]
test_idx = indices[train_size:]

coords_train = torch.from_numpy(coords_all[train_idx]).float().to(DEVICE)
coords_test = torch.from_numpy(coords_all[test_idx]).float().to(DEVICE)

ic_train = torch.from_numpy(ic_all[train_idx]).float().to(DEVICE)
ic_test = torch.from_numpy(ic_all[test_idx]).float().to(DEVICE)

target_train = torch.from_numpy(target_norm[train_idx]).float().to(DEVICE)
target_test = torch.from_numpy(target_norm[test_idx]).float().to(DEVICE)

train_ds = TensorDataset(ic_train, coords_train, target_train)
test_ds = TensorDataset(ic_test, coords_test, target_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# 6. DEFINE MODELS
# ============================================================
print("\n6. Defining DeepONet models...")

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DeepONet(nn.Module):
    def __init__(self, trunk, branch):
        super().__init__()
        self.trunk = trunk
        self.branch = branch

    def forward(self, ic, coords):
        trunk_out = self.trunk(coords)  # (B, N_MODES)
        branch_out = self.branch(ic)    # (B, N_MODES)
        u = torch.sum(trunk_out * branch_out, dim=1)  # (B,)
        return u

trunk = MLP(3, TRUNK_HIDDEN_DIM, N_MODES, TRUNK_N_LAYERS).to(DEVICE)
branch = MLP(ic_dim, BRANCH_HIDDEN_DIM, N_MODES, BRANCH_N_LAYERS).to(DEVICE)
model = DeepONet(trunk, branch).to(DEVICE)

# ============================================================
# 7. TRAINING LOOP
# ============================================================
print("\n7. Training...")
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

train_losses = []
test_losses = []

for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss = 0.0
    for ic_b, coords_b, target_b in train_loader:
        optimizer.zero_grad()
        pred = model(ic_b, coords_b)
        loss = loss_fn(pred, target_b)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    train_loss = epoch_loss / len(train_loader)
    train_losses.append(train_loss)

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_loss_val = 0.0
        for ic_b, coords_b, target_b in test_loader:
            pred = model(ic_b, coords_b)
            loss = loss_fn(pred, target_b)
            test_loss_val += loss.item()
        test_loss = test_loss_val / len(test_loader)
        test_losses.append(test_loss)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:4d}/{N_EPOCHS} | Train Loss={train_loss:.6e} | Test Loss={test_loss:.6e}")

# ============================================================
# 8. SAVE CHECKPOINT
# ============================================================
print("\n8. Saving checkpoint...")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
ckpt_path = os.path.join(SCRIPT_DIR, f"data/deeponet_joint_supervised_{timestamp}.pth")

checkpoint = {
    'model_state_dict': model.state_dict(),
    'config': {
        'n_modes': N_MODES,
        'n_sensors': N_SENSORS,
        'trunk_hidden_dim': TRUNK_HIDDEN_DIM,
        'trunk_n_layers': TRUNK_N_LAYERS,
        'branch_hidden_dim': BRANCH_HIDDEN_DIM,
        'branch_n_layers': BRANCH_N_LAYERS,
        'points_per_sample': POINTS_PER_SAMPLE,
    },
    'normalization': {
        'ic_min': ic_min,
        'ic_max': ic_max,
        'ic_range': ic_range,
        'u_min': u_min,
        'u_max': u_max,
        'u_range': u_range,
    },
    'train_loss': train_losses,
    'test_loss': test_losses,
}

torch.save(checkpoint, ckpt_path)
print(f"✓ Checkpoint saved to {ckpt_path}")

# ============================================================
# 9. QUICK EVAL ON SAMPLE 0 AT 5 TIMES
# ============================================================
print("\n9. Evaluating sample 0 at 5 time instants...")

model.eval()
with torch.no_grad():
    ic0 = ic_norm[0:1, :]  # (1, ic_dim)
    ic0_tensor = torch.from_numpy(ic0).float().to(DEVICE)
    
    time_instants = [0.0, 0.25, 0.5, 0.75, 1.0]
    time_indices = [int(ti * (Nt - 1)) for ti in time_instants]
    
    preds = []
    gts = []
    for t_val, t_idx in zip(time_instants, time_indices):
        coords_t = np.stack([X[:, :, t_idx].flatten('F'),
                             Y[:, :, t_idx].flatten('F'),
                             np.full((Nx * Ny,), t_val)], axis=1)
        coords_tensor = torch.from_numpy(coords_t).float().to(DEVICE)
        ic_batch = ic0_tensor.repeat(coords_tensor.shape[0], 1)
        
        u_pred_norm = model(ic_batch, coords_tensor).cpu().numpy()
        u_pred = (u_pred_norm + 1) * u_range / 2 + u_min
        
        u_gt = u_fom[:, :, t_idx, 0].flatten('F')
        preds.append(u_pred.reshape(Nx, Ny, order='F'))
        gts.append(u_gt.reshape(Nx, Ny, order='F'))

# Plot GT vs Pred vs Error
fig, axes = plt.subplots(len(time_instants), 3, figsize=(15, 4 * len(time_instants)))
for i, t_val in enumerate(time_instants):
    gt = gts[i]
    pred = preds[i]
    err = np.abs(pred - gt)
    vmin = min(gt.min(), pred.min())
    vmax = max(gt.max(), pred.max())

    im0 = axes[i, 0].imshow(gt, cmap='seismic', origin='lower', vmin=vmin, vmax=vmax)
    axes[i, 0].set_title(f"GT (t={t_val:.2f})")
    axes[i, 0].set_xticks([])
    axes[i, 0].set_yticks([])
    plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)

    im1 = axes[i, 1].imshow(pred, cmap='seismic', origin='lower', vmin=vmin, vmax=vmax)
    axes[i, 1].set_title(f"Pred (t={t_val:.2f})")
    axes[i, 1].set_xticks([])
    axes[i, 1].set_yticks([])
    plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)

    im2 = axes[i, 2].imshow(err, cmap='hot', origin='lower')
    axes[i, 2].set_title(f"Abs Error (t={t_val:.2f})")
    axes[i, 2].set_xticks([])
    axes[i, 2].set_yticks([])
    plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)

plt.tight_layout()
plot_path = os.path.join(SCRIPT_DIR, f"data/deeponet_joint_eval_sample0_{timestamp}.png")
plt.savefig(plot_path, dpi=150)
print(f"✓ Evaluation plot saved to {plot_path}")
plt.close()

# ============================================================
# 10. LOSS CURVES
# ============================================================
plt.figure(figsize=(8, 5))
plt.semilogy(train_losses, label='Train')
plt.semilogy(test_losses, label='Test')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('DeepONet Joint Training Loss')
plt.legend()
plt.grid(True, alpha=0.3)
loss_path = os.path.join(SCRIPT_DIR, f"data/deeponet_joint_training_loss_{timestamp}.png")
plt.savefig(loss_path, dpi=150)
print(f"✓ Loss plot saved to {loss_path}")
plt.close()

print("\n" + "=" * 70)
print("✓ TRAINING COMPLETE")
print("=" * 70)
