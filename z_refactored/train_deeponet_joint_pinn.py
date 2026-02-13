"""
Train DeepONet (trunk + branch) jointly with Physics-Informed loss.
- Supervise only initial condition (t=0) from MATLAB data
- Enforce PDE residual in the interior (no source)
PDE: u_tt + k*u_t - c^2*(u_xx + u_yy) = 0
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
BATCH_SIZE = 1 #2048
LEARNING_RATE = 1e-3
N_EPOCHS = 300
TRAIN_TEST_SPLIT = 0.9

# PINN settings
POINTS_PER_SAMPLE_IC = 2000  # spatial points at t=0 per sample
N_COLLOC_PER_EPOCH = 50000    # random collocation points per epoch
IC_LOSS_WEIGHT = 1.0
PDE_LOSS_WEIGHT = 1.0

# PDE parameters (match MATLAB if known)
WAVE_SPEED = 1.0
DAMPING_COEFF = 1.0

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 70)
print("FINE-TUNING DEEPONET WITH PINN LOSS FROM SUPERVISED CHECKPOINT")
print("=" * 70)
print(f"Device: {DEVICE}")

# ============================================================
# 1. LOAD MATLAB DATA (IC only)
# ============================================================
print("\n1. Loading MATLAB data...")
matlab_path = os.path.join(SCRIPT_DIR, 'data/test_cases.mat')
if not os.path.exists(matlab_path):
    raise FileNotFoundError(f"MATLAB data not found: {matlab_path}")

with h5py.File(matlab_path, 'r') as f:
    u_fom = np.array(f['U_data']).T  # (Nx, Ny, Nt, N_samples)

Nx, Ny, Nt, N_samples = u_fom.shape
n_spatial = Nx * Ny
print(f"u_fom shape: {u_fom.shape}")

# ============================================================
# 2. BUILD COORDINATES GRID
# ============================================================
print("\n2. Building coordinate grid...")
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
t = np.linspace(0, 1, Nt)
X, Y = np.meshgrid(x, y, indexing='ij')
coords_spatial = np.stack([X.flatten('F'), Y.flatten('F')], axis=1)  # (Nx*Ny, 2)

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
# 4. BUILD IC DATASET (t=0 supervision)
# ============================================================
print("\n4. Building IC dataset...")
coords_ic_list = []
ic_list = []
target_list = []

for s in range(N_samples):
    u_ic_full = u_fom[:, :, 0, s].reshape(n_spatial, order='F')
    if POINTS_PER_SAMPLE_IC >= n_spatial:
        idx = np.random.choice(n_spatial, size=POINTS_PER_SAMPLE_IC, replace=True)
    else:
        idx = np.random.choice(n_spatial, size=POINTS_PER_SAMPLE_IC, replace=False)

    coords_xy = coords_spatial[idx]
    coords_t0 = np.hstack([coords_xy, np.zeros((coords_xy.shape[0], 1))])  # t=0

    coords_ic_list.append(coords_t0)
    ic_list.append(np.repeat(ic_norm[s:s+1, :], len(idx), axis=0))
    target_list.append(u_ic_full[idx])

coords_ic = np.concatenate(coords_ic_list, axis=0)
ic_ic = np.concatenate(ic_list, axis=0)
target_ic = np.concatenate(target_list, axis=0)

print(f"IC dataset size: {coords_ic.shape[0]}")

# Train/test split for IC supervision
N_total = coords_ic.shape[0]
indices = np.random.permutation(N_total)
train_size = int(N_total * TRAIN_TEST_SPLIT)
train_idx = indices[:train_size]
test_idx = indices[train_size:]

coords_ic_train = torch.from_numpy(coords_ic[train_idx]).float().to(DEVICE)
coords_ic_test = torch.from_numpy(coords_ic[test_idx]).float().to(DEVICE)

ic_ic_train = torch.from_numpy(ic_ic[train_idx]).float().to(DEVICE)
ic_ic_test = torch.from_numpy(ic_ic[test_idx]).float().to(DEVICE)

target_ic_train = torch.from_numpy(target_ic[train_idx]).float().to(DEVICE)
target_ic_test = torch.from_numpy(target_ic[test_idx]).float().to(DEVICE)

train_ds = TensorDataset(ic_ic_train, coords_ic_train, target_ic_train)
test_ds = TensorDataset(ic_ic_test, coords_ic_test, target_ic_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# 5. DEFINE MODELS
# ============================================================
print("\n5. Defining DeepONet models...")

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
    def __init__(self, trunk, branch, c, k):
        super().__init__()
        self.trunk = trunk
        self.branch = branch
        self.c = c
        self.k = k

    def forward(self, ic, coords):
        trunk_out = self.trunk(coords)  # (B, N_MODES)
        branch_out = self.branch(ic)    # (B, N_MODES)
        u = torch.sum(trunk_out * branch_out, dim=1)  # (B,)
        return u

    def compute_pde_residual(self, ic, xyt):
        xyt_grad = xyt.clone().requires_grad_(True)
        u = self.forward(ic, xyt_grad)

        # First derivatives
        grad_u = torch.autograd.grad(
            u, xyt_grad,
            torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        u_x = grad_u[:, 0]
        u_y = grad_u[:, 1]
        u_t = grad_u[:, 2]

        # Second derivatives in space
        u_xx = torch.autograd.grad(
            u_x, xyt_grad,
            torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0][:, 0]

        u_yy = torch.autograd.grad(
            u_y, xyt_grad,
            torch.ones_like(u_y),
            create_graph=True,
            retain_graph=True
        )[0][:, 1]

        # Second derivative in time
        u_tt = torch.autograd.grad(
            u_t, xyt_grad,
            torch.ones_like(u_t),
            create_graph=True
        )[0][:, 2]

        # PDE residual WITHOUT source: u_tt + k*u_t - c^2*(u_xx + u_yy) = 0
        residual = u_tt + self.k * u_t - self.c**2 * (u_xx + u_yy)
        return residual

trunk = MLP(3, TRUNK_HIDDEN_DIM, N_MODES, TRUNK_N_LAYERS).to(DEVICE)
branch = MLP(ic_dim, BRANCH_HIDDEN_DIM, N_MODES, BRANCH_N_LAYERS).to(DEVICE)
model = DeepONet(trunk, branch, WAVE_SPEED, DAMPING_COEFF).to(DEVICE)

# Load pre-trained supervised model
pretrained_path = os.path.join(SCRIPT_DIR, 'data/deeponet_joint_supervised_20260206_130551.pth')
if os.path.exists(pretrained_path):
    print(f"\n>>> Loading pre-trained supervised model from: {pretrained_path}")
    pretrained_ckpt = torch.load(pretrained_path, map_location=DEVICE)
    model.load_state_dict(pretrained_ckpt['model_state_dict'])
    print("✓ Pre-trained model loaded successfully")
    print(">>> Now fine-tuning with PINN loss...")
else:
    print(f"\n>>> WARNING: Pre-trained model not found at {pretrained_path}")
    print(">>> Starting from random initialization")

# ============================================================
# 6. TRAINING LOOP (IC + PDE)
# ============================================================
print("\n6. Training...")
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

train_losses = []
test_losses = []

for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss = 0.0

    for ic_b, coords_b, target_b in train_loader:
        optimizer.zero_grad()

        # IC supervised loss (t=0)
        pred_ic = model(ic_b, coords_b)
        ic_loss = loss_fn(pred_ic, target_b)

        # PDE residual loss (random collocation points)
        # Sample random collocation points and random samples for IC
        colloc_idx = np.random.randint(0, N_samples, size=N_COLLOC_PER_EPOCH)
        ic_colloc = ic_norm[colloc_idx]
        x_colloc = np.random.rand(N_COLLOC_PER_EPOCH, 1)
        y_colloc = np.random.rand(N_COLLOC_PER_EPOCH, 1)
        t_colloc = np.random.rand(N_COLLOC_PER_EPOCH, 1)
        coords_colloc = np.hstack([x_colloc, y_colloc, t_colloc])

        ic_colloc_t = torch.from_numpy(ic_colloc).float().to(DEVICE)
        coords_colloc_t = torch.from_numpy(coords_colloc).float().to(DEVICE)

        residual = model.compute_pde_residual(ic_colloc_t, coords_colloc_t)
        pde_loss = torch.mean(residual**2)

        loss = IC_LOSS_WEIGHT * ic_loss + PDE_LOSS_WEIGHT * pde_loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    train_loss = epoch_loss / len(train_loader)
    train_losses.append(train_loss)

    # Evaluate IC loss on test set
    model.eval()
    with torch.no_grad():
        test_loss_val = 0.0
        for ic_b, coords_b, target_b in test_loader:
            pred_ic = model(ic_b, coords_b)
            loss = loss_fn(pred_ic, target_b)
            test_loss_val += loss.item()
        test_loss = test_loss_val / len(test_loader)
        test_losses.append(test_loss)

    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch {epoch+1:4d}/{N_EPOCHS} | Train Loss={train_loss:.6e} | "
            f"Test IC Loss={test_loss:.6e} | PDE Loss={pde_loss.item():.6e}"
        )

# ============================================================
# 7. SAVE CHECKPOINT
# ============================================================
print("\n7. Saving checkpoint...")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
ckpt_path = os.path.join(SCRIPT_DIR, f"data/deeponet_pinn_finetuned_{timestamp}.pth")

checkpoint = {
    'model_state_dict': model.state_dict(),
    'config': {
        'n_modes': N_MODES,
        'n_sensors': N_SENSORS,
        'trunk_hidden_dim': TRUNK_HIDDEN_DIM,
        'trunk_n_layers': TRUNK_N_LAYERS,
        'branch_hidden_dim': BRANCH_HIDDEN_DIM,
        'branch_n_layers': BRANCH_N_LAYERS,
        'points_per_sample_ic': POINTS_PER_SAMPLE_IC,
        'n_colloc_per_epoch': N_COLLOC_PER_EPOCH,
        'wave_speed': WAVE_SPEED,
        'damping_coeff': DAMPING_COEFF,
        'pretrained_from': pretrained_path if os.path.exists(pretrained_path) else None,
    },
    'normalization': {
        'ic_min': ic_min,
        'ic_max': ic_max,
        'ic_range': ic_range,
    },
    'train_loss': train_losses,
    'test_loss': test_losses,
}

torch.save(checkpoint, ckpt_path)
print(f"✓ Checkpoint saved to {ckpt_path}")

# ============================================================
# 8. QUICK EVAL ON SAMPLE 0 AT 5 TIMES
# ============================================================
print("\n8. Evaluating sample 0 at 5 time instants...")

model.eval()
with torch.no_grad():
    ic0 = ic_norm[0:1, :]
    ic0_tensor = torch.from_numpy(ic0).float().to(DEVICE)

    time_instants = [0.0, 0.25, 0.5, 0.75, 1.0]
    time_indices = [int(ti * (Nt - 1)) for ti in time_instants]

    preds = []
    gts = []
    for t_val, t_idx in zip(time_instants, time_indices):
        coords_t = np.stack([X.flatten('F'), Y.flatten('F'), np.full((Nx * Ny,), t_val)], axis=1)
        coords_tensor = torch.from_numpy(coords_t).float().to(DEVICE)
        ic_batch = ic0_tensor.repeat(coords_tensor.shape[0], 1)

        u_pred = model(ic_batch, coords_tensor).cpu().numpy()
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
plot_path = os.path.join(SCRIPT_DIR, f"data/deeponet_pinn_finetuned_eval_sample0_{timestamp}.png")
plt.savefig(plot_path, dpi=150)
print(f"✓ Evaluation plot saved to {plot_path}")
plt.close()

# ============================================================
# 9. LOSS CURVES
# ============================================================
plt.figure(figsize=(8, 5))
plt.semilogy(train_losses, label='Train (IC+PDE)')
plt.semilogy(test_losses, label='Test (IC)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('DeepONet PINN Training Loss')
plt.legend()
plt.grid(True, alpha=0.3)
loss_path = os.path.join(SCRIPT_DIR, f"data/deeponet_pinn_finetuned_loss_{timestamp}.png")
plt.savefig(loss_path, dpi=150)
print(f"✓ Loss plot saved to {loss_path}")
plt.close()

print("\n" + "=" * 70)
print("✓ TRAINING COMPLETE")
print("=" * 70)
