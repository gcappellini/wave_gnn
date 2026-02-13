import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
from datetime import datetime
from model_2d import TrunkNet, FourierFeatureTransform

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- LOGGING SETUP ---
timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
LOG_DIR = os.path.join(SCRIPT_DIR, 'outputs', timestamp)
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, 'pretrain_trunk.log')

class Tee:
    def __init__(self, log_file, also_stdout=True):
        self.log_file = log_file
        self.also_stdout = also_stdout
        self._stdout = sys.stdout if also_stdout else None

    def write(self, message):
        if self.also_stdout and self._stdout:
            self._stdout.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        if self.also_stdout and self._stdout:
            self._stdout.flush()
        self.log_file.flush()
    
    def isatty(self):
        """Return True if stdout is a TTY (required for torch.compile compatibility)."""
        if self.also_stdout and self._stdout:
            return self._stdout.isatty()
        return False

log_fh = open(LOG_PATH, 'w')
sys.stdout = Tee(log_fh, also_stdout=True)
sys.stderr = sys.stdout

# --- CONFIGURATION ---
SVD_PATH = os.path.join(SCRIPT_DIR, 'data', 'svd_basis_data.npy')
MODEL_SAVE_PATH = os.path.join(LOG_DIR, 'pretrained_trunk.pth')
RANK = 64                # We limit to 64 for this test run
TRUNK_HIDDEN = 300       # Hidden layer size for Trunk
TRUNK_N_LAYERS = 6       # Number of hidden layers
BATCH_SIZE = 100000      # Massive batch size for GPU saturation (tune as needed)
EPOCHS = 2000
LR = 1e-3
USE_COMPILE = True       # Enable torch.compile for kernel fusion (PyTorch 2.0+)
USE_AMP = True           # Enable Automatic Mixed Precision (FP16/BF16)

# Save configuration to log directory
config_dict = {
    'timestamp': timestamp,
    'svd_path': SVD_PATH,
    'model_save_path': MODEL_SAVE_PATH,
    'rank': RANK,
    'trunk_hidden': TRUNK_HIDDEN,
    'trunk_n_layers': TRUNK_N_LAYERS,
    'batch_size': BATCH_SIZE,
    'epochs': EPOCHS,
    'lr': LR,
    'device': str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
    'train_test_split': 0.9,
    'use_compile': USE_COMPILE,
    'use_amp': USE_AMP
}

import json
config_path = os.path.join(LOG_DIR, 'config.json')
with open(config_path, 'w') as f:
    json.dump(config_dict, f, indent=4)
print(f"Configuration saved to {config_path}\n")

# --- 2. DATA LOADING ---
print(f"Loading SVD data from {SVD_PATH}...")
data_dict = np.load(SVD_PATH, allow_pickle=True).item()
U_basis = data_dict['basis'] # Shape: (Nx*Ny*Nt, Total_Modes)
Nx, Ny, Nt = data_dict['grid_info']

# Check dimensions
print(f"Basis shape: {U_basis.shape}")
if U_basis.shape[1] < RANK:
    raise ValueError(f"SVD only has {U_basis.shape[1]} modes, but RANK is set to {RANK}")

# Slice the first RANK modes
targets = U_basis[:, :RANK] # Shape: (SpaceTimePoints, RANK)

# Generate Coordinate Grid (Must match the flattened order of U_basis!)
# Assuming U_basis was flattened with order='F' (MATLAB style) as discussed
print("Generating coordinates...")
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
t = np.linspace(0, 1, Nt)

# Meshgrid ordered to match MATLAB flattening (x varies fastest, then y, then t)
X, Y, T = np.meshgrid(x, y, t, indexing='ij') 
# Flatten with 'F' to match the SVD data
coords = np.stack([X.flatten('F'), Y.flatten('F'), T.flatten('F')], axis=1)

print(f"Coords shape: {coords.shape}, Targets shape: {targets.shape}")

# Convert to Tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inputs_t = torch.tensor(coords, dtype=torch.float32).to(device)
targets_t = torch.tensor(targets, dtype=torch.float32).to(device)

print(f"✓ Full dataset pre-loaded to {device.type.upper()}")
print(f"  Inputs shape: {inputs_t.shape}, GPU memory: {inputs_t.element_size() * inputs_t.nelement() / 1e9:.2f} GB")
print(f"  Targets shape: {targets_t.shape}, GPU memory: {targets_t.element_size() * targets_t.nelement() / 1e9:.2f} GB")

# Manual train/test split indices (no DataLoader, no workers)
total_size = len(inputs_t)
train_size = int(0.9 * total_size)
test_size = total_size - train_size

# Random permutation for shuffling
perm = torch.randperm(total_size, device=device)
train_indices = perm[:train_size]
test_indices = perm[train_size:]

# Pre-allocate batches on GPU for faster access
train_inputs = inputs_t[train_indices]
train_targets = targets_t[train_indices]
test_inputs = inputs_t[test_indices]
test_targets = targets_t[test_indices]

print(f"Train size: {train_size}, Test size: {test_size}")
print(f"✓ Manual batching setup complete (no DataLoader overhead)")

# --- 3. TRAINING LOOP ---
fft_dict = {"input_dim": 3,
    "m_spatial_x": 64, 
    "m_spatial_y": 64,
    "m_temporal": 64,
    "sigma_spatial_x": 0.10,
    "sigma_spatial_y": 0.10,
    "sigma_temporal_list": [0.10],
    "seed": 42}

# Update config with FFT parameters
config_dict['fft_trunk_args'] = fft_dict
with open(config_path, 'w') as f:
    json.dump(config_dict, f, indent=4)

fft_trunk = FourierFeatureTransform(**fft_dict)
trunk = TrunkNet(
    hidden_dim=TRUNK_HIDDEN,
    output_dim=RANK,
    n_hidden_layers=TRUNK_N_LAYERS,
    activation=nn.Tanh(),
    fft_transform=fft_trunk
).to(device)

# =========== KEY OPTIMIZATION 1: TORCH COMPILE ===========
# Fuse kernels and reduce Python overhead (PyTorch 2.0+)
if USE_COMPILE:
    try:
        trunk = torch.compile(trunk, mode='max-autotune')
        print("✓ Model compiled with torch.compile (kernel fusion enabled)")
    except Exception as e:
        print(f"⚠ torch.compile not available or failed ({e}), running in eager mode")

optimizer = optim.Adam(trunk.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
criterion = nn.MSELoss()
scaler = GradScaler(enabled=(device.type == "cuda") and USE_AMP)

if USE_AMP:
    print(f"✓ Automatic Mixed Precision (AMP) enabled")



print("Starting Trunk Pre-training...")
train_loss_history = []
test_loss_history = []

training_start = time.time()

# Compute initial loss for normalization
trunk.eval()
with torch.no_grad():
    initial_loss = 0
    n_batches = 0
    for idx in range(0, train_size, BATCH_SIZE):
        batch_end = min(idx + BATCH_SIZE, train_size)
        batch_x = train_inputs[idx:batch_end]
        batch_y = train_targets[idx:batch_end]
        with autocast(enabled=(device.type == "cuda") and USE_AMP):
            pred = trunk(batch_x)
            loss = criterion(pred, batch_y)
        initial_loss += loss.item()
        n_batches += 1
    initial_loss /= n_batches
    print(f"Initial Loss (before training): {initial_loss:.2e}")

for epoch in range(EPOCHS):
    # ========== TRAINING LOOP ==========
    trunk.train()
    total_loss = 0
    n_batches = 0
    
    # Shuffle training data
    perm = torch.randperm(train_size, device=device)
    shuffled_inputs = train_inputs[perm]
    shuffled_targets = train_targets[perm]
    
    # Manual batching with GPU pre-loaded data
    for idx in range(0, train_size, BATCH_SIZE):
        batch_end = min(idx + BATCH_SIZE, train_size)
        batch_x = shuffled_inputs[idx:batch_end]
        batch_y = shuffled_targets[idx:batch_end]
        
        optimizer.zero_grad(set_to_none=True)
        
        # =========== KEY OPTIMIZATION 2: AUTOMATIC MIXED PRECISION ===========
        with autocast(enabled=(device.type == "cuda") and USE_AMP):
            pred = trunk(batch_x)
            loss = criterion(pred, batch_y)
        
        # Scale loss and backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.detach().item()
        n_batches += 1
    
    avg_train_loss = total_loss / n_batches
    normalized_train_loss = avg_train_loss / initial_loss
    train_loss_history.append(normalized_train_loss)
    
    # ========== VALIDATION LOOP ==========
    trunk.eval()
    total_test_loss = 0
    n_batches = 0
    with torch.no_grad():
        for idx in range(0, test_size, BATCH_SIZE):
            batch_end = min(idx + BATCH_SIZE, test_size)
            batch_x = test_inputs[idx:batch_end]
            batch_y = test_targets[idx:batch_end]
            with autocast(enabled=(device.type == "cuda") and USE_AMP):
                pred = trunk(batch_x)
                loss = criterion(pred, batch_y)
            total_test_loss += loss.detach().item()
            n_batches += 1
    
    avg_test_loss = total_test_loss / n_batches
    normalized_test_loss = avg_test_loss / initial_loss
    test_loss_history.append(normalized_test_loss)
    
    scheduler.step()
    
    if epoch % 100 == 0:
        elapsed = time.time() - training_start
        print(f"Epoch {epoch:4d}/{EPOCHS} | Train Loss: {normalized_train_loss:.4f} | Test Loss: {normalized_test_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.1e} | Time: {elapsed/60:.1f}m")

training_time_sec = time.time() - training_start
print(f"Training finished in {training_time_sec/60:.2f} min ({training_time_sec:.1f} sec)")

# --- 4. SAVE & VISUALIZE ---
torch.save(trunk.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

plt.figure(figsize=(10, 5))
plt.semilogy(train_loss_history, label='Train', linewidth=2)
plt.semilogy(test_loss_history, label='Test', linewidth=2, linestyle='--')
plt.title("Trunk Pre-training Loss (Normalized)")
plt.xlabel("Epochs")
plt.ylabel("Normalized Loss (Initial Loss = 1.0)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(LOG_DIR, 'trunk_pretrain_loss.png'))
plt.show()

# Verification Plot
trunk.eval()
with torch.no_grad():
    # Predict bases for the whole domain
    pred_bases = trunk(inputs_t).cpu().numpy()

# Compare True Basis #0 vs Predicted Basis #0 at t=0.5
mid_t = Nt // 2
layer_size = Nx * Ny
# Extract slice for t=mid_t
start_idx = mid_t * layer_size
end_idx = start_idx + layer_size

true_slice = targets[start_idx:end_idx, 0].reshape(Nx, Ny, order='F')
pred_slice = pred_bases[start_idx:end_idx, 0].reshape(Nx, Ny, order='F')

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].imshow(true_slice, origin='lower', cmap='seismic')
ax[0].set_title("True SVD Basis #0")
ax[1].imshow(pred_slice, origin='lower', cmap='seismic')
ax[1].set_title("Neural Trunk Basis #0")
plt.savefig(os.path.join(LOG_DIR, 'trunk_verification.png'))
plt.show()