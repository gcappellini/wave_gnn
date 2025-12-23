import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
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

log_fh = open(LOG_PATH, 'w')
sys.stdout = Tee(log_fh, also_stdout=True)
sys.stderr = sys.stdout

# --- CONFIGURATION ---
SVD_PATH = os.path.join(SCRIPT_DIR, 'data', 'svd_basis_data.npy')
MODEL_SAVE_PATH = os.path.join(LOG_DIR, 'pretrained_trunk_p64.pth')
RANK = 64                # We limit to 64 for this test run
TRUNK_HIDDEN = 300       # Hidden layer size for Trunk
TRUNK_N_LAYERS = 6       # Number of hidden layers
BATCH_SIZE = 10000       # Large batch for fast training
EPOCHS = 2000
LR = 1e-3

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

# Train/Test split for validation
from torch.utils.data import random_split
dataset = TensorDataset(inputs_t, targets_t)
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

print(f"Train size: {train_size}, Test size: {test_size}")

# --- 3. TRAINING LOOP ---
fft_dict = {"input_dim": 3,
    "m_spatial_x": 64, 
    "m_spatial_y": 64,
    "m_temporal": 64,
    "sigma_spatial_x": 0.10,
    "sigma_spatial_y": 0.10,
    "sigma_temporal_list": [0.10],
    "seed": 42}

fft_trunk = FourierFeatureTransform(**fft_dict)
trunk = TrunkNet(
    hidden_dim=TRUNK_HIDDEN,
    output_dim=RANK,
    n_hidden_layers=TRUNK_N_LAYERS,
    activation=nn.Tanh(),
    fft_transform=fft_trunk
).to(device)
optimizer = optim.Adam(trunk.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
criterion = nn.MSELoss()
scaler = GradScaler(enabled=device.type == "cuda")

print("Starting Trunk Pre-training...")
train_loss_history = []
test_loss_history = []

training_start = time.time()

# Compute initial loss for normalization
trunk.eval()
with torch.no_grad():
    initial_loss = 0
    for batch_x, batch_y in train_loader:
        pred = trunk(batch_x)
        loss = criterion(pred, batch_y)
        initial_loss += loss.item()
    initial_loss /= len(train_loader)
    print(f"Initial Loss (before training): {initial_loss:.2e}")

for epoch in range(EPOCHS):
    # Training
    trunk.train()
    total_loss = 0
    
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=device.type == "cuda"):
            pred = trunk(batch_x)
            loss = criterion(pred, batch_y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.detach().item()
    
    avg_train_loss = total_loss / len(train_loader)
    normalized_train_loss = avg_train_loss / initial_loss
    train_loss_history.append(normalized_train_loss)
    
    # Validation
    trunk.eval()
    total_test_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            with autocast(enabled=device.type == "cuda"):
                pred = trunk(batch_x)
                loss = criterion(pred, batch_y)
            total_test_loss += loss.detach().item()
    
    avg_test_loss = total_test_loss / len(test_loader)
    normalized_test_loss = avg_test_loss / initial_loss
    test_loss_history.append(normalized_test_loss)
    
    scheduler.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {normalized_train_loss:.4f} | Test Loss: {normalized_test_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.1e}")

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
plt.savefig(os.path.join(SCRIPT_DIR, 'data', 'trunk_pretrain_loss.png'))
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
plt.savefig(os.path.join(SCRIPT_DIR, 'data', 'trunk_verification.png'))
plt.show()