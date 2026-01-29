"""
Plot MATLAB sample 0 on full spatial grid at 5 time instants.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MATLAB_PATH = os.path.join(SCRIPT_DIR, 'data/test_cases.mat')

if not os.path.exists(MATLAB_PATH):
    raise FileNotFoundError(f"MATLAB data not found: {MATLAB_PATH}")

# MATLAB v7.3 files use HDF5 format, need h5py
with h5py.File(MATLAB_PATH, 'r') as f:
    # h5py reads MATLAB arrays in transposed format, so we transpose back
    u_fom = np.array(f['U_data']).T  # (Nx, Ny, Nt, N_samples)

Nx, Ny, Nt, N_samples = u_fom.shape
print(f"Loaded u_fom: {u_fom.shape}")

sample_idx = 0
u_sample0 = u_fom[:, :, :, sample_idx]  # (Nx, Ny, Nt)

# Time instants to plot
time_instants = [0.0, 0.25, 0.5, 0.75, 1.0]
time_indices = [int(t * (Nt - 1)) for t in time_instants]

print(f"Plotting sample {sample_idx} at time indices: {time_indices}")

fig, axes = plt.subplots(1, len(time_instants), figsize=(4 * len(time_instants), 4), constrained_layout=True)

for i, (t_val, t_idx) in enumerate(zip(time_instants, time_indices)):
    field = u_sample0[:, :, t_idx]
    ax = axes[i]
    im = ax.imshow(field, cmap='seismic', origin='lower')
    ax.set_title(f"t={t_val:.2f}")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046)

save_path = os.path.join(SCRIPT_DIR, 'data/matlab_sample0_5times.png')
plt.savefig(save_path, dpi=150)
print(f"Saved figure to {save_path}")
plt.close()
