import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. CONFIGURAZIONE
DATA_PATH = 'data/test_cases.mat'  # Il file generato da MATLAB
N_MODES = 128                    # Numero di basi da estrarre (p)
VISUALIZE = True

print(f"Loading data from {DATA_PATH}...")
# MATLAB v7.3 files use HDF5 format, need h5py instead of scipy.io
with h5py.File(os.path.join(SCRIPT_DIR, DATA_PATH), 'r') as f:
    # h5py reads MATLAB arrays in transposed format, so we transpose back
    U_raw = np.array(f['U_data']).T  # Now shape: (Nx, Ny, Nt, N_samples) 
print(f"Original Shape: {U_raw.shape}")

# 2. RESHAPING PER SVD
# Vogliamo appiattire (Nx, Ny, Nt) in una sola dimensione "Features"
# E tenere N_samples come dimensione "Samples"
Nx, Ny, Nt, N_samples = U_raw.shape
features_dim = Nx * Ny * Nt

print("Reshaping for SVD...")
# Flattening: (Nx*Ny*Nt, N_samples)
# Use Fortran order to match coordinate generation (x varies fastest, then y, then t)
U_matrix = U_raw.reshape(features_dim, N_samples, order='F') 

print(f"Matrix for SVD: {U_matrix.shape} (Rows=SpaceTime Points, Cols=Samples)")

# 3. CALCOLO SVD (Randomized per velocità su matrici grandi)
# Maximum modes = min(features_dim, N_samples)
max_modes = min(features_dim, N_samples)
n_modes_to_compute = min(N_MODES, max_modes)

print(f"Computing Randomized SVD (requested k={N_MODES}, actual k={n_modes_to_compute})...")
# U_basis: Le basi spaziotemporali (Features x p)
# Sigma: I valori singolari (p,)
# VT: I coefficienti temporali/sample (p x Samples) - NON ci servono per il Trunk, servono per la Branch
U_basis, Sigma, VT = randomized_svd(U_matrix, n_components=n_modes_to_compute, random_state=42)

actual_modes = U_basis.shape[1]
print(f"SVD Done. Basis Shape: {U_basis.shape}, Actual Modes: {actual_modes}")

# Compute cumulative energy captured by modes
cumulative_energy = np.cumsum(Sigma**2) / np.sum(Sigma**2)
energy_captured = cumulative_energy[-1] * 100
print(f"Energy captured by {actual_modes} modes: {energy_captured:.2f}%")
print(f"Energy by first 50 modes: {cumulative_energy[min(49, actual_modes-1)]*100:.2f}%")
print(f"Energy by first 100 modes: {cumulative_energy[min(99, actual_modes-1)]*100:.2f}%")

# 4. SALVATAGGIO DATI PROCESSATI
# Salviamo le basi U_basis e i valori singolari. 
# La Trunk Net dovrà imparare a predire U_basis[x,y,t] dato (x,y,t).
output_dict = {
    'basis': U_basis,       # (Nx*Ny*Nt, p)
    'singular_values': Sigma,
    'grid_info': np.array([Nx, Ny, Nt])
}
np.save(os.path.join(SCRIPT_DIR, 'data/svd_basis_data.npy'), output_dict)
print("SVD data saved to svd_basis_data.npy")

# 5. VISUALIZZAZIONE (Check Fisico)
if VISUALIZE:
    print("Visualizing first 18 modes...")
    # Reshape delle basi per plottarle: (Nx, Ny, Nt, actual_modes)
    # Use Fortran order to match the flattening order
    modes_reshaped = U_basis.reshape(Nx, Ny, Nt, actual_modes, order='F')
    
    # Plottiamo i primi 18 modi al tempo t=Nt/2 (metà simulazione)
    t_idx = Nt // 3
    n_modes_to_plot = min(18, actual_modes)
    
    fig, axes = plt.subplots(6, 3, figsize=(15, 30))
    for i in range(n_modes_to_plot):
        row = i // 3
        col = i % 3
        mode_slice = modes_reshaped[:, :, t_idx, i]
        im = axes[row, col].imshow(mode_slice, cmap='seismic', origin='lower')
        axes[row, col].set_title(f"SVD Mode {i} (at t={t_idx})")
        plt.colorbar(im, ax=axes[row, col])
    
    # Hide unused subplots if we have fewer than 18 modes
    for i in range(n_modes_to_plot, 18):
        row = i // 3
        col = i % 3
        axes[row, col].axis('off')
    
    plt.suptitle(f"Trunk Basis Functions ({actual_modes} modes captured)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'data/svd_modes_check.png'), dpi=150)
    plt.show()

    # Plot Decadimento Valori Singolari (Energia)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Singular values decay
    axes[0].semilogy(Sigma)
    axes[0].set_title("Singular Values Decay")
    axes[0].set_xlabel("Mode Index")
    axes[0].set_ylabel("Sigma (Log Scale)")
    axes[0].grid(True, which="both", ls="--")
    
    # Right: Cumulative energy
    axes[1].plot(cumulative_energy * 100, linewidth=2)
    axes[1].axhline(y=90, color='r', linestyle='--', label='90% energy')
    axes[1].axhline(y=99, color='orange', linestyle='--', label='99% energy')
    axes[1].set_title("Cumulative Energy Captured")
    axes[1].set_xlabel("Number of Modes")
    axes[1].set_ylabel("Energy (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'data/svd_energy_decay.png'))
    plt.show()
    
    # Se la curva scende rapidamente, significa che i modi disponibili sono sufficienti.

print(f"\nNote: With {N_samples} samples, max achievable modes = {max_modes}")
print(f"To get {N_MODES} modes, increase N_samples in MATLAB to at least {N_MODES}.")