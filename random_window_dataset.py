import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from pathlib import Path

try:
    import h5py
    _H5PY_AVAILABLE = True
except Exception:
    h5py = None
    _H5PY_AVAILABLE = False

from dataset import build_laplacian_matrix

class H5RandomTimeStepDataset(Dataset):
    """
    Random time-step dataset sampled from HDF5 simulations with per-epoch random windows.

    For each simulation (S), we pick a random window start s in [0, T-window_length]
    at initialization (and can resample per epoch). The dataset length is S * window_length.
    Index i maps to (sim_idx, offset) with t = s + offset.

    Returns torch_geometric.data.Data per item with:
      - x: [N, 3] tensor for [u, v, f] at time t
      - edge_index: [2, E]
      - bc_mask: [N] bool
      - coords: [N, 1]
      - laplacian: sparse COO (N, N)
    """
    def __init__(self, h5_path: str, window_length: int = 200, num_samples: int = None, num_timesteps: int = None):
        """
        Args:
            h5_path: Path to HDF5 file
            window_length: Number of timesteps per random window
            num_samples: Number of simulations to load (None = all)
            num_timesteps: Max timesteps per simulation to use (None = all)
        """
        if not _H5PY_AVAILABLE:
            raise ImportError("h5py is required for H5RandomTimeStepDataset. Please pip install h5py.")
        
        self.h5_path = str(h5_path)
        self.window_length = int(window_length)
        
        # Validate HDF5 file exists and is valid
        h5_file = Path(self.h5_path)
        if not h5_file.exists():
            raise FileNotFoundError(
                f"HDF5 file not found: {self.h5_path}\n"
                f"Please generate it first with: python dataset.py"
            )
        
        if h5_file.stat().st_size == 0:
            raise ValueError(
                f"HDF5 file is empty: {self.h5_path}\n"
                f"Please regenerate it with: python dataset.py"
            )
        
        # Validate it's a valid HDF5 file
        if not h5py.is_hdf5(self.h5_path):
            raise ValueError(
                f"File is not a valid HDF5 file: {self.h5_path}\n"
                f"File size: {h5_file.stat().st_size} bytes\n"
                f"Please regenerate it with: python dataset.py"
            )

        with h5py.File(self.h5_path, 'r') as f:
            feats = f['features']  # (S, T, N, C)
            S_total, T_total, self.N, self.C = feats.shape
            self.coords_np = f['coords'][:]  # (N, 1)
            self.edge_index_np = f['edge_index'][:]  # (2, E)
            self.bc_mask_np = f['bc_mask'][:]  # (N,)
            self.dt = float(f.attrs.get('dt', 0.01))

        # Apply num_samples and num_timesteps limits
        self.S = min(num_samples, S_total) if num_samples is not None else S_total
        self.T = min(num_timesteps, T_total) if num_timesteps is not None else T_total
        
        # Clamp window_length to available timesteps
        self.window_length = min(self.window_length, self.T)

        # Precompute Laplacian from coords assuming 1D chain with uniform spacing
        coords_1d = self.coords_np[:, 0]
        if self.N > 1:
            dx = float(coords_1d[1] - coords_1d[0])
        else:
            dx = 1.0
        self.laplacian = build_laplacian_matrix(self.N, dx)

        self._resample_window_starts()

    def _resample_window_starts(self):
        max_start = max(0, self.T - self.window_length)
        self.starts = np.random.randint(0, max_start + 1, size=self.S)

    def on_epoch_end(self):
        self._resample_window_starts()

    def __len__(self):
        return self.S * self.window_length

    def __getitem__(self, idx):
        per_sim = self.window_length
        sim_idx = idx // per_sim
        offset = idx % per_sim
        start = int(self.starts[sim_idx])
        t = min(start + offset, self.T - 1)

        with h5py.File(self.h5_path, 'r') as f:
            x_np = f['features'][sim_idx, t]  # (N, 3)

        data = Data(
            x=torch.from_numpy(x_np).float(),
            edge_index=torch.from_numpy(self.edge_index_np).long(),
            bc_mask=torch.from_numpy(self.bc_mask_np).bool(),
            coords=torch.from_numpy(self.coords_np).float(),
            laplacian=self.laplacian,
        )
        return data
