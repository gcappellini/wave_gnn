# Setup Guide for Remote Server

## HDF5 Dataset Generation Error

If you encounter this error on the remote server:
```
OSError: Unable to synchronously open file (file signature not found)
```

This means the HDF5 dataset file doesn't exist or is corrupted.

## Solution

### 1. Generate the HDF5 Dataset

Before running training, you must generate the dataset file:

```bash
# On remote server
cd /home/jupyter-gcap/wave_gnn
python dataset.py
```

This will create `data/simulations.h5` with 100 simulations × 500 timesteps.

### 2. Verify Dataset Creation

Check that the file was created successfully:

```bash
ls -lh data/simulations.h5
```

You should see a file size of several hundred MB.

### 3. Run Training

Now you can run training:

```bash
python main.py
```

## Configuration Options

### Fast Dataset (for testing)
```bash
python main.py dataset=fast_dataset
```
- Uses 20 simulations × 50 timesteps
- Faster generation and training

### SSH Dataset (full training)
```bash
python main.py dataset=ssh_dataset
```
- Uses 100 simulations × 200 timesteps
- Full training configuration

## Custom Dataset Generation

To generate a custom dataset:

```python
from dataset import generate_h5_simulations

generate_h5_simulations(
    out_path="data/my_dataset.h5",
    num_samples=50,      # Number of simulations
    num_steps=300,       # Timesteps per simulation
    base_seed=2025,      # Random seed
    overwrite=True       # Overwrite existing file
)
```

## Troubleshooting

### File exists but is corrupted
```bash
# Remove corrupted file
rm data/simulations.h5
# Regenerate
python dataset.py
```

### Permission errors
```bash
# Create data directory if it doesn't exist
mkdir -p data
chmod 755 data
```

### Disk space issues
```bash
# Check available space
df -h .
```

Each simulation is approximately 2-5 MB, so 100 simulations requires ~200-500 MB.
