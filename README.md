# GCN Wave Equation Experiments

This project implements Graph Convolutional Networks (GCN) for simulating 1D wave equations. The experiments are managed using Hydra for configuration management.

## Project Structure

```
1_gcn_string/
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main config with defaults
│   ├── dataset/
│   │   └── default.yaml       # Dataset configuration
│   ├── model/
│   │   ├── gcn.yaml          # Standard GCN model config
│   │   └── deep_gcn.yaml     # Deep GCN model config
│   └── training/
│       ├── default.yaml       # Default training config
│       └── fast.yaml          # Fast training for testing
├── main.py                     # Main experiment script
├── train.py                    # Training functions and model definition
├── test_gcn.py                 # Testing and evaluation functions
├── import_mesh.py              # Dataset creation and physics solver
├── plot.py                     # Visualization utilities
├── requirements.txt            # Python dependencies
└── outputs/                    # Generated experiment outputs
    └── YYYY-MM-DD/
        └── HH-MM-SS/
            ├── config.yaml     # Saved experiment config
            ├── summary.yaml    # Experiment results
            ├── figures/        # Generated plots
            └── matlab/         # MATLAB data exports
```

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

Run with default configuration:
```bash
python main.py
```

### Custom Configurations

Use different model architecture:
```bash
python main.py model=deep_gcn
```

Use fast training config (fewer epochs):
```bash
python main.py training=fast
```

### Override Parameters

Override specific parameters from command line:
```bash
# Change learning rate
python main.py training.learning_rate=0.01

# Change number of epochs
python main.py training.epochs=100

# Change dataset size
python main.py dataset.num_graphs=200

# Use different random seed
python main.py experiment.seed=123

# Enable/disable data scaling
python main.py dataset.scaling.enabled=true
python main.py dataset.scaling.method=minmax

# Combine multiple overrides
python main.py model=deep_gcn training=fast experiment.seed=42
```

### Data Scaling

Control input/output normalization:
```bash
# Enable standard normalization (z-score) - recommended
python main.py dataset.scaling.enabled=true dataset.scaling.method=standard

# Min-max normalization
python main.py dataset.scaling.enabled=true dataset.scaling.method=minmax

# Disable scaling (default)
python main.py dataset.scaling.enabled=false
```

See [SCALING_GUIDE.md](SCALING_GUIDE.md) for details.

### Adaptive Loss Weighting

Automatically balance multiple loss terms during training:
```bash
# Use equal initialization (equalizes all losses at epoch 1)
python main.py training=adaptive_equal_init

# Use EMA strategy (continuous adaptation)
python main.py training=adaptive_ema

# Compare strategies
python main.py -m training=default,adaptive_equal_init,adaptive_ema
```

See [ADAPTIVE_WEIGHTS_GUIDE.md](ADAPTIVE_WEIGHTS_GUIDE.md) for comprehensive documentation.

# Use min-max scaling
python main.py dataset.scaling.enabled=true dataset.scaling.method=minmax

# Disable scaling
python main.py dataset.scaling.enabled=false
```

See `SCALING_GUIDE.md` for detailed information about data scaling implementation.

### Residual Formulation

Use residual learning (predict changes Δu, Δv instead of absolute values):
```bash
# Use residual GCN model (predicts changes)
python main.py model=residual_gcn

# Enable residual on any model
python main.py model=deep_gcn model.residual=true

# Compare residual vs absolute formulation
python main.py -m model.residual=true,false
```

See `RESIDUAL_FORMULATION.md` for detailed information about residual learning.

### Multiple Experiments (Sweeps)

Run experiments with different configurations:
```bash
# Sweep over learning rates
python main.py -m training.learning_rate=0.001,0.005,0.01

# Sweep over model architectures
python main.py -m model=gcn,deep_gcn

# Sweep over seeds for multiple runs
python main.py -m experiment.seed=1,2,3,4,5
```

## Configuration

### Dataset Configuration (`configs/dataset/default.yaml`)

- `num_graphs`: Number of training graphs to generate
- `num_steps`: Timesteps per simulation
- `dt`: Time step size
- `train_ratio`: Train/validation split ratio
- `batch_size`: Batch size for training
- `wave_speed`: Wave propagation speed (c)
- `damping`: Damping coefficient (k)

### Model Configuration (`configs/model/gcn.yaml`)

- `in_channels`: Input feature dimension (default: 3 for u, v, f)
- `hidden_channels`: Hidden layer dimensions
- `out_channels`: Output dimension (default: 2 for u, v)
- `layer_types`: Type of layers ("GCN" or "Linear")
- `activation`: Activation function ("relu" or "tanh")
- `dropout`: Dropout probability

### Training Configuration (`configs/training/default.yaml`)

- `epochs`: Number of training epochs
- `learning_rate`: Learning rate for optimizer
- `weight_decay`: L2 regularization weight
- `loss.w1_PI`: Weight for displacement loss (physics-informed)
- `loss.w2_PI`: Weight for velocity loss (physics-informed)
- `loss.w1_rk4`: Weight for RK4 displacement loss
- `loss.w2_rk4`: Weight for RK4 velocity loss
- `loss.use_rk4`: Enable RK4 loss term
- `loss.use_gn_solver`: Enable GN solver loss term
- `loss.adaptive.enabled`: Enable adaptive loss weighting
- `loss.adaptive.strategy`: Weighting strategy ('equal_init', 'equal_init_ema', 'ema', 'fixed')
- `log_interval`: Logging frequency (epochs)
- `early_stopping.enabled`: Enable early stopping
- `early_stopping.patience`: Patience for early stopping

## Output Structure

Each experiment creates a timestamped directory in `outputs/` containing:

- `config.yaml`: Full configuration used for the experiment
- `summary.yaml`: Training metrics and test results
- `figures/`: Generated visualizations
- `matlab/`: MATLAB-compatible data exports

Checkpoints are saved in `checkpoints/` directory.

## Legacy Mode

The original scripts can still be run independently:

```bash
# Old training script (standalone)
python train.py

# Old testing script (standalone)
python test_gcn.py
```

## Examples

### Quick test run
```bash
python main.py training=fast dataset.num_graphs=20
```

### Production training
```bash
python main.py training.epochs=100 dataset.num_graphs=500
```

### Experiment with different architectures
```bash
python main.py -m model=gcn,deep_gcn training.epochs=50
```

## Notes

- All experiments are automatically logged with Hydra
- Outputs are organized by date and time
- Configuration files are saved with each experiment for reproducibility
- Random seeds can be set for reproducible results

## Troubleshooting

If you encounter import errors, make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

For CUDA issues, ensure PyTorch is installed with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
