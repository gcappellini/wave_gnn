# DeepONet Supervised Learning Setup

## Overview

We've implemented a **DeepONet architecture** with supervised learning:
- **Trunk Network** (frozen): Pre-trained MLP mapping space-time coordinates (x, y, t) → 18 SVD basis values
- **Branch Network** (trainable): New MLP mapping initial condition field → 18 branch coefficients
- **Interaction**: Element-wise multiplication to produce final predictions

## Architecture

### Trunk Network (Frozen)
```
Input: (x, y, t) ∈ ℝ³
  ↓
MLP: 3 → 128 → 128 → 128 → 128 → 18
  ↓
Output: [T₁(x,y,t), T₂(x,y,t), ..., T₁₈(x,y,t)] ∈ ℝ¹⁸
```

**Status**: Loaded from `data/trunk_svd_simple.pth` (trained in Phase 1)
**Weights**: Frozen (no gradient updates)

### Branch Network (Trainable)
```
Input: Initial Condition Field ∈ ℝ^(18·Nₓ·Nᵧ)
       Flattened SVD basis at t=0: [a₁(IC), a₂(IC), ..., a₁₈(IC)]
  ↓
MLP: 18·Nₓ·Nᵧ → 128 → 128 → 128 → 128 → 18
  ↓
Output: [B₁(IC), B₂(IC), ..., B₁₈(IC)] ∈ ℝ¹⁸
```

**Status**: Newly initialized and trained
**Weights**: Updated via gradient descent

### Forward Pass (DeepONet)
```
Given: IC (initial condition), (x, y, t)

u(x,y,t | IC) = ∑ᵢ Tᵢ(x,y,t) * Bᵢ(IC)

where:
  - Tᵢ = Trunk output (i-th SVD basis value)
  - Bᵢ = Branch output (i-th coefficient)
  - Element-wise multiplication in ℝ¹⁸
```

## Training Setup

### Loss Function
```
ℒ = MSE(ŷ, y)
  = (1/N) ∑ₙ ||ŷₙ - yₙ||₂²

where:
  - ŷₙ = DeepONet prediction at space-time point n
  - yₙ = Ground truth SVD coefficients at point n
  - || · ||₂² is L2 norm squared over 18 modes
```

### Optimization
- **Optimizer**: Adam
- **Learning Rate**: 1e-3 with ReduceLROnPlateau scheduler
- **Batch Size**: 256
- **Epochs**: 500 (with early stopping)
- **Patience**: 50 epochs

### Data Preparation
1. Extract initial condition: Solution at t=0 (Nₓ × Nᵧ spatial points)
2. Flatten IC in SVD basis: 18 coefficients × Nₓ × Nᵧ → single vector
3. Replicate IC for each space-time point in dataset
4. Targets: SVD coefficients at all space-time points
5. Normalize targets to [-1, 1] during training
6. Train/test split: 80/20

## Key Files

### Training
- **`train_deeponet_supervised.py`**
  - Loads trunk checkpoint
  - Initializes branch network
  - Trains DeepONet end-to-end (branch learns while trunk is frozen)
  - Saves checkpoint: `data/deeponet_supervised_YYYYMMDD_HHMMSS.pth`
  - Outputs: `deeponet_training_curves.png`

### Evaluation
- **`visualize_deeponet_predictions.py`**
  - Loads DeepONet checkpoint (both branch & trunk)
  - Makes batch predictions on all space-time points
  - Plots 1: Ground truth vs predicted SVD coefficients (6×6 grid per time step)
  - Plots 2: Absolute errors (6×3 grid per time step)
  - 5 time instants: t ∈ {0, 0.25, 0.5, 0.75, 1.0}
  - Outputs: 10 PNG files (`deeponet_predictions_t_X.XX.png`, `deeponet_abs_errors_t_X.XX.png`)

## Why This Setup?

1. **Supervised baseline**: Establishes that the architecture can learn without physics constraints
2. **Frozen trunk**: Leverages pre-trained basis functions; reduces training cost
3. **Branch learning**: Discovers how IC affects each SVD coefficient
4. **Element-wise product**: Standard DeepONet interaction; interpretable coefficient modulation

## Next Steps

After validating supervised performance:
1. Add physics-informed loss (PDE residuals in frequency domain)
2. Experiment with relative time weighting
3. Fine-tune or unfreeze trunk if needed
4. Evaluate on out-of-distribution ICs

## Important Details

- **Coordinate matching**: Fortran-order flattening ensures IC indices align with SVD basis
- **Normalization**: Both training and denormalization use stored min/max/range from trunk training
- **Batch inference**: Predictions computed in 2048-point batches for memory efficiency
- **Error metrics**: Both absolute and relative errors computed; visualized separately
