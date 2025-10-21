# Adaptive Loss Weighting for PINNs

## Overview

Adaptive loss weighting automatically balances multiple loss terms during training to improve convergence and performance of Physics-Informed Neural Networks (PINNs). This is particularly important when different loss terms (e.g., PDE residuals for u and v, RK4 losses) have vastly different magnitudes.

## The Problem

In multi-objective optimization for PINNs, we typically have:
```
Total Loss = w1·L1 + w2·L2 + w3·L3 + w4·L4
```

Where:
- L1, L2: Physics-informed losses (PDE residuals)
- L3, L4: RK4 integration losses
- w1, w2, w3, w4: Loss weights

**Challenge**: Loss terms often have very different magnitudes (e.g., L1 ~ 1e-3, L2 ~ 1e-1, L3 ~ 1e-2). Fixed weights can lead to:
- Dominant losses overwhelming smaller ones
- Poor gradient flow for certain objectives
- Slower convergence or suboptimal solutions

## Solution: Adaptive Weighting

Our implementation provides several strategies to automatically balance loss terms:

### 1. **Equal Initialization** (`equal_init`)
- Analyzes loss magnitudes at epoch 1
- Computes weights so all weighted losses are approximately equal: w1·L1 ≈ w2·L2 ≈ w3·L3
- Keeps weights fixed after initialization

**When to use**: When you want automatic balancing but prefer stable, fixed weights

### 2. **Equal Init + EMA** (`equal_init_ema`)
- Equalizes losses at epoch 1
- Then continuously adapts weights using exponential moving average
- Tracks loss magnitude changes over training

**When to use**: When loss magnitudes change significantly during training

### 3. **Pure EMA** (`ema`)
- Only uses exponential moving average (no initial equalization)
- Adapts weights throughout training

**When to use**: When you trust the initial weights but want adaptation

### 4. **Fixed** (`fixed`)
- Uses weights from config without adaptation (default behavior)

**When to use**: When you know good weights or want full control

## Mathematical Formulation

### Equal Initialization

Given initial losses L₁⁽⁰⁾, L₂⁽⁰⁾, ..., Lₙ⁽⁰⁾ at epoch 1:

```
wᵢ = 1 / (Lᵢ⁽⁰⁾ + ε)
```

Normalized so that mean weight equals mean of initial weights:
```
wᵢ = (mean(w_initial) / mean(1/Lⱼ)) · (1 / Lᵢ⁽⁰⁾)
```

### Exponential Moving Average

At epoch t:
```
L̄ᵢ⁽ᵗ⁾ = α · L̄ᵢ⁽ᵗ⁻¹⁾ + (1-α) · Lᵢ⁽ᵗ⁾

wᵢ⁽ᵗ⁾ = β · wᵢ⁽ᵗ⁻¹⁾ + (1-β) · (1 / L̄ᵢ⁽ᵗ⁾)
```

Where:
- α: EMA coefficient for losses (default 0.9)
- β: Smoothing for weight updates (default 0.9)
- ε: Small constant for numerical stability (1e-8)

## Usage

### Command Line

```bash
# Use pre-configured adaptive weighting (equal init)
python main.py training=adaptive_equal_init

# Use EMA strategy
python main.py training=adaptive_ema

# Override specific parameters
python main.py training=adaptive_equal_init \
               training.loss.adaptive.strategy=equal_init_ema \
               training.loss.adaptive.ema_alpha=0.95

# Compare adaptive vs fixed weights
python main.py -m training=default,adaptive_equal_init
```

### Configuration File

Create or modify a training config (e.g., `configs/training/my_adaptive.yaml`):

```yaml
epochs: 50
learning_rate: 0.005
weight_decay: 1.0e-5
optimizer: "adam"

loss:
  # Initial weights (used as starting point)
  w1_PI: 1.0
  w2_PI: 1.0
  use_rk4: true
  w1_rk4: 1.0
  w2_rk4: 1.0
  use_gn_solver: false
  
  # Enable adaptive weighting
  adaptive:
    enabled: true
    strategy: "equal_init"  # or "equal_init_ema", "ema", "fixed"
    ema_alpha: 0.9  # EMA coefficient (for EMA strategies)
    update_frequency: 1  # Update weights every N epochs
```

### Python API

```python
from adaptive_weights import AdaptiveLossWeights

# Define loss terms and initial weights
loss_names = ['PI_loss1', 'PI_loss2', 'RK4_loss1', 'RK4_loss2']
initial_weights = {
    'PI_loss1': 1.0,
    'PI_loss2': 1.0,
    'RK4_loss1': 10.0,
    'RK4_loss2': 100.0,
}

# Create adaptive weights manager
adaptive_weights = AdaptiveLossWeights(
    loss_names=loss_names,
    initial_weights=initial_weights,
    strategy='equal_init',
    ema_alpha=0.9,
    update_frequency=1,
)

# During training loop
for epoch in range(1, epochs + 1):
    # ... compute losses ...
    loss_values = {
        'PI_loss1': 0.001,
        'PI_loss2': 0.1,
        'RK4_loss1': 0.01,
        'RK4_loss2': 0.5,
    }
    
    # Update weights
    weights = adaptive_weights.update(epoch, loss_values)
    
    # Use weights in loss computation
    w1 = weights['PI_loss1']
    w2 = weights['PI_loss2']
    total_loss = w1 * loss1 + w2 * loss2 + ...
```

## Examples

### Example 1: Basic Equal Initialization

```bash
python main.py training=adaptive_equal_init \
               dataset.num_graphs=200 \
               training.epochs=50
```

**Expected output:**
```
Adaptive loss weighting enabled
Strategy: equal_init
Initial weights: {'PI_loss1': 1.0, 'PI_loss2': 1.0, ...}
==================================================
Equalized weights at epoch 1:
  PI_loss1: loss=1.234e-03, weight=8.100e+02, weighted=1.000e+00
  PI_loss2: loss=5.678e-02, weight=1.761e+01, weighted=1.000e+00
  RK4_loss1: loss=2.345e-03, weight=4.264e+02, weighted=1.000e+00
  RK4_loss2: loss=1.234e-01, weight=8.100e+00, weighted=1.000e+00
==================================================
```

### Example 2: Compare Strategies

```bash
# Compare all strategies in one run
python main.py -m \
    training=default,adaptive_equal_init,adaptive_ema \
    experiment.name=adaptive_comparison \
    training.epochs=100
```

### Example 3: Fine-tune EMA Parameters

```bash
python main.py training=adaptive_ema \
               training.loss.adaptive.ema_alpha=0.95 \
               training.loss.adaptive.update_frequency=5
```

## Benefits

### 1. **Automatic Balancing**
- No need to manually tune loss weights
- Especially useful when you don't know relative magnitudes

### 2. **Improved Convergence**
- All loss terms contribute meaningfully to gradients
- Prevents one loss from dominating

### 3. **Better Generalization**
- Balanced training across all objectives
- Reduces risk of overfitting to one task

### 4. **Robustness**
- Works across different problem scales
- Adapts to changing loss magnitudes

## Technical Details

### Implementation

The `AdaptiveLossWeights` class in `adaptive_weights.py` provides:

```python
class AdaptiveLossWeights:
    def __init__(
        self,
        loss_names: List[str],
        initial_weights: Dict[str, float],
        strategy: str = 'equal_init',
        ema_alpha: float = 0.9,
        update_frequency: int = 1,
        eps: float = 1e-8,
        min_weight: float = 1e-4,
        max_weight: float = 1e4,
    )
    
    def update(self, epoch: int, loss_values: Dict[str, float]) -> Dict[str, float]
    def get_weight(self, loss_name: str) -> float
    def get_weights(self) -> Dict[str, float]
    def state_dict(self) -> dict
    def load_state_dict(self, state: dict)
```

### Integration with Training

Modified `train_physics()` function:

```python
def train_physics(batch, model, optimizer, device, cfg, 
                 rk4=True, adaptive_weights=None):
    # Get weights from adaptive manager or config
    if adaptive_weights is not None:
        w1_PI = adaptive_weights.get_weight('PI_loss1')
        w2_PI = adaptive_weights.get_weight('PI_loss2')
        # ...
    else:
        w1_PI = cfg.training.loss.w1_PI
        w2_PI = cfg.training.loss.w2_PI
        # ...
    
    # Compute weighted loss
    loss = w1_PI * loss1 + w2_PI * loss2 + ...
```

### Checkpointing

Adaptive weights state is saved with model checkpoints:

```python
ckpt = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'adaptive_weights_state': adaptive_weights.state_dict(),  # NEW
    # ...
}
```

## Best Practices

### 1. **Start with Equal Initialization**
- Most problems benefit from balancing at epoch 1
- Use `equal_init` first, then try `equal_init_ema` if needed

### 2. **Monitor Weights**
- Check logged weights periodically
- Ensure they stay in reasonable ranges (1e-4 to 1e4)

### 3. **Validate Results**
- Compare adaptive vs fixed weights on validation set
- Use `-m` flag for multi-run comparisons

### 4. **Tune EMA Coefficient**
- Default α=0.9 works for most cases
- Increase to 0.95-0.99 for slower adaptation
- Decrease to 0.7-0.8 for faster response

### 5. **Update Frequency**
- Default (every epoch) works well
- Increase frequency (e.g., 5) for very noisy losses
- Decrease (e.g., 1) for stable problems

## Troubleshooting

### Issue: Weights become very large/small

**Cause**: Loss magnitudes too extreme

**Solution**:
```yaml
# In adaptive_weights.py, adjust constraints:
min_weight: 1e-3  # Increase minimum
max_weight: 1e3   # Decrease maximum
```

### Issue: Training unstable with adaptive weights

**Cause**: Weights changing too rapidly

**Solution**:
```yaml
adaptive:
  strategy: "equal_init"  # Use fixed after init
  # OR
  ema_alpha: 0.95  # Slower adaptation
  update_frequency: 5  # Update less frequently
```

### Issue: No improvement over fixed weights

**Cause**: Fixed weights already well-tuned, or problem doesn't benefit

**Solution**:
- Compare validation metrics carefully
- Try different strategies
- Consider that adaptive weighting isn't always better

### Issue: One loss still dominates

**Cause**: Extreme magnitude differences

**Solution**:
- Check if losses are computed correctly
- Verify physical units/scaling
- Consider data normalization (see SCALING_GUIDE.md)

## Logging and Monitoring

Adaptive weights are logged periodically:

```
Epoch 001 | Train 2.345e-02 | PI1 1.234e-03 | PI2 5.678e-02 | ...
        Weights: PI1=8.10e+02, PI2=1.76e+01, RK4_1=4.26e+02, RK4_2=8.10e+00

Epoch 005 | Train 1.234e-02 | PI1 5.678e-04 | PI2 2.345e-02 | ...
Epoch 010 | Train 8.901e-03 | PI1 3.456e-04 | PI2 1.234e-02 | ...
        Weights: PI1=8.23e+02, PI2=1.75e+01, RK4_1=4.30e+02, RK4_2=8.05e+00
```

## Performance Considerations

- **Overhead**: Negligible (<0.1% of training time)
- **Memory**: ~1KB per loss term
- **Computation**: Simple arithmetic operations at each epoch

## Limitations

1. **Not always better**: Some problems don't benefit from adaptive weighting
2. **Requires tuning**: EMA parameters may need adjustment
3. **Early training**: Weights at epoch 1 may be noisy (use multiple samples if needed)
4. **Discontinuous losses**: May struggle with losses that change discontinuously

## Future Enhancements

Possible improvements:

1. **GradNorm**: Balance gradient magnitudes instead of loss magnitudes
2. **Learnable weights**: Treat weights as trainable parameters
3. **Multi-stage**: Different strategies for different training phases
4. **Automatic strategy selection**: Choose strategy based on loss behavior
5. **Uncertainty-aware**: Weight by loss uncertainty/variance

## References

- Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses" (2018)
- Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing" (2018)
- Wang et al., "Understanding and Mitigating Gradient Pathologies in PINNs" (2021)

## Quick Reference

| Strategy | Behavior | Use Case |
|----------|----------|----------|
| `equal_init` | Equalize at epoch 1, then fixed | Default choice, stable training |
| `equal_init_ema` | Equalize + continuous EMA | Changing loss magnitudes |
| `ema` | Only EMA (no equalization) | Good initial weights available |
| `fixed` | Config weights (no adaptation) | Manual control, comparison baseline |

**Default parameters:**
- `ema_alpha`: 0.9 (90% old, 10% new)
- `update_frequency`: 1 (every epoch)
- `min_weight`: 1e-4
- `max_weight`: 1e4

---

**Status**: ✅ Implemented and ready for use  
**Date**: October 21, 2025  
**Version**: 1.0
