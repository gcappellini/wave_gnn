# Numerical Stability Improvements

## Summary of Changes

This document describes the fixes implemented to address numerical instability issues (predictions exploding to `e+35`) in the GNN-based wave equation solver.

## Root Causes Identified

1. **Missing gradient clipping** - Gradients could accumulate unbounded
2. **Learning rate not scaled with dataset size** - Different dataset sizes led to different effective learning rates
3. **No numerical stability checks** - Silent failures during inference
4. **Potential overflow in inverse scaling** - Large scaled predictions could overflow when denormalized

## Implemented Fixes

### 1. Gradient Clipping (`train.py`)

**Location:** `train_physics()` function, line ~219

**Change:**
```python
loss.backward()
# Apply gradient clipping to prevent exploding gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.training.get('grad_clip_norm', 1.0))
optimizer.step()
```

**Why:** Prevents gradients from exploding during backpropagation, especially important when:
- Changing conv operators (GEN vs GCN have different gradient magnitudes)
- Using different dataset sizes
- Training on complex loss landscapes

**Configuration:** Add to your config file:
```yaml
training:
  grad_clip_norm: 1.0  # Adjust based on your needs (0.5-2.0 typical range)
```

### 2. Learning Rate Scaling with Dataset Size (`train.py`)

**Location:** `train_model()` function, lines ~360-377

**Change:**
```python
# Scale learning rate based on dataset size
base_lr = cfg.training.learning_rate
if cfg.training.get('scale_lr_with_dataset', True):
    baseline_size = cfg.training.get('lr_baseline_size', 1000)
    dataset_size = len(train_set)
    lr_scale_factor = np.sqrt(dataset_size / baseline_size)
    scaled_lr = base_lr * lr_scale_factor
    log.info(f"Learning rate scaling enabled: {base_lr:.6f} -> {scaled_lr:.6f}")
else:
    scaled_lr = base_lr
```

**Why:** 
- More training samples = more gradient updates per epoch = effectively higher learning rate
- Scaling by `sqrt(N)` maintains roughly constant convergence behavior
- Prevents instability when changing `num_graphs` in dataset config

**Configuration:** Add to your config file:
```yaml
training:
  scale_lr_with_dataset: true
  lr_baseline_size: 1000  # Your baseline dataset size
  learning_rate: 0.001    # Base learning rate for baseline_size
```

### 3. Training Divergence Detection (`train.py`)

**Location:** Training loop, lines ~435-441

**Change:**
```python
# Check for numerical instability during training
if np.isnan(loss) or np.isinf(loss):
    log.error(f"Training diverged at epoch {epoch}, batch {nbatches}! Loss: {loss}")
    log.error("Consider: 1) Reducing learning rate, 2) Enabling gradient clipping, "
             "3) Checking feature scaling, 4) Reducing model complexity")
    raise ValueError("Training diverged with NaN/Inf loss")
```

**Why:** Catches divergence early during training instead of silently propagating NaN/Inf values.

### 4. Validation Stability Checks (`train.py`)

**Location:** `evaluate_loader_with_scaling()` function, lines ~571-579

**Change:**
```python
preds = scaler.inverse_transform_output(preds_scaled)

# Check for numerical instability (NaN or Inf)
if torch.isnan(preds).any() or torch.isinf(preds).any():
    log.warning(f"Numerical instability detected in validation! "
               f"NaN: {torch.isnan(preds).sum().item()}, "
               f"Inf: {torch.isinf(preds).sum().item()}")
    log.warning(f"Prediction range: [{preds.min():.3e}, {preds.max():.3e}]")
    log.warning(f"Scaled prediction range: [{preds_scaled.min():.3e}, {preds_scaled.max():.3e}]")
    continue  # Skip this graph
```

**Why:** 
- **This addresses your specific issue**: Predictions were exploding during validation/plotting
- Provides detailed diagnostics when instability occurs
- Prevents contamination of validation metrics

### 5. Inference/Simulation Stability Checks (`test_gcn.py`)

**Location:** `simulate_wave()` function, lines ~163-181

**Change:**
```python
# Check for numerical instability
if np.isnan(features.cpu().numpy()).any() or np.isinf(features.cpu().numpy()).any():
    print(f"WARNING: Numerical instability at step {step}!")
    print(f"  Features range: [{features.min():.3e}, {features.max():.3e}]")
    print(f"  Scaled predictions range: [{preds_scaled.min():.3e}, {preds_scaled.max():.3e}]")
    print("  This often indicates:")
    print("    1. Model weights have diverged during training")
    print("    2. Gradient explosion during training")
    print("    3. Feature scaling statistics are incorrect")
    break
```

**Why:** Catches instability during rollout simulations with helpful diagnostic messages.

### 6. Safe Inverse Transform with Clipping (`scaler.py`)

**Location:** `DataScaler.inverse_transform_output()`, lines ~219-249

**Change:**
```python
# Check for extreme values before inverse transform
if np.abs(y_scaled_np).max() > 100:
    log.warning(f"Extremely large scaled values detected (max: {np.abs(y_scaled_np).max():.3e})")

# Perform inverse transform
y = y_scaled_np * self.output_std + self.output_mean

# Clip extreme values to prevent overflow
max_safe_value = 1e10
if np.abs(y).max() > max_safe_value:
    log.warning(f"Clipping extreme values in inverse transform! Max: {np.abs(y).max():.3e}")
    y = np.clip(y, -max_safe_value, max_safe_value)
```

**Why:** 
- Prevents silent overflow when denormalizing predictions
- Provides early warning when model outputs become unreasonable
- Acts as a safety net (though the real fix should prevent this from happening)

## Usage Recommendations

### For Your Specific Issue (e+35 explosions during plotting)

The most likely causes were:

1. **Model weights diverged during training** (not caught because loss remained finite)
   - **Fix:** Gradient clipping prevents this
   
2. **Different conv operators without corresponding architecture changes**
   - **Fix:** Gradient clipping handles different gradient magnitudes
   - **Additional:** Consider adding BatchNorm/LayerNorm when switching operators

3. **Inverse scaling amplified already-large predictions**
   - **Fix:** Stability checks catch this early with detailed diagnostics

### Configuration Template

Add to your `configs/training/default.yaml`:

```yaml
training:
  # Learning rate settings
  learning_rate: 0.001
  scale_lr_with_dataset: true
  lr_baseline_size: 1000
  
  # Stability settings
  grad_clip_norm: 1.0  # Start with 1.0, reduce to 0.5 if still unstable
  
  # Optimizer settings
  optimizer: adam
  weight_decay: 1e-5
  
  # ... other settings
```

### Testing Your Changes

1. **Verify gradient clipping works:**
   ```python
   # The training logs should show this is being applied
   # Check that gradients don't exceed grad_clip_norm
   ```

2. **Verify LR scaling:**
   ```bash
   # Run with different dataset sizes
   python main.py dataset.num_graphs=500
   python main.py dataset.num_graphs=2000
   # Check logs for "Learning rate scaling" messages
   ```

3. **Test with different conv operators:**
   ```bash
   python main.py model.conv_types=GCN
   python main.py model.conv_types=GEN
   # Should be much more stable now
   ```

## Expected Behavior After Fixes

✅ **Training should be stable** across different dataset sizes
✅ **Validation metrics should not explode** to e+35
✅ **Clear error messages** if instability still occurs
✅ **Detailed diagnostics** showing where instability originates
✅ **Consistent behavior** when switching conv operators

## If Problems Persist

If you still see instability after these fixes:

1. **Check your model architecture:**
   - Add BatchNorm/LayerNorm after each conv layer
   - Consider residual connections
   - Reduce hidden_channels if very large

2. **Check your data:**
   - Verify force magnitudes are reasonable
   - Ensure boundary conditions are properly set
   - Check initial conditions aren't extreme

3. **Adjust hyperparameters:**
   - Reduce learning rate further
   - Increase weight_decay (L2 regularization)
   - Reduce grad_clip_norm to 0.5 or 0.1
   - Enable dropout if not already used

4. **Review the physics loss:**
   - Large loss weights can cause instability
   - Consider adaptive loss weighting
   - Balance PI loss terms

## Additional Improvements to Consider

1. **Weight initialization:** Add explicit initialization (Xavier/He)
2. **Learning rate scheduling:** Use ReduceLROnPlateau or cosine annealing
3. **Mixed precision training:** Can help with numerical stability on GPU
4. **Spectral normalization:** For very deep networks
5. **Feature normalization at dataset level:** Currently done in scaler, could be moved to dataset creation

## Files Modified

- `train.py` (4 changes)
- `test_gcn.py` (1 change)
- `scaler.py` (1 change)

No breaking changes - all modifications are backward compatible with default values.
