# WaveGNN Integration Guide

## Quick Start: Drop-in Replacement for DeepGCN

### Step 1: Import the Model

In `train.py`, add at the top:
```python
from try_gno import WaveGNN, create_wavegnn_from_config
```

### Step 2: Replace Model Instantiation

Find this block (~line 433):
```python
model = DeepGCN(
    in_channels=cfg.model.in_channels,
    hidden_channels=cfg.model.hidden_channels,
    ...
).to(device)
```

Replace with:
```python
# Option 1: Use config factory (recommended)
model = create_wavegnn_from_config(cfg).to(device)

# Option 2: Manual instantiation
model = WaveGNN(
    hidden_dim=128,
    num_layers=3,
    dt=cfg.dataset.dt,
    dropout=cfg.model.dropout,
    in_channels=3,
    out_channels=2
).to(device)
```

### Step 3: Update Config

Change the model config in your YAML:
```yaml
defaults:
  - model: wavegnn_model  # Instead of ssh_model or fast_model
```

Or in `configs/config.yaml`:
```yaml
defaults:
  - dataset: ssh_dataset
  - model: wavegnn_model  # <-- Change this
  - training: ssh_training
```

### Step 4: Run Training

```bash
python main.py
```

That's it! No other changes needed.

---

## Interface Compatibility

### ✅ COMPATIBLE - No Changes Needed

1. **Forward Pass Signature**
   - DeepGCN: `model(x, edge_index, bc_mask)` → `[N, 2]`
   - WaveGNN: `model(x, edge_index, bc_mask)` → `[N, 2]`
   - ✅ Same interface

2. **Input Format**
   - DeepGCN: `x` is `[N, 3]` tensor with `[u, v, f]`
   - WaveGNN: `x` is `[N, 3]` tensor with `[u, v, f]`
   - ✅ Same format

3. **Output Format**
   - DeepGCN: Returns `[N, 2]` with `[u_next, v_next]`
   - WaveGNN: Returns `[N, 2]` with `[u_next, v_next]`
   - ✅ Same format

4. **Boundary Conditions**
   - DeepGCN: Uses `model.bc_mask` and `bc_mask` argument
   - WaveGNN: Uses `model.bc_mask` and `bc_mask` argument
   - ✅ Same behavior

5. **Loss Functions**
   - Both use same physics-informed losses
   - Both output `[u_next, v_next]` for loss computation
   - ✅ Compatible

6. **Training Loop**
   - Both use `train_physics()` function
   - Both iterate over batches the same way
   - ✅ No changes needed

---

## Architecture Differences

### WaveGNN Features

1. **Global Message Passing**
   - Each node communicates with neighbors AND a global node
   - Captures long-range interactions better than local GCN

2. **Physics-Informed Integration**
   - Predicts velocity change `dv` instead of absolute values
   - Uses trapezoidal rule for time integration
   - More physically accurate than direct prediction

3. **Normalization**
   - Built-in feature normalization (u_scale, v_scale, f_scale)
   - Helps with training stability

4. **Simpler Architecture**
   - No skip connections, batch norm, or complex GCN variants
   - Just: Lift → Global Message Passing → Project → Physics Integration

### Parameter Count Comparison

- **DeepGCN** (2 layers, [64, 128]): ~40K parameters
- **WaveGNN** (3 layers, 128 hidden): ~150K parameters

---

## Configuration Reference

### WaveGNN-Specific Config

Add these to your dataset config (`configs/dataset/ssh_dataset.yaml`):
```yaml
# Normalization scales (optional, defaults provided)
u_scale: 0.04  # Max expected displacement
v_scale: 0.08  # Max expected velocity
f_scale: 3.0   # Max expected force
```

Add these to your model config (`configs/model/wavegnn_model.yaml`):
```yaml
hidden_dim: 128   # Hidden dimension for message passing
num_layers: 3     # Number of message passing layers
dropout: 0.1      # Dropout rate
```

### Tuning Recommendations

1. **Start with defaults**: `hidden_dim=128, num_layers=3`
2. **If underfitting**: Increase `hidden_dim` to 256 or `num_layers` to 4-5
3. **If overfitting**: Add more `dropout` (0.1 → 0.2-0.3)
4. **If unstable**: Check normalization scales match your data range

---

## Testing the Integration

Run the test script:
```bash
python try_gno.py
```

Expected output:
```
Model created with 150,000 trainable parameters

============================================================
TEST 1: DeepGCN-compatible interface
============================================================
Input shape: torch.Size([100, 3])
Output shape: torch.Size([100, 2])
Boundary conditions satisfied: u_boundary=0.000000, v_boundary=0.000000

============================================================
TEST 2: Original WaveGNN interface (still supported)
============================================================
...

✓ Model ready for drop-in replacement!
```

---

## Troubleshooting

### Issue: "bc_mask must be provided"
**Solution**: Ensure `model.bc_mask` is set before forward pass:
```python
model.bc_mask = data.bc_mask.to(device)
out_sub = model(data.x.to(device), data.edge_index.to(device), data.bc_mask.to(device))
```

### Issue: Shape mismatch in loss
**Check**: Output shape should be `[N, 2]`, not tuple `(u_next, v_next)`
```python
print(f"Output shape: {output.shape}")  # Should be torch.Size([N, 2])
```

### Issue: Poor performance
**Try**:
1. Increase model capacity: `hidden_dim=256, num_layers=4`
2. Adjust learning rate: May need different LR than DeepGCN
3. Check normalization scales match your data distribution

---

## Next Steps

1. ✅ Test with `python try_gno.py`
2. ✅ Update config to use `wavegnn_model`
3. ✅ Run short training: `python main.py training.epochs=10`
4. ✅ Compare with DeepGCN baseline
5. ✅ Tune hyperparameters if needed

Good luck! 🚀
