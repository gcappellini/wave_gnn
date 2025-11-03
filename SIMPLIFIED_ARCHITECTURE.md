# Simplified Model Architecture System

## Overview
The model loading and training system has been simplified to use a unified configuration-based approach. Instead of maintaining complex parameter lists and compatibility layers, both DeepGCN and WaveGNN now use the same factory pattern that reads directly from Hydra configs.

## Key Changes

### 1. Unified Model Factory (`train.py`)

**New Function:** `create_model_from_config(cfg, device)`

```python
# Old approach (commented out):
model = DeepGCN(
    in_channels=cfg.model.in_channels,
    hidden_channels=cfg.model.hidden_channels,
    out_channels=cfg.model.out_channels,
    conv_types=cfg.model.conv_types,
    # ... 15+ more parameters
)

# New approach:
model = create_model_from_config(cfg, device)
```

**Benefits:**
- ✅ Single line to create any model type
- ✅ Auto-detects model type from config (checks for `hidden_dim` vs `hidden_channels`)
- ✅ No need to pass unused compatibility parameters
- ✅ Config is the single source of truth

### 2. Simplified Checkpoint Loading (`test_gcn.py`)

**New Function:** `create_model_from_checkpoint(ckpt, device)`

```python
# Old approach:
model_cls = detect_model_type()
model_kwargs = extract_kwargs_with_compatibility()
model = model_cls(**model_kwargs)

# New approach:
model = create_model_from_checkpoint(ckpt, device)
```

**Benefits:**
- ✅ Automatically detects and instantiates correct model type
- ✅ Handles all compatibility defaults internally
- ✅ No need to import both model classes
- ✅ Clean, simple API

### 3. Updated `load_best_model()` Signature

```python
# Old:
load_best_model(path, device, model_cls=None, model_kwargs=None)

# New:
load_best_model(path, device=None)
```

**Benefits:**
- ✅ No need to specify model class or kwargs
- ✅ Everything auto-detected from checkpoint
- ✅ Fewer parameters = fewer mistakes

## Model Type Detection

The system uses a simple heuristic:

```python
# WaveGNN indicators:
has 'hidden_dim' AND 'num_layers' AND (missing 'hidden_channels' OR hidden_channels=None)

# DeepGCN indicators:
has 'hidden_channels' (as list or int)
```

## Configuration Files

### WaveGNN Config (`configs/model/wavegnn_model.yaml`)
```yaml
# Required WaveGNN params
hidden_dim: 128
num_layers: 3
dropout: 0.1

# Compatibility params (can be present but ignored)
in_channels: 3
out_channels: 2
hidden_channels: null  # Must be null to trigger WaveGNN
```

### DeepGCN Config (`configs/model/ssh_model.yaml`)
```yaml
# Required DeepGCN params
hidden_channels: [64, 128]
in_channels: 3
out_channels: 2
conv_types: ["GEN", "GEN"]
# ... other DeepGCN-specific params
```

## Usage Examples

### Training
```bash
# Train with WaveGNN
python main.py model=wavegnn_model

# Train with DeepGCN
python main.py model=ssh_model
```

### Testing
```bash
# Auto-detects model type from checkpoint
python test_gcn.py
```

### Programmatic Usage
```python
from train import create_model_from_config
from test_gcn import load_best_model, create_model_from_checkpoint

# Create new model from config
model = create_model_from_config(cfg, device)

# Load trained model
model, ckpt = load_best_model('./best_model.pt')

# Or manually from checkpoint dict
ckpt = torch.load('./best_model.pt')
model = create_model_from_checkpoint(ckpt, device)
```

## Backward Compatibility

The system handles old checkpoints gracefully:

1. **Missing `model_config`**: Uses sensible defaults for DeepGCN
2. **Missing skip connection params**: Adds `use_ed_skip=False`, `ed_skip_type='concat'`
3. **Unexpected keys**: Loads with `strict=False` and warns about mismatches

## What Was Removed

### From `train.py`:
- ❌ 25+ lines of explicit DeepGCN instantiation
- ❌ `create_wavegnn_from_config()` import (replaced with direct WaveGNN import)
- ❌ Manual parameter extraction from config

### From `test_gcn.py`:
- ❌ Complex model class detection logic
- ❌ `model_cls` and `model_kwargs` parameters
- ❌ ~80 lines of the old `load_best_model_old()` function
- ❌ Separate imports with try/except blocks

### From WaveGNN:
- ❌ No need for fake DeepGCN compatibility attributes (`bc_mask`, `in_channels`, etc.)
- ❌ No need to accept `**kwargs` for unused parameters
- ❌ Cleaner, purpose-built constructor

## Error Messages

The new system provides clear diagnostic output:

```
Created WaveGNN model (hidden_dim=128, num_layers=3)
✓ Loaded WaveGNN (hidden_dim=128, num_layers=3)
```

or

```
Created DeepGCN model (hidden_channels=[64, 128])
✓ Loaded DeepGCN (hidden_channels=[64, 128])
```

## Migration Guide

If you have existing code that uses the old API:

### Before:
```python
from try_gno import create_wavegnn_from_config
model = create_wavegnn_from_config(cfg).to(device)
```

### After:
```python
from train import create_model_from_config
model = create_model_from_config(cfg, device)
```

### Before:
```python
from dataset import DeepGCN
model, ckpt = load_best_model(path, device, model_cls=DeepGCN, model_kwargs={...})
```

### After:
```python
model, ckpt = load_best_model(path, device)
```

## Summary

The simplified architecture removes ~150 lines of boilerplate code and eliminates the need for compatibility parameters. The config file is now the single source of truth for model architecture, making the codebase more maintainable and less error-prone.
