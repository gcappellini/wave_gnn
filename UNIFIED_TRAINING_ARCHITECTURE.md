# Unified Training Architecture

## Overview

This refactoring introduces a **clean separation of concerns** between model definition and training logic, using a **scheduled loss weighting approach** to replace the complex phase-based training system.

### Key Changes

#### 1. **train_unified.py** (NEW)
- Contains `UnifiedTrainer` class handling all training logic
- Single configurable training loop with loss weight scheduling
- No need for separate phase1, phase1b, phase2, phase3 methods
- Cleaner, more maintainable code

#### 2. **main_unified.py** (NEW)
- Simplified main entry point (40 lines vs 480+ lines)
- Clear data pipeline: Load → Create Model → Load Checkpoint → Train → Validate → Save
- No complex case-by-case logic
- Easier to understand and debug

#### 3. **config_setup.yaml** (UPDATED)
- New `loss_schedule` section with configurable stages
- Each stage has:
  - `name`: Stage identifier
  - `epochs`: Number of epochs for this stage
  - `weights`: Loss weights for IC_u, IC_v, PDE
  - `lr`: Learning rate for this stage

#### 4. **model_2d.py** (UNCHANGED)
- Remains focused on model forward pass, loss computation
- Old phase1/phase2 training methods can be deprecated (kept for backward compatibility)

---

## How It Works

### Loss Schedule Configuration

```yaml
loss_schedule:
  enabled: true
  stages:
    - name: "IC_u_pretraining"
      epochs: 500
      weights:
        w_ic_u: 1.0
        w_ic_v: 0.0
        w_pde: 0.0
      lr: 1.0e-3
      
    - name: "IC_v_training"
      epochs: 300
      weights:
        w_ic_u: 1.0
        w_ic_v: 1.0
        w_pde: 0.0
      lr: 5.0e-4
      
    - name: "PDE_training"
      epochs: 500
      weights:
        w_ic_u: 1.0
        w_ic_v: 1.0
        w_pde: 1.0
      lr: 1.0e-4
```

### Execution Flow

1. **Stage 1: IC_u Pretraining** (500 epochs)
   - Only optimize IC_u loss (u₀ initial condition)
   - PDE and IC_v weights set to 0
   - Higher learning rate (1e-3)

2. **Stage 2: IC_v Training** (300 epochs)
   - Optimize both IC_u and IC_v
   - Still no PDE loss
   - Lower learning rate (5e-4)

3. **Stage 3: PDE Training** (500 epochs)
   - Full loss: IC_u + IC_v + PDE
   - Lowest learning rate (1e-4)
   - Fine-tune the full model

### Advantages Over Previous Approach

| Aspect | Old (Phase-based) | New (Scheduled) |
|--------|------------------|-----------------|
| **Code Complexity** | 480+ lines (Cases 1-4) | ~100 lines |
| **Training Logic** | Scattered across methods | Single `train()` method |
| **Configurability** | Hard-coded phases | YAML-configured stages |
| **Extensibility** | Add new phase = modify code | Add stage = modify config |
| **Reproducibility** | Phase info implicit | Schedule saved in pickle |
| **Learning Rates** | Per-phase hardcoded | Per-stage configurable |
| **Loss Weights** | Phase-dependent logic | Explicit weights in config |

---

## Usage

### Basic Training from Scratch

```bash
python main_unified.py training.loss_schedule.enabled=true
```

### Load Pretrained Model and Continue

```bash
python main_unified.py \
  run.load_pretrain=true \
  run.load_pretrain_from=[2025-12-16,12-27-31] \
  training.loss_schedule.stages[0].epochs=0
```

The last argument starts with stage 2 (IC_v_training) by setting stage 1 epochs to 0.

### Custom Schedule

```bash
python main_unified.py \
  training.loss_schedule.stages[0].epochs=1000 \
  training.loss_schedule.stages[1].lr=2e-4 \
  training.loss_schedule.stages[2].weights.w_pde=0.5
```

---

## Training History

The unified trainer saves complete training history:

```python
{
    'history': {
        'epoch': [...],
        'stage': [...],
        'loss_total': [...],
        'loss_ic_u': [...],
        'loss_ic_v': [...],
        'loss_pde': [...],
        'w_ic_u': [...],  # Actual weights per epoch
        'w_ic_v': [...],
        'w_pde': [...],
        'lr': [...],      # Actual LR per epoch
        'val_epochs': [...],
        'val_loss_ic_u': [...],
        'val_loss_ic_v': [...],
        'val_loss_pde': [...]
    },
    'schedule': [  # Full config for reproducibility
        {
            'name': 'IC_u_pretraining',
            'epochs': 500,
            'weights': {'w_ic_u': 1.0, ...},
            'lr': 1e-3
        },
        ...
    ],
    'best_metric': 0.00123
}
```

This allows complete reproducibility and analysis of training dynamics.

---

## Migration from Old Code

### Old Way (Phase-based)
```python
# Multiple cases, multiple training methods
if load_pretrain and not continue_phase1_lbfgs:
    # Case 1a: load + train Phase 2
elif continue_phase1_lbfgs:
    # Case 1b: load + continue LBFGS
# ... etc 4+ cases
```

### New Way (Schedule-based)
```python
# Single unified pipeline
trainer = UnifiedTrainer(model, cfg, device)
history = trainer.train(test_cases, output_fold)
```

---

## Next Steps

1. ✅ Created `train_unified.py` with `UnifiedTrainer` class
2. ✅ Created `main_unified.py` as simplified entry point
3. ✅ Updated `config_setup.yaml` with `loss_schedule`
4. **TODO**: Test the unified training on a small example
5. **TODO**: Gradually migrate old code (keep backward compatibility)
6. **TODO**: Add advanced features (e.g., dynamic weight adjustment, multi-task learning)

---

## Questions for Implementation

- Should we keep old training methods in model_2d.py for backward compatibility?
- Do you want validation logic inside UnifiedTrainer or separate?
- Should we add visualization of loss schedule convergence?
- Any additional metrics or callbacks needed?
