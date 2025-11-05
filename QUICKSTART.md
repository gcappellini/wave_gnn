# Quick Start Guide: Spectral GNN Phases

## Installation

Make sure you have the required packages:
```bash
pip install torch torch-geometric scipy numpy omegaconf
```

## Quick Integration into Existing Training

### Option 1: Replace model creation in your training script

**Before (Phase 1 - Global Node):**
```python
from try_gno import WaveGNN

model = WaveGNN(cfg)
output = model(data.x, data.edge_index, data.bc_mask)
```

**After - Phase 2 (Pure Spectral):**
```python
from spectral_models import SpectralWaveGNN

model = SpectralWaveGNN(cfg)
output = model(data.x, bc_mask=data.bc_mask, eigenvectors=data.eigenvectors)
```

**After - Phase 3 (Hybrid):**
```python
from spectral_models import HybridWaveGNN

model = HybridWaveGNN(cfg)
output = model(data.x, data.edge_index, data.bc_mask, eigenvectors=data.eigenvectors)
```

### Option 2: Use the factory function

```python
from spectral_models import create_spectral_model

# Choose phase
model = create_spectral_model(cfg, phase='hybrid')  # or 'spectral'
output = model(data.x, data.edge_index, data.bc_mask, eigenvectors=data.eigenvectors)
```

## Configuration Updates

Add these to your config YAML:

```yaml
model:
  # Existing parameters
  hidden_dim: 128
  num_layers: 4
  dropout: 0.1
  
  # NEW: Spectral parameters
  num_spectral_modes: 50  # Number of Laplacian eigenvectors to use
                          # Recommended: N/2 to N/4 where N = number of nodes
  
  # NEW: For Phase 3 (Hybrid) only
  spatial_weight: 0.5     # Weight for spatial branch
  spectral_weight: 0.5    # Weight for spectral branch
```

## What Changed in dataset.py

The `create_graph()` function now computes eigendecomposition:

```python
# New: Compute Laplacian eigenbasis (done ONCE at graph creation)
num_spectral_modes = getattr(cfg.model, 'num_spectral_modes', num_nodes // 2)
eigenvalues, eigenvectors = compute_laplacian_eigenbasis(
    L, 
    num_modes=num_spectral_modes,
    normalized=True
)

# Store in graph data
data = Data(
    x=x,
    edge_index=edge_index,
    bc_mask=bc_mask,
    eigenvalues=eigenvalues,    # NEW
    eigenvectors=eigenvectors,  # NEW
    ...
)
```

This is efficient because:
- Eigendecomposition is computed **once** when creating the graph
- Stored in the graph data object
- Reused in every forward pass (no recomputation)

## Training Script Modifications

### Minimal changes needed:

```python
# Your existing training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass - just add eigenvectors parameter
        if isinstance(model, (SpectralWaveGNN, HybridWaveGNN)):
            output = model(
                batch.x, 
                batch.edge_index, 
                batch.bc_mask,
                eigenvectors=batch.eigenvectors  # NEW
            )
        else:
            # Phase 1 (original)
            output = model(batch.x, batch.edge_index, batch.bc_mask)
        
        # Rest is the same
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

## Testing Your Setup

Run the test script to verify everything works:

```bash
python test_phases.py
```

This will:
1. Create a test graph with eigendecomposition
2. Initialize all three phase models
3. Run forward passes
4. Compare outputs and show architecture details

## Choosing the Right Phase

| Phase | Use When | Pros | Cons |
|-------|----------|------|------|
| **Phase 1** | Baseline | Simple, fast | No graph structure in global comm |
| **Phase 2** | Global patterns matter | Principled, interpretable | No local edge info |
| **Phase 3** | Best performance | Local + global | More parameters, slower |

**Recommendation:** Start with Phase 3 (hybrid) for best results.

## Common Issues

### 1. "eigenvectors must be provided"
Make sure your graph data has the eigenvector matrix:
```python
data = create_graph(cfg=cfg)
assert hasattr(data, 'eigenvectors'), "Graph missing eigenvectors!"
```

### 2. "num_spectral_modes not in config"
Add to your config:
```yaml
model:
  num_spectral_modes: 50
```

### 3. Shape mismatch errors
Ensure `num_spectral_modes <= num_nodes` and that your graph has the eigendecomposition:
```python
print(f"Nodes: {data.x.shape[0]}")
print(f"Eigenvectors: {data.eigenvectors.shape}")
print(f"Modes used: {model.num_spectral_modes}")
```

## Performance Tips

1. **Number of modes**: Start with `num_spectral_modes = N // 2`
   - Too few: might miss important frequencies
   - Too many: slower, potentially overfit
   
2. **Memory**: Eigenvector matrix is `[N, K]` floats
   - For 1000 nodes, 500 modes = 2MB
   - Usually not a problem
   
3. **Speed**: Spectral layers are matrix multiplications
   - Phase 2: ~same speed as Phase 1
   - Phase 3: ~2x slower (dual branches)
   
4. **GPU**: Everything works on GPU automatically
   ```python
   model = model.to('cuda')
   data.eigenvectors = data.eigenvectors.to('cuda')
   ```

## Next Steps

1. **Experiment with hyperparameters**: Try different `num_spectral_modes`
2. **Visualize frequency modes**: Plot eigenvectors to see patterns
3. **Compare phases**: Train all three and compare on validation set
4. **Multi-scale**: Modify to use different frequency bands separately

## Questions?

See `SPECTRAL_README.md` for detailed theory and implementation notes.
