# Spectral Graph Neural Networks for Wave Equation

This directory implements three phases of global communication in Graph Neural Networks (GNNs) for wave equation prediction.

## Overview

### Phase 1: Global Node Broadcast (Baseline)
**File:** `try_gno.py` - `WaveGNN` class

- Uses a virtual **global node** that aggregates information from all nodes
- Broadcasts global information back to all nodes
- Simple but doesn't leverage graph structure for global communication

**Architecture:**
```
Lifting → Global MP Layers → Projection → Physics
         (local edges + global node)
```

### Phase 2: Pure Spectral Model
**File:** `spectral_models.py` - `SpectralWaveGNN` class

- Replaces global node with **Fourier transform via Laplacian eigenbasis**
- Uses graph structure (encoded in Laplacian eigenvectors) for principled global communication
- Processes different frequency modes separately

**Architecture:**
```
Lifting → Spectral Layers → Projection → Physics

Spectral Layer:
  h_spatial = node features in spatial domain
  ↓
  h_spectral = U^T @ h_spatial    (transform to frequency domain)
  ↓
  h_filtered = Learnable_Filter(h_spectral)  (process each mode)
  ↓
  h_spatial_new = U @ h_filtered   (transform back to spatial)
```

**Key Idea:**
- U = Laplacian eigenvectors (computed once in `dataset.py`)
- Low eigenvalues → global, smooth patterns (long-range)
- High eigenvalues → local, detailed patterns (short-range)

### Phase 3: Hybrid Spatial-Spectral Model
**File:** `spectral_models.py` - `HybridWaveGNN` class

- Combines **local message passing** (spatial) with **spectral processing**
- Best of both worlds: local structure + global patterns

**Architecture:**
```
Lifting → Parallel Branches → Combine → Projection → Physics

Branch 1 (Spatial): Local message passing layers
  - Captures local interactions via graph edges
  - Traditional GNN approach

Branch 2 (Spectral): Fourier layers
  - Captures global patterns via Laplacian eigenbasis
  - Spectral graph convolutions

Combine:
  h_final = MLP(concat[h_spatial, h_spectral]) + residual
```

## Files

- **`try_gno.py`**: Phase 1 implementation (global node baseline)
- **`spectral_layers.py`**: Core spectral convolution layers
  - `SpectralConv`: Basic spectral convolution
  - `SpectralLayer`: Spectral conv + normalization + activation
  - `SpectralMessagePassing`: Complete spectral layer with residual
  - `MultiScaleSpectralLayer`: Multi-frequency processing
  
- **`spectral_models.py`**: Phase 2 and Phase 3 implementations
  - `SpectralWaveGNN`: Pure spectral model (Phase 2)
  - `HybridWaveGNN`: Hybrid spatial-spectral model (Phase 3)
  - `create_spectral_model()`: Factory function
  
- **`dataset.py`**: Dataset creation with eigendecomposition
  - `compute_laplacian_eigenbasis()`: Computes U (eigenvectors) and λ (eigenvalues)
  - `create_graph()`: Creates graph with precomputed eigenbasis
  
- **`test_phases.py`**: Demo script comparing all three phases

## Eigendecomposition

The Laplacian eigendecomposition is computed **once during graph creation** for efficiency:

```python
# In dataset.py - create_graph()
eigenvalues, eigenvectors = compute_laplacian_eigenbasis(
    L,  # Laplacian matrix
    num_modes=50,  # Number of modes to use (default: N/2)
    normalized=True  # Use normalized Laplacian
)

# Store in graph data
data = Data(
    x=x, 
    edge_index=edge_index,
    eigenvectors=eigenvectors,  # [N, num_modes]
    eigenvalues=eigenvalues,    # [num_modes]
    ...
)
```

The eigenvectors are then passed to the model during forward pass:

```python
# Phase 2 or 3
output = model(x, edge_index, bc_mask, eigenvectors=data.eigenvectors)
```

## Usage

### Training with Different Phases

```python
from try_gno import WaveGNN
from spectral_models import SpectralWaveGNN, HybridWaveGNN
from dataset import create_graph

# Create graph with eigendecomposition
cfg = ...  # Your config
data = create_graph(cfg=cfg)

# Phase 1: Global node
model = WaveGNN(cfg)
output = model(data.x, data.edge_index, data.bc_mask)

# Phase 2: Pure spectral
model = SpectralWaveGNN(cfg)
output = model(data.x, bc_mask=data.bc_mask, eigenvectors=data.eigenvectors)

# Phase 3: Hybrid
model = HybridWaveGNN(cfg)
output = model(data.x, data.edge_index, data.bc_mask, eigenvectors=data.eigenvectors)
```

### Configuration

Add to your config file:

```yaml
model:
  hidden_dim: 128
  num_layers: 4
  dropout: 0.1
  num_spectral_modes: 50  # Number of Fourier modes to use
  spatial_weight: 0.5     # Weight for spatial branch (Phase 3 only)
  spectral_weight: 0.5    # Weight for spectral branch (Phase 3 only)

dataset:
  dt: 0.01
  u_scale: 0.04
  v_scale: 0.08
  f_scale: 3.0
```

### Testing

Run the demo script to see all three phases in action:

```bash
python test_phases.py
```

This will:
1. Create a graph with eigendecomposition
2. Initialize all three models
3. Run forward passes
4. Compare outputs
5. Print parameter counts and architecture details

## Theory: Spectral Graph Convolutions

### Why Laplacian Eigenvectors?

The graph Laplacian **L** encodes the graph structure. Its eigendecomposition:
```
L @ U = U @ Λ
```

gives us:
- **U**: Eigenvector matrix (Fourier basis on the graph)
- **Λ**: Diagonal matrix of eigenvalues (frequencies)

**Interpretation:**
- Small eigenvalues → smooth eigenvectors → global patterns
- Large eigenvalues → oscillating eigenvectors → local details

### Spectral Convolution

A convolution in the graph frequency domain:
```
h_out = U @ g(Λ) @ U^T @ h_in
```

where `g(Λ)` is a learnable filter function.

In our implementation:
```python
# Transform to frequency domain
h_spectral = U^T @ h_in  # [num_modes, features]

# Apply learnable filter (element-wise in frequency)
h_filtered = learnable_weights * h_spectral  # [num_modes, features]

# Transform back to spatial domain
h_out = U @ h_filtered  # [N, features]
```

### Advantages

1. **Principled global communication**: Based on graph structure (Laplacian)
2. **Multi-scale processing**: Different frequencies = different scales
3. **Efficient for global patterns**: O(N * k) where k = num_modes << N
4. **Interpretable**: Low frequencies = smooth/global, high = local/detailed

## Performance Comparison

Expected characteristics:

| Phase | Local Structure | Global Patterns | Parameters | Speed |
|-------|----------------|-----------------|------------|-------|
| 1: Global Node | ✓ (via edges) | ✓ (via broadcast) | Medium | Fast |
| 2: Pure Spectral | ✗ | ✓✓ (principled) | Medium | Medium |
| 3: Hybrid | ✓✓ | ✓✓ | High | Slower |

**Phase 1** is the simplest baseline.

**Phase 2** should excel when global communication is critical and the graph structure contains important frequency information.

**Phase 3** should perform best overall by combining local and global processing.

## Future Extensions

1. **Adaptive mode selection**: Learn which frequency modes are important
2. **Multi-scale spectral layers**: Process low/high frequencies differently
3. **Graph coarsening**: Hierarchical spectral processing
4. **Attention over modes**: Weighted combination of frequency components
5. **Time-dependent eigenbasis**: Update eigenvectors for dynamic graphs

## References

- Bruna et al. (2013): "Spectral Networks and Deep Locally Connected Networks"
- Defferrard et al. (2016): "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering"
- Kipf & Welling (2017): "Semi-Supervised Classification with Graph Convolutional Networks"
- Li et al. (2020): "Fourier Neural Operator for Parametric Partial Differential Equations"

## Citation

If you use this code, please cite:
```bibtex
@software{spectral_wave_gnn,
  title={Spectral Graph Neural Networks for Wave Equation Prediction},
  author={Your Name},
  year={2025}
}
```
