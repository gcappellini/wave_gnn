# Encoder-Decoder Architecture with Middle Pooling

This document describes the encoder-decoder architecture enabled by setting `pooling_position='middle'` in the `DeepGCN` class.

## Overview

The encoder-decoder architecture with middle pooling creates an **information bottleneck** by:
1. **Encoding** node features into latent representations
2. **Pooling** to compress information into graph-level features
3. **Broadcasting** graph features back to all nodes
4. **Decoding** to produce final node-level predictions

This forces the model to learn **global patterns** and can enforce **conservation laws** important in physics.

## Architecture Comparison

### Standard Architecture (default)
```
Input [N, in_channels]
  ↓
Layers 1-4 (all process nodes)
  ↓
Output [N, out_channels]
```
Each node's prediction depends on its local neighborhood.

### Encoder-Decoder Architecture (pooling_position='middle')
```
Input [N, in_channels]
  ↓
ENCODER Layers 1-2
  ↓
POOLING: [N, hidden] → [1 or B, graph_dim]
  ↓
BROADCAST: [1 or B, graph_dim] → [N, graph_dim]
  ↓
DECODER Layers 3-4
  ↓
Output [N, out_channels]
```
Each node's prediction incorporates global information from the entire graph.

## Basic Usage

```python
from dataset import DeepGCN, create_graph

# Create encoder-decoder model
model = DeepGCN(
    hidden_channels=[64, 64, 32, 32],
    conv_types="GCN",
    final_layer_type="Linear",
    in_channels=3,
    out_channels=2,
    use_global_pooling=True,
    pooling_position="middle",  # KEY: enables encoder-decoder
    pooling_type="mean",
    encoder_layers=2,  # First 2 layers = encoder
    # Remaining 2 layers = decoder (automatic)
)

# Use like normal
data = create_graph(seed=42)
output = model(data.x, data.edge_index, data.bc_mask)
# Output shape: [num_nodes, 2]
```

## Configuration Parameters

### Required Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `use_global_pooling` | `True` | Must enable pooling |
| `pooling_position` | `"middle"` | Place pooling between encoder and decoder |

### Architecture Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `encoder_layers` | int | `None` | Number of layers in encoder. If `None`, uses `len(hidden_channels) // 2` |
| `decoder_channels` | list[int] | `None` | Hidden dimensions for decoder layers. If `None`, mirrors encoder in reverse |
| `graph_output_dim` | int | `None` | Bottleneck dimension after pooling. If `None`, uses encoder output dim |

### Pooling Strategy

| Parameter | Options | Best For |
|-----------|---------|----------|
| `pooling_type` | `"mean"` | General purpose, stable training |
| | `"max"` | Detecting extreme values |
| | `"sum"` | Conservation laws (total energy, mass) |
| | `"attention"` | Learning importance weights |

## Detailed Examples

### Example 1: Symmetric Encoder-Decoder

Split layers evenly between encoder and decoder:

```python
model = DeepGCN(
    hidden_channels=[128, 64, 32, 16],  # 4 layers
    conv_types="GCN",
    in_channels=3,
    out_channels=2,
    use_global_pooling=True,
    pooling_position="middle",
    encoder_layers=2,  # Encoder: 128 → 64
    # Decoder: 32 → 16 (automatic)
)
```

Architecture:
- **Encoder**: [3] → 128 → 64
- **Pooling**: [N, 64] → [1, 64]
- **Decoder**: [64] → 32 → 16
- **Output**: [16] → [2]

### Example 2: Asymmetric with Custom Decoder

More encoder layers than decoder, custom decoder architecture:

```python
model = DeepGCN(
    hidden_channels=[128, 64, 32],  # 3 layers for encoder
    conv_types=["GCN", "GAT", "SAGE"],
    in_channels=3,
    out_channels=2,
    use_global_pooling=True,
    pooling_position="middle",
    pooling_type="attention",
    encoder_layers=3,  # Use all 3 layers as encoder
    decoder_channels=[64, 32, 16],  # Custom 3-layer decoder
    graph_output_dim=16,  # Small bottleneck
)
```

Architecture:
- **Encoder**: [3] → 128 → 64 → 32 (3 layers)
- **Pooling**: [N, 32] → [1, 16] (with attention)
- **Decoder**: [16] → 64 → 32 → 16 (3 layers)
- **Output**: [16] → [2]

### Example 3: Deep Bottleneck

Aggressive compression for strong regularization:

```python
model = DeepGCN(
    hidden_channels=[256, 128, 64, 32],
    conv_types="GCN",
    in_channels=3,
    out_channels=2,
    use_global_pooling=True,
    pooling_position="middle",
    pooling_type="mean",
    encoder_layers=4,  # All 4 layers as encoder
    decoder_channels=[64, 128, 256],  # Mirror in reverse
    graph_output_dim=8,  # Extreme bottleneck: 32 → 8
)
```

Architecture:
- **Encoder**: [3] → 256 → 128 → 64 → 32
- **Pooling**: [N, 32] → [1, 8] ← **Extreme compression**
- **Decoder**: [8] → 64 → 128 → 256
- **Output**: [256] → [2]

Compression ratio: If N=100, this compresses 100×32=3200 values → 8 values (400x compression!)

### Example 4: Physics-Informed Wave Equation

Use sum pooling to enforce energy conservation:

```python
model = DeepGCN(
    hidden_channels=[128, 64, 32],
    conv_types="GCN",
    in_channels=3,  # u, v, f
    out_channels=2,  # u_next, v_next
    use_global_pooling=True,
    pooling_position="middle",
    pooling_type="sum",  # Sum = total energy
    encoder_layers=2,
    graph_output_dim=4,  # Small bottleneck enforces conservation
)
```

Why this works:
- Sum pooling captures **total energy** across all nodes
- Small bottleneck forces model to **conserve energy** through the bottleneck
- Decoder must **redistribute energy** appropriately to nodes

## Use Cases

### 1. Global Constraints

**Problem**: Your physics system has conservation laws (energy, mass, momentum)

**Solution**: Use encoder-decoder with appropriate pooling:
```python
# Energy conservation
pooling_type="sum"  # Total energy = sum of node energies

# Average properties
pooling_type="mean"  # Average temperature, pressure, etc.
```

### 2. Regularization

**Problem**: Model overfits by memorizing training graphs

**Solution**: Information bottleneck acts as regularization:
```python
graph_output_dim=8  # Small bottleneck forces generalization
```

The model **cannot** memorize all details; it must learn generalizable patterns.

### 3. Global Feature Learning

**Problem**: Need to classify graphs or predict global properties

**Solution**: Encoder-decoder learns useful graph representations:
```python
# After training, extract encoder+pooling for graph classification
graph_features = model.encoder_layers(x)
graph_features = model.global_pooling(graph_features, batch)
# Use graph_features for classification
```

### 4. Multi-Scale Physics

**Problem**: Physics operates at both local and global scales

**Solution**: Encoder captures local patterns, decoder applies global context:
```python
# Encoder: local wave interactions
# Pooling: global wave energy
# Decoder: how global energy affects local behavior
```

## Batched Processing

The encoder-decoder architecture handles batched graphs correctly:

```python
from torch_geometric.data import Batch

# Create multiple graphs
graphs = [create_graph(seed=i) for i in range(10)]
batch = Batch.from_data_list(graphs)

# Each graph gets its own bottleneck
output = model(
    batch.x, 
    batch.edge_index, 
    batch.bc_mask,
    batch=batch.batch  # Important!
)
```

What happens:
1. Each graph encodes independently
2. Pooling creates **one feature vector per graph** (10 graphs → 10 vectors)
3. Each vector broadcasts back to **its own nodes**
4. Each graph decodes independently

## Advanced: Custom Architectures

### Mix Different Layer Types

```python
model = DeepGCN(
    hidden_channels=[128, 64, 32, 16],
    conv_types=["GCN", "GAT", "SAGE", "GCN"],  # Different types
    use_global_pooling=True,
    pooling_position="middle",
    encoder_layers=2,  # GCN, GAT
    # Decoder uses: SAGE, GCN
)
```

### Skip Connections Through Bottleneck

You can add skip connections manually:

```python
# Forward with intermediate features
x_input = data.x.clone()

# Encoder
for layer in model.encoder_layers:
    x = layer(x, edge_index)

# Pool
graph_feat = model.global_pooling(x, batch)

# Add skip connection: concatenate input features
x_with_skip = torch.cat([model.broadcast(graph_feat), x_input], dim=1)

# Decoder (needs to handle larger input)
for layer in model.decoder_layers:
    x = layer(x_with_skip, edge_index)
```

## Training Considerations

### 1. Bottleneck Size

- **Too small**: Model cannot capture necessary information → underfitting
- **Too large**: No regularization benefit → like standard architecture
- **Rule of thumb**: 1/4 to 1/8 of encoder output dimension

```python
encoder_output_dim = 64
graph_output_dim = 8  # 1/8 of encoder output
```

### 2. Learning Rate

Encoder-decoder may need different learning rates:

```python
optimizer = torch.optim.Adam([
    {'params': model.encoder_layers.parameters(), 'lr': 1e-3},
    {'params': model.decoder_layers.parameters(), 'lr': 1e-3},
    {'params': model.attention_weights.parameters(), 'lr': 5e-4},  # Lower for stability
])
```

### 3. Initialization

For physics-informed problems, initialize bottleneck to preserve energy:

```python
# After model creation
if model.graph_projection is not None:
    torch.nn.init.eye_(model.graph_projection.weight[:min(rows, cols), :min(rows, cols)])
```

### 4. Loss Functions

You can add auxiliary losses on graph features:

```python
# Forward pass
node_output = model(x, edge_index, bc_mask)

# Compute graph features for auxiliary loss
with torch.no_grad():
    encoded = ...  # Get encoder output
    graph_feat = model.global_pooling(encoded, batch)

# Multi-task loss
node_loss = criterion(node_output, node_target)
graph_loss = conservation_penalty(graph_feat)  # E.g., total energy should be constant
total_loss = node_loss + 0.1 * graph_loss
```

## Performance

### Memory Usage

| Architecture | Memory per Graph |
|--------------|------------------|
| Standard | O(N × hidden_dim) |
| Encoder-Decoder | O(N × hidden_dim + graph_dim) |

Memory is similar since graph_dim << N × hidden_dim

### Computational Cost

- **Encoder**: Same as standard architecture
- **Pooling**: O(N) - very cheap
- **Broadcast**: O(N) - very cheap  
- **Decoder**: Same as standard architecture

**Total**: Similar to standard architecture (pooling/broadcast are negligible)

### Training Speed

- Slightly slower due to additional operations
- Usually < 5% overhead
- Worth it for improved generalization

## Troubleshooting

### Issue: Output is all similar values

**Cause**: Bottleneck too small, model cannot encode diversity

**Solution**: Increase `graph_output_dim` or `encoder_layers`

### Issue: Model doesn't learn global patterns

**Cause**: Bottleneck too large, no compression

**Solution**: Decrease `graph_output_dim` to force compression

### Issue: Training unstable

**Cause**: Attention pooling with small bottleneck can be unstable

**Solution**: 
- Use mean or sum pooling
- Lower learning rate for attention weights
- Add gradient clipping

### Issue: Boundary conditions not respected

**Cause**: Decoder broadcasts to boundary nodes

**Solution**: Already handled! Boundary conditions applied **after** decoder:
```python
# In forward():
x = decoder(...)
x[bc_mask] = 0.0  # ✓ Applied after everything
```

## Summary

| Feature | Standard | Encoder-Decoder |
|---------|----------|-----------------|
| **Architecture** | Direct node → node | node → graph → node |
| **Information flow** | Local neighborhoods | Global aggregation |
| **Parameters** | `pooling_position="end"` | `pooling_position="middle"` |
| **Use cases** | Local physics | Global constraints |
| **Bottleneck** | None | Yes (regularization) |
| **Memory** | O(N × dim) | O(N × dim + graph_dim) |
| **Speed** | Baseline | ~5% slower |

## Quick Start

```python
# Minimal encoder-decoder setup
model = DeepGCN(
    hidden_channels=[64, 32],      # 2 layers
    in_channels=3,
    out_channels=2,
    use_global_pooling=True,
    pooling_position="middle",      # Enable encoder-decoder
    encoder_layers=1,               # 1 encoder, 1 decoder
)

# That's it! Use normally
output = model(x, edge_index, bc_mask)
```

For more examples, run:
```bash
python test_encoder_decoder.py
```
