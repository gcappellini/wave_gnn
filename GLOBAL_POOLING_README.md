# DeepGCN Global Pooling

This document describes the global pooling functionality added to the `DeepGCN` class.

## Overview

Global pooling aggregates node-level features into a single graph-level representation. This is useful for:
- **Graph classification**: Predict a label for the entire graph
- **Graph regression**: Predict global properties (e.g., total energy, average temperature)
- **Graph comparison**: Generate embeddings for graph similarity tasks
- **Hierarchical models**: Use graph-level features as input to downstream models

## Usage

### Basic Setup

```python
from dataset import DeepGCN, create_graph

# Create model with global pooling
model = DeepGCN(
    hidden_channels=[64, 64, 32],
    conv_types="GCN",
    final_layer_type="Linear",
    in_channels=3,
    out_channels=2,
    use_global_pooling=True,      # Enable global pooling
    pooling_type="mean",           # Pooling strategy
    graph_output_dim=16            # Optional: project to different dimension
)

# Create a graph
data = create_graph(seed=42)

# Forward pass returns graph-level features
graph_features = model(data.x, data.edge_index, data.bc_mask)
# Output shape: [1, 16] for single graph
```

### Pooling Types

Four pooling strategies are available:

#### 1. Mean Pooling (default)
```python
model = DeepGCN(..., pooling_type="mean")
```
Computes the average of all node features:
```
graph_feature = (1/N) * Σ(node_features)
```
- **Best for**: Most general-purpose use cases
- **Properties**: Smooth, stable gradients

#### 2. Max Pooling
```python
model = DeepGCN(..., pooling_type="max")
```
Takes the maximum value across all nodes for each feature:
```
graph_feature = max(node_features, dim=0)
```
- **Best for**: Detecting presence of specific patterns
- **Properties**: Sparse gradients, emphasizes extremes

#### 3. Sum Pooling
```python
model = DeepGCN(..., pooling_type="sum")
```
Sums all node features:
```
graph_feature = Σ(node_features)
```
- **Best for**: When total quantity matters (e.g., total energy)
- **Properties**: Sensitive to graph size

#### 4. Attention Pooling
```python
model = DeepGCN(..., pooling_type="attention")
```
Learns importance weights for each node:
```
attention_weights = softmax(W * node_features)
graph_feature = Σ(attention_weights * node_features)
```
- **Best for**: When some nodes are more important than others
- **Properties**: Learnable, adaptive, interpretable

### Batched Graphs

Process multiple graphs simultaneously:

```python
from torch_geometric.data import Batch

# Create multiple graphs
graphs = [create_graph(seed=i) for i in range(10)]

# Batch them together
batch = Batch.from_data_list(graphs)

# Forward pass with batch vector
graph_features = model(
    batch.x, 
    batch.edge_index, 
    batch.bc_mask, 
    batch=batch.batch  # Important: pass batch vector
)
# Output shape: [10, output_dim] - one row per graph
```

### Return Both Node and Graph Features

Sometimes you need both node-level and graph-level outputs:

```python
node_features, graph_features = model(
    data.x, 
    data.edge_index, 
    data.bc_mask,
    return_pooled=True  # Returns tuple
)
# node_features shape: [num_nodes, out_channels]
# graph_features shape: [1, graph_output_dim or out_channels]
```

## Parameters

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_global_pooling` | bool | `False` | Enable global pooling |
| `pooling_type` | str | `"mean"` | Type of pooling: `"mean"`, `"max"`, `"sum"`, `"attention"` |
| `graph_output_dim` | int | `None` | Optional projection dimension. If `None`, uses `out_channels` |

### Forward Method Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch` | Tensor | `None` | Batch vector for multi-graph batching. If `None`, assumes single graph |
| `return_pooled` | bool | `False` | If `True`, returns `(node_features, graph_features)` tuple |

## Examples

### Example 1: Graph Classification

```python
# Model for binary classification
model = DeepGCN(
    hidden_channels=[64, 64, 32],
    conv_types=["GCN", "GCN", "GAT"],
    final_layer_type="Linear",
    in_channels=3,
    out_channels=1,              # Single output for binary classification
    use_global_pooling=True,
    pooling_type="attention",    # Learn important nodes
    graph_output_dim=1           # Single value output
)

# Training loop
for graphs, labels in dataloader:
    predictions = model(graphs.x, graphs.edge_index, graphs.bc_mask, graphs.batch)
    loss = criterion(predictions, labels)
    # ... backward pass ...
```

### Example 2: Residual Predictions with Pooling

```python
# Predict both node-level changes and graph-level energy
model = DeepGCN(
    hidden_channels=[128, 64, 32],
    conv_types="GCN",
    final_layer_type="Linear",
    in_channels=3,
    out_channels=2,
    residual=True,               # Predict Δu, Δv
    use_global_pooling=True,
    pooling_type="sum",          # Total energy is a sum
    graph_output_dim=1
)

# Get both outputs
node_changes, total_energy = model(
    data.x, 
    data.edge_index, 
    data.bc_mask,
    return_pooled=True
)
```

### Example 3: Multi-Task Learning

```python
# Predict node values AND graph classification
model = DeepGCN(
    hidden_channels=[128, 128, 64, 32],
    conv_types="GCN",
    final_layer_type="Linear",
    in_channels=3,
    out_channels=2,              # Node-level predictions
    use_global_pooling=True,
    pooling_type="mean",
    graph_output_dim=5           # Graph-level classification (5 classes)
)

# Custom training loop
node_pred, graph_pred = model(
    batch.x, 
    batch.edge_index, 
    batch.bc_mask, 
    batch.batch,
    return_pooled=True
)

# Multi-task loss
node_loss = node_criterion(node_pred, node_targets)
graph_loss = graph_criterion(graph_pred, graph_labels)
total_loss = node_loss + alpha * graph_loss
```

## Architecture Details

### Without Global Pooling (default)
```
Input [N, in_channels]
  ↓
Input Projection (optional)
  ↓
DeepGCN Layers (with skip connections)
  ↓
Final Layer
  ↓
Apply Boundary Conditions
  ↓
Output [N, out_channels]
```

### With Global Pooling
```
Input [N, in_channels]
  ↓
Input Projection (optional)
  ↓
DeepGCN Layers (with skip connections)
  ↓
Final Layer
  ↓
Apply Boundary Conditions
  ↓
Global Pooling (mean/max/sum/attention)
  ↓
Graph Projection (optional)
  ↓
Output [num_graphs, graph_output_dim]
```

## Implementation Notes

### Attention Pooling Details

The attention mechanism is implemented as:
```python
attention_scores = Linear(node_features)  # [N, feature_dim] → [N, 1]
attention_weights = softmax(attention_scores, dim=0)  # Normalize over all nodes
pooled = Σ(node_features * attention_weights)
```

This creates learnable parameters that determine which nodes contribute most to the graph representation.

### Batch Processing

The `batch` parameter is crucial for processing multiple graphs:
- It's a vector of integers where `batch[i] = j` means node `i` belongs to graph `j`
- PyTorch Geometric's `Batch.from_data_list()` automatically creates this vector
- Pooling operations respect batch boundaries (e.g., mean pooling averages nodes within each graph separately)

### Memory Considerations

- **Node-level output**: Memory scales with `O(N * out_channels)` where N is total nodes
- **Graph-level output**: Memory scales with `O(B * graph_output_dim)` where B is batch size
- **Attention pooling**: Adds `O(out_channels)` learnable parameters

## Testing

Run the included test script to see all features in action:

```bash
python test_global_pooling.py
```

This will demonstrate:
1. Single graph pooling
2. Batched graph pooling
3. Different pooling strategies
4. Attention-based pooling
5. Node-level vs graph-level predictions

## Performance Tips

1. **Choose the right pooling type**:
   - Start with `mean` for most tasks
   - Use `attention` if you need interpretability or have varying node importance
   - Use `sum` for extensive properties (scales with graph size)
   - Use `max` for detecting presence of features

2. **Batch size**: Larger batches improve GPU utilization for graph-level tasks

3. **Graph output dimension**: 
   - Smaller dimension (8-32) for simple classification
   - Larger dimension (64-256) for complex embeddings

4. **Memory optimization**: If you only need graph-level features, don't set `return_pooled=True`

## Troubleshooting

**Q: My model outputs different shapes than expected**

Check `use_global_pooling` and `return_pooled` settings. Default behavior returns only node-level features.

**Q: Batched predictions have wrong shape**

Make sure to pass the `batch` parameter: `model(..., batch=batch.batch)`

**Q: Attention pooling gives poor results**

Attention needs sufficient model capacity to learn good weights. Try increasing `hidden_channels` or adding more layers.

**Q: Results differ significantly between pooling types**

This is expected! Different pooling types emphasize different aspects of the graph. Validate which makes sense for your task.
