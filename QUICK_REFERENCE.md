# DeepGCN: Quick Reference Guide

## Three Pooling Modes

### 1. No Pooling (Default)
```python
model = DeepGCN(
    hidden_channels=[64, 32],
    use_global_pooling=False  # Default
)
```
**Output**: Node-level predictions `[num_nodes, out_channels]`

**Use case**: Standard node-level tasks (e.g., predict displacement at each point)

---

### 2. Pooling at End
```python
model = DeepGCN(
    hidden_channels=[64, 32],
    use_global_pooling=True,
    pooling_position="end"  # Default when pooling enabled
)
```
**Output**: Graph-level predictions `[num_graphs, out_channels]`

**Use case**: Graph classification, predicting global properties

---

### 3. Pooling in Middle (Encoder-Decoder)
```python
model = DeepGCN(
    hidden_channels=[64, 64, 32, 32],
    use_global_pooling=True,
    pooling_position="middle",  # NEW!
    encoder_layers=2
)
```
**Output**: Node-level predictions `[num_nodes, out_channels]`

**Use case**: Node predictions with global constraints, physics conservation laws

---

## Architecture Visualizations

### No Pooling
```
Input [N, 3]
    ↓
Layer 1 [N, 64]
    ↓
Layer 2 [N, 32]
    ↓
Output [N, 2]
```
Each node processes independently through local neighborhoods.

---

### Pooling at End
```
Input [N, 3]
    ↓
Layer 1 [N, 64]
    ↓
Layer 2 [N, 32]
    ↓
Global Pool [1, 32] or [B, 32]
    ↓
Output [1, 2] or [B, 2]
```
Aggregates all nodes into single graph representation.

---

### Pooling in Middle (Encoder-Decoder)
```
Input [N, 3]
    ↓
Encoder Layer 1 [N, 64]
    ↓
Encoder Layer 2 [N, 64]
    ↓
Global Pool [1, 64] ← Bottleneck
    ↓
Broadcast [N, 64]
    ↓
Decoder Layer 1 [N, 32]
    ↓
Decoder Layer 2 [N, 32]
    ↓
Output [N, 2]
```
Compresses to graph-level, then expands back to nodes.

---

## Pooling Types

| Type | Formula | Best For | Example Use Case |
|------|---------|----------|------------------|
| `mean` | `avg(features)` | General purpose | Average wave amplitude |
| `max` | `max(features)` | Peak detection | Maximum displacement |
| `sum` | `sum(features)` | Conservation | Total energy |
| `attention` | `weighted_sum(features)` | Learned importance | Adaptive focus |

---

## Parameter Cheat Sheet

### Essential Parameters

```python
model = DeepGCN(
    # Architecture
    hidden_channels=[64, 32],       # Layer sizes (int or list)
    in_channels=3,                  # Input features per node
    out_channels=2,                 # Output features per node/graph
    
    # Pooling mode
    use_global_pooling=False,       # Enable pooling?
    pooling_position="end",         # "end" or "middle"
    pooling_type="mean",            # "mean", "max", "sum", "attention"
    
    # Encoder-decoder (when pooling_position="middle")
    encoder_layers=None,            # How many layers in encoder?
    decoder_channels=None,          # Decoder architecture (optional)
    graph_output_dim=None,          # Bottleneck size (optional)
)
```

### Configuration Examples

**Simple node prediction:**
```python
DeepGCN(hidden_channels=64, in_channels=3, out_channels=2)
```

**Graph classification:**
```python
DeepGCN(
    hidden_channels=[64, 32],
    in_channels=3,
    out_channels=5,  # 5 classes
    use_global_pooling=True,
    pooling_position="end"
)
```

**Physics-informed with conservation:**
```python
DeepGCN(
    hidden_channels=[128, 64, 32, 32],
    in_channels=3,
    out_channels=2,
    use_global_pooling=True,
    pooling_position="middle",
    pooling_type="sum",  # Conservation
    encoder_layers=2,
    graph_output_dim=8  # Tight bottleneck
)
```

---

## Forward Pass Examples

### Single Graph

```python
data = create_graph(seed=42)
output = model(data.x, data.edge_index, data.bc_mask)
```

### Batched Graphs

```python
from torch_geometric.data import Batch

graphs = [create_graph(seed=i) for i in range(10)]
batch = Batch.from_data_list(graphs)

output = model(
    batch.x,
    batch.edge_index,
    batch.bc_mask,
    batch=batch.batch  # Important for pooling!
)
```

### Get Both Node and Graph Features

```python
# Only works with pooling_position="end"
node_features, graph_features = model(
    data.x,
    data.edge_index,
    data.bc_mask,
    return_pooled=True  # Returns tuple
)
```

---

## Decision Tree

**Q: What do you want to predict?**

→ **Node values** (e.g., displacement at each point)
  - **Q: Do you need global information?**
    - No → Use **no pooling**
    - Yes → Use **pooling_position="middle"**

→ **Graph properties** (e.g., graph classification)
  - Use **pooling_position="end"**

→ **Both node and graph**
  - Use **pooling_position="end"** with `return_pooled=True`

---

## Common Recipes

### Recipe 1: Standard Wave Equation Solver
```python
model = DeepGCN(
    hidden_channels=[128, 64, 32],
    conv_types="GCN",
    in_channels=3,
    out_channels=2,
    use_global_pooling=False  # Local physics only
)
```

### Recipe 2: Energy-Conserving Wave Solver
```python
model = DeepGCN(
    hidden_channels=[128, 64, 32, 32],
    conv_types="GCN",
    in_channels=3,
    out_channels=2,
    use_global_pooling=True,
    pooling_position="middle",
    pooling_type="sum",  # Total energy
    encoder_layers=2,
    graph_output_dim=16
)
```

### Recipe 3: Wave Pattern Classifier
```python
model = DeepGCN(
    hidden_channels=[128, 64, 32],
    conv_types="GAT",
    in_channels=3,
    out_channels=10,  # 10 classes
    use_global_pooling=True,
    pooling_position="end",
    pooling_type="attention"
)
```

### Recipe 4: Multi-Task: Predict Nodes + Classify
```python
model = DeepGCN(
    hidden_channels=[128, 64, 32],
    in_channels=3,
    out_channels=2,  # Node predictions
    use_global_pooling=True,
    pooling_position="end",
    graph_output_dim=10  # Graph classification
)

# In training loop:
node_pred, graph_class = model(..., return_pooled=True)
node_loss = node_criterion(node_pred, node_target)
graph_loss = graph_criterion(graph_class, graph_label)
total_loss = node_loss + alpha * graph_loss
```

---

## Performance Tips

1. **Start simple**: Begin with no pooling, add complexity if needed
2. **Bottleneck size**: Use 1/4 to 1/8 of encoder output dimension
3. **Batch size**: Larger batches help pooling operations
4. **Pooling type**: 
   - Unsure? → Use `mean`
   - Conservation laws? → Use `sum`
   - Interpretability? → Use `attention`
5. **Debugging**: Use `return_pooled=True` to inspect intermediate features

---

## Files

- `dataset.py` - Main implementation
- `test_global_pooling.py` - Examples for pooling at end
- `test_encoder_decoder.py` - Examples for pooling in middle
- `GLOBAL_POOLING_README.md` - Detailed docs for end pooling
- `ENCODER_DECODER_README.md` - Detailed docs for middle pooling
- `QUICK_REFERENCE.md` - This file

---

## Testing

Run examples:
```bash
# Test pooling at end
python test_global_pooling.py

# Test encoder-decoder
python test_encoder_decoder.py
```

Verify implementation:
```python
# Create models with each mode
model1 = DeepGCN(hidden_channels=64, use_global_pooling=False)
model2 = DeepGCN(hidden_channels=64, use_global_pooling=True, pooling_position="end")
model3 = DeepGCN(hidden_channels=[64,32], use_global_pooling=True, pooling_position="middle", encoder_layers=1)

# Test forward pass
data = create_graph(seed=42)
out1 = model1(data.x, data.edge_index, data.bc_mask)  # [N, 2]
out2 = model2(data.x, data.edge_index, data.bc_mask)  # [1, 2]
out3 = model3(data.x, data.edge_index, data.bc_mask)  # [N, 2]

print(f"No pooling: {out1.shape}")
print(f"End pooling: {out2.shape}")
print(f"Middle pooling: {out3.shape}")
```
