"""
Example script demonstrating global pooling with DeepGCN.

This script shows how to:
1. Create a model with global pooling enabled
2. Process single graphs
3. Process batched graphs
4. Use different pooling strategies (mean, max, sum, attention)
"""

import torch
from torch_geometric.data import Batch
from dataset import DeepGCN, create_graph

def example_single_graph():
    """Example: Global pooling on a single graph"""
    print("\n" + "="*60)
    print("Example 1: Single Graph with Mean Pooling")
    print("="*60)
    
    # Create model with global pooling enabled
    model = DeepGCN(
        hidden_channels=[64, 64, 32],
        conv_types="GCN",
        final_layer_type="Linear",
        in_channels=3,
        out_channels=2,
        use_global_pooling=True,
        pooling_type="mean",
        graph_output_dim=16  # Optional: project to different dimension
    )
    
    # Create a sample graph
    data = create_graph(seed=42)
    
    print(f"Input shape: {data.x.shape}")  # [num_nodes, 3]
    print(f"Number of nodes: {data.x.shape[0]}")
    
    # Forward pass - returns only graph-level features
    graph_features = model(data.x, data.edge_index, data.bc_mask)
    print(f"Graph-level output shape: {graph_features.shape}")  # [1, 16]
    
    # Forward pass - return both node and graph features
    node_features, graph_features = model(
        data.x, data.edge_index, data.bc_mask, return_pooled=True
    )
    print(f"Node-level output shape: {node_features.shape}")  # [num_nodes, 2]
    print(f"Graph-level output shape: {graph_features.shape}")  # [1, 16]


def example_batched_graphs():
    """Example: Global pooling on batched graphs"""
    print("\n" + "="*60)
    print("Example 2: Batched Graphs with Max Pooling")
    print("="*60)
    
    # Create model with max pooling
    model = DeepGCN(
        hidden_channels=[64, 32],
        conv_types="GCN",
        final_layer_type="Linear",
        in_channels=3,
        out_channels=2,
        use_global_pooling=True,
        pooling_type="max"
    )
    
    # Create multiple graphs
    graphs = [create_graph(seed=i) for i in range(5)]
    
    # Batch them together
    batch = Batch.from_data_list(graphs)
    
    print(f"Batch size: {len(graphs)}")
    print(f"Total nodes in batch: {batch.x.shape[0]}")
    print(f"Batch vector: {batch.batch[:10]}... (first 10 nodes)")
    
    # Forward pass with batch vector
    graph_features = model(batch.x, batch.edge_index, batch.bc_mask, batch=batch.batch)
    print(f"Graph-level output shape: {graph_features.shape}")  # [5, 2]
    print(f"Each row represents one graph's pooled features")


def example_attention_pooling():
    """Example: Attention-based global pooling"""
    print("\n" + "="*60)
    print("Example 3: Attention-Based Pooling")
    print("="*60)
    
    # Create model with attention pooling
    model = DeepGCN(
        hidden_channels=[64, 32],
        conv_types="GCN",
        final_layer_type="Linear",
        in_channels=3,
        out_channels=2,
        use_global_pooling=True,
        pooling_type="attention",  # Learns which nodes are more important
        graph_output_dim=8
    )
    
    # Create a sample graph
    data = create_graph(seed=123)
    
    print(f"Model has learnable attention weights")
    print(f"Attention weight layer: {model.attention_weights}")
    
    # Forward pass
    graph_features = model(data.x, data.edge_index, data.bc_mask)
    print(f"Graph-level output shape: {graph_features.shape}")  # [1, 8]


def example_all_pooling_types():
    """Compare different pooling strategies"""
    print("\n" + "="*60)
    print("Example 4: Comparing Pooling Strategies")
    print("="*60)
    
    # Create a sample graph
    data = create_graph(seed=99)
    
    pooling_types = ["mean", "max", "sum", "attention"]
    
    for pool_type in pooling_types:
        model = DeepGCN(
            hidden_channels=[32],
            conv_types="GCN",
            final_layer_type="Linear",
            in_channels=3,
            out_channels=2,
            use_global_pooling=True,
            pooling_type=pool_type
        )
        
        with torch.no_grad():
            graph_features = model(data.x, data.edge_index, data.bc_mask)
        
        print(f"\n{pool_type.upper()} pooling:")
        print(f"  Output shape: {graph_features.shape}")
        print(f"  Output values: {graph_features.squeeze().numpy()}")


def example_node_level_vs_graph_level():
    """Example: Switch between node-level and graph-level predictions"""
    print("\n" + "="*60)
    print("Example 5: Node-level vs Graph-level Predictions")
    print("="*60)
    
    data = create_graph(seed=42)
    
    # Model for node-level predictions (default)
    node_model = DeepGCN(
        hidden_channels=[64, 32],
        conv_types="GCN",
        final_layer_type="Linear",
        in_channels=3,
        out_channels=2,
        use_global_pooling=False  # Disabled
    )
    
    # Model for graph-level predictions
    graph_model = DeepGCN(
        hidden_channels=[64, 32],
        conv_types="GCN",
        final_layer_type="Linear",
        in_channels=3,
        out_channels=2,
        use_global_pooling=True,  # Enabled
        pooling_type="mean"
    )
    
    with torch.no_grad():
        node_output = node_model(data.x, data.edge_index, data.bc_mask)
        graph_output = graph_model(data.x, data.edge_index, data.bc_mask)
    
    print(f"\nNode-level model (use_global_pooling=False):")
    print(f"  Output shape: {node_output.shape}")  # [num_nodes, 2]
    print(f"  Use case: Predict per-node values (u, v at each position)")
    
    print(f"\nGraph-level model (use_global_pooling=True):")
    print(f"  Output shape: {graph_output.shape}")  # [1, 2]
    print(f"  Use case: Predict global properties (e.g., total energy, classification)")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("DeepGCN Global Pooling Examples")
    print("="*60)
    
    # Run all examples
    example_single_graph()
    example_batched_graphs()
    example_attention_pooling()
    example_all_pooling_types()
    example_node_level_vs_graph_level()
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60 + "\n")
