"""
Example script demonstrating encoder-decoder architecture with global pooling in the middle.

This architecture:
1. Encodes node features into latent representations
2. Pools to graph-level features (compression/bottleneck)
3. Broadcasts graph features back to nodes
4. Decodes to final node-level predictions

Use cases:
- Force the model to learn global patterns
- Create an information bottleneck for regularization
- Learn graph-level representations while predicting node values
- Physics-informed models where global conservation laws matter
"""

import torch
from torch_geometric.data import Batch
from dataset import DeepGCN, create_graph
import matplotlib.pyplot as plt
import numpy as np


def example_basic_encoder_decoder():
    """Example 1: Basic encoder-decoder with middle pooling"""
    print("\n" + "="*70)
    print("Example 1: Basic Encoder-Decoder Architecture")
    print("="*70)
    
    # Create model with pooling in middle
    model = DeepGCN(
        hidden_channels=[64, 64, 32, 32],  # 4 layers total
        conv_types="GCN",
        final_layer_type="Linear",
        in_channels=3,
        out_channels=2,
        use_global_pooling=True,
        pooling_position="middle",  # Pool in the middle!
        pooling_type="mean",
        encoder_layers=2,  # First 2 layers are encoder
        # Remaining 2 layers become decoder
    )
    
    print(f"Architecture:")
    print(f"  Input: [num_nodes, 3]")
    print(f"  ↓")
    print(f"  Encoder: 2 layers (64 → 64)")
    print(f"  ↓")
    print(f"  Global Pooling: mean ([num_nodes, 64] → [1, 64])")
    print(f"  ↓")
    print(f"  Broadcast: [1, 64] → [num_nodes, 64]")
    print(f"  ↓")
    print(f"  Decoder: 2 layers (64 → 32 → 32)")
    print(f"  ↓")
    print(f"  Output: [num_nodes, 2]")
    
    # Test with a graph
    data = create_graph(seed=42)
    print(f"\nInput shape: {data.x.shape}")
    
    with torch.no_grad():
        output = model(data.x, data.edge_index, data.bc_mask)
    
    print(f"Output shape: {output.shape}")
    print(f"\nThe model learned to:")
    print(f"  1. Extract features from node neighborhoods (encoder)")
    print(f"  2. Aggregate global information (pooling)")
    print(f"  3. Distribute global context to all nodes (broadcast)")
    print(f"  4. Make node-level predictions (decoder)")


def example_custom_decoder():
    """Example 2: Custom decoder architecture"""
    print("\n" + "="*70)
    print("Example 2: Custom Decoder Architecture")
    print("="*70)
    
    model = DeepGCN(
        hidden_channels=[128, 64, 32],  # Encoder layers
        conv_types=["GCN", "GAT", "SAGE"],
        final_layer_type="Linear",
        in_channels=3,
        out_channels=2,
        use_global_pooling=True,
        pooling_position="middle",
        pooling_type="attention",  # Learn which nodes are important
        encoder_layers=3,  # All 3 layers are encoder
        decoder_channels=[64, 32, 16],  # Custom decoder: 3 layers
        graph_output_dim=16,  # Compress to smaller bottleneck
    )
    
    print(f"Encoder: [3] → 128 → 64 → 32")
    print(f"Pooling: [num_nodes, 32] → [1, 16] (with attention)")
    print(f"Decoder: [16] → 64 → 32 → 16")
    print(f"Output:  [16] → [2]")
    
    data = create_graph(seed=99)
    
    with torch.no_grad():
        output = model(data.x, data.edge_index, data.bc_mask)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Bottleneck dimension: 16 (compressed from 32)")


def example_batched_graphs():
    """Example 3: Process multiple graphs with encoder-decoder"""
    print("\n" + "="*70)
    print("Example 3: Batched Graphs with Encoder-Decoder")
    print("="*70)
    
    model = DeepGCN(
        hidden_channels=[64, 32],
        conv_types="GCN",
        final_layer_type="Linear",
        in_channels=3,
        out_channels=2,
        use_global_pooling=True,
        pooling_position="middle",
        pooling_type="mean",
        encoder_layers=1,  # 1 encoder, 1 decoder
    )
    
    # Create multiple graphs
    graphs = [create_graph(seed=i) for i in range(5)]
    batch = Batch.from_data_list(graphs)
    
    print(f"Batch size: {len(graphs)}")
    print(f"Total nodes: {batch.x.shape[0]}")
    
    with torch.no_grad():
        output = model(batch.x, batch.edge_index, batch.bc_mask, batch=batch.batch)
    
    print(f"Output shape: {output.shape}")
    print(f"\nEach graph:")
    print(f"  1. Encoded independently")
    print(f"  2. Pooled to its own graph feature")
    print(f"  3. Broadcast back to its nodes")
    print(f"  4. Decoded independently")


def example_comparison():
    """Example 4: Compare standard vs encoder-decoder"""
    print("\n" + "="*70)
    print("Example 4: Standard Architecture vs Encoder-Decoder")
    print("="*70)
    
    # Standard model (no pooling)
    model_standard = DeepGCN(
        hidden_channels=[64, 32],
        conv_types="GCN",
        final_layer_type="Linear",
        in_channels=3,
        out_channels=2,
        use_global_pooling=False,  # No pooling
    )
    
    # Encoder-decoder model
    model_encdec = DeepGCN(
        hidden_channels=[64, 32],
        conv_types="GCN",
        final_layer_type="Linear",
        in_channels=3,
        out_channels=2,
        use_global_pooling=True,
        pooling_position="middle",
        pooling_type="mean",
        encoder_layers=1,
    )
    
    data = create_graph(seed=42)
    
    with torch.no_grad():
        out_standard = model_standard(data.x, data.edge_index, data.bc_mask)
        out_encdec = model_encdec(data.x, data.edge_index, data.bc_mask)
    
    print(f"\nStandard Model:")
    print(f"  Architecture: Direct node → node")
    print(f"  Output shape: {out_standard.shape}")
    print(f"  Each node prediction depends on local neighborhood")
    
    print(f"\nEncoder-Decoder Model:")
    print(f"  Architecture: node → graph → node")
    print(f"  Output shape: {out_encdec.shape}")
    print(f"  Each node prediction incorporates global information")
    
    # Show that all nodes have some shared information
    print(f"\nVariance in standard output: {out_standard.var().item():.6f}")
    print(f"Variance in enc-dec output: {out_encdec.var().item():.6f}")


def example_physics_informed():
    """Example 5: Physics-informed use case"""
    print("\n" + "="*70)
    print("Example 5: Physics-Informed Wave Equation")
    print("="*70)
    
    # For wave equation: global pooling can enforce conservation laws
    model = DeepGCN(
        hidden_channels=[128, 64, 32],
        conv_types="GCN",
        final_layer_type="Linear",
        in_channels=3,  # u, v, f
        out_channels=2,  # u_next, v_next
        use_global_pooling=True,
        pooling_position="middle",
        pooling_type="sum",  # Sum pooling for total energy
        encoder_layers=2,
        graph_output_dim=8,  # Small bottleneck for conservation
    )
    
    data = create_graph(seed=123)
    
    with torch.no_grad():
        output = model(data.x, data.edge_index, data.bc_mask)
    
    print(f"Wave equation model with encoder-decoder:")
    print(f"  - Encoder extracts wave features")
    print(f"  - Sum pooling captures total energy")
    print(f"  - Bottleneck (dim=8) enforces energy conservation")
    print(f"  - Decoder distributes energy back to nodes")
    print(f"  - Output: wave state at next time step")
    
    print(f"\nInput total energy: {data.x[:, :2].abs().sum().item():.4f}")
    print(f"Output total energy: {output.abs().sum().item():.4f}")


def visualize_information_flow():
    """Example 6: Visualize how information flows through encoder-decoder"""
    print("\n" + "="*70)
    print("Example 6: Information Flow Visualization")
    print("="*70)
    
    model = DeepGCN(
        hidden_channels=[32, 16],
        conv_types="GCN",
        final_layer_type="Linear",
        in_channels=3,
        out_channels=2,
        use_global_pooling=True,
        pooling_position="middle",
        pooling_type="mean",
        encoder_layers=1,
        graph_output_dim=4,  # Small for visualization
    )
    
    data = create_graph(seed=42)
    
    # Hook to capture intermediate features
    encoder_output = None
    graph_features = None
    
    def hook_encoder(module, input, output):
        nonlocal encoder_output
        encoder_output = output.detach().clone()
    
    # Register hook on last encoder layer
    model.encoder_layers[-1].register_forward_hook(hook_encoder)
    
    with torch.no_grad():
        output = model(data.x, data.edge_index, data.bc_mask)
        
        # Manually compute graph features for visualization
        graph_features = model.global_pooling(encoder_output, None)
    
    print(f"Input shape: {data.x.shape}")
    print(f"After encoder: {encoder_output.shape}")
    print(f"After pooling: {graph_features.shape}")
    print(f"Output shape: {output.shape}")
    
    print(f"\nInformation compression:")
    print(f"  Nodes: {data.x.shape[0]} × {data.x.shape[1]} = {data.x.numel()} values")
    print(f"  Bottleneck: {graph_features.numel()} values")
    print(f"  Compression ratio: {data.x.numel() / graph_features.numel():.1f}x")


def example_different_pooling_strategies():
    """Example 7: Compare pooling strategies in encoder-decoder"""
    print("\n" + "="*70)
    print("Example 7: Pooling Strategies in Encoder-Decoder")
    print("="*70)
    
    data = create_graph(seed=42)
    pooling_types = ["mean", "max", "sum", "attention"]
    
    for pool_type in pooling_types:
        model = DeepGCN(
            hidden_channels=[32, 16],
            conv_types="GCN",
            final_layer_type="Linear",
            in_channels=3,
            out_channels=2,
            use_global_pooling=True,
            pooling_position="middle",
            pooling_type=pool_type,
            encoder_layers=1,
        )
        
        with torch.no_grad():
            output = model(data.x, data.edge_index, data.bc_mask)
        
        print(f"\n{pool_type.upper()} pooling:")
        print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"  Output mean: {output.mean().item():.4f}")
        print(f"  Output std: {output.std().item():.4f}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DeepGCN Encoder-Decoder with Middle Pooling Examples")
    print("="*70)
    
    example_basic_encoder_decoder()
    example_custom_decoder()
    example_batched_graphs()
    example_comparison()
    example_physics_informed()
    visualize_information_flow()
    example_different_pooling_strategies()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70 + "\n")
    
    print("Key Takeaways:")
    print("  • pooling_position='middle' creates encoder-decoder architecture")
    print("  • Encoder compresses node info → graph info (bottleneck)")
    print("  • Decoder expands graph info → node predictions")
    print("  • Useful for enforcing global constraints and learning global patterns")
    print("  • encoder_layers controls encoder/decoder split")
    print("  • decoder_channels customizes decoder architecture")
