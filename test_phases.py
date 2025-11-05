"""
Example usage and comparison of all three phases of GNN models.

This script demonstrates:
1. Phase 1: Global node broadcast (try_gno.py)
2. Phase 2: Pure spectral model (spectral_models.py)
3. Phase 3: Hybrid spatial-spectral model (spectral_models.py)
"""

import torch
import numpy as np
from omegaconf import OmegaConf
import sys
import os

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from try_gno import WaveGNN
from spectral_models import SpectralWaveGNN, HybridWaveGNN, create_spectral_model
from dataset import create_graph


def create_dummy_config():
    """Create a dummy configuration for testing."""
    cfg = OmegaConf.create({
        'model': {
            'hidden_dim': 64,
            'num_layers': 3,
            'dropout': 0.1,
            'num_spectral_modes': 30,  # Use 30 spectral modes out of 100 nodes
            'spatial_weight': 0.5,
            'spectral_weight': 0.5,
        },
        'dataset': {
            'dt': 0.01,
            'u_scale': 0.04,
            'v_scale': 0.08,
            'f_scale': 3.0,
            'force': {
                'location': 'middle',
                'sign': -1,
                'forcing_type': 'middle',
                'margin': 0.1,
            }
        }
    })
    return cfg


def test_phase1(cfg, data, device='cpu'):
    """Test Phase 1: Global node broadcast model."""
    print("\n" + "="*70)
    print("PHASE 1: Global Node Broadcast")
    print("="*70)
    
    model = WaveGNN(cfg).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Prepare data
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    bc_mask = data.bc_mask.to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x, edge_index, bc_mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range - u: [{output[:, 0].min():.6f}, {output[:, 0].max():.6f}]")
    print(f"Output range - v: [{output[:, 1].min():.6f}, {output[:, 1].max():.6f}]")
    
    return model, output


def test_phase2(cfg, data, device='cpu'):
    """Test Phase 2: Pure spectral model."""
    print("\n" + "="*70)
    print("PHASE 2: Pure Spectral Model (Laplacian Eigenbasis)")
    print("="*70)
    
    model = SpectralWaveGNN(cfg).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of spectral modes: {model.num_spectral_modes}")
    
    # Prepare data
    x = data.x.to(device)
    bc_mask = data.bc_mask.to(device)
    eigenvectors = data.eigenvectors.to(device)
    
    print(f"Eigenvector matrix shape: {eigenvectors.shape}")
    print(f"Eigenvalue range: [{data.eigenvalues.min():.6f}, {data.eigenvalues.max():.6f}]")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x, edge_index=None, bc_mask=bc_mask, eigenvectors=eigenvectors)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range - u: [{output[:, 0].min():.6f}, {output[:, 0].max():.6f}]")
    print(f"Output range - v: [{output[:, 1].min():.6f}, {output[:, 1].max():.6f}]")
    
    return model, output


def test_phase3(cfg, data, device='cpu'):
    """Test Phase 3: Hybrid spatial-spectral model."""
    print("\n" + "="*70)
    print("PHASE 3: Hybrid Spatial-Spectral Model")
    print("="*70)
    
    model = HybridWaveGNN(cfg).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of spectral modes: {model.num_spectral_modes}")
    print(f"Spatial branch: {len(model.spatial_layers)} layers")
    print(f"Spectral branch: {len(model.spectral_layers)} layers")
    
    # Prepare data
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    bc_mask = data.bc_mask.to(device)
    eigenvectors = data.eigenvectors.to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x, edge_index, bc_mask=bc_mask, eigenvectors=eigenvectors)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range - u: [{output[:, 0].min():.6f}, {output[:, 0].max():.6f}]")
    print(f"Output range - v: [{output[:, 1].min():.6f}, {output[:, 1].max():.6f}]")
    
    return model, output


def compare_models(outputs):
    """Compare outputs from different models."""
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    phase1_out, phase2_out, phase3_out = outputs
    
    # Compute differences
    diff_12 = torch.abs(phase1_out - phase2_out).mean()
    diff_13 = torch.abs(phase1_out - phase3_out).mean()
    diff_23 = torch.abs(phase2_out - phase3_out).mean()
    
    print(f"Mean absolute difference:")
    print(f"  Phase 1 vs Phase 2: {diff_12:.6f}")
    print(f"  Phase 1 vs Phase 3: {diff_13:.6f}")
    print(f"  Phase 2 vs Phase 3: {diff_23:.6f}")
    
    # Note: Differences are expected since models are randomly initialized
    print("\nNote: Differences are expected since all models are randomly initialized.")
    print("After training, we can compare their performance on test data.")


def main():
    """Main function to run all tests."""
    print("="*70)
    print("Graph Neural Operator - Three Phases Demo")
    print("="*70)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create config and data
    print("\nCreating configuration and dataset...")
    cfg = create_dummy_config()
    data = create_graph(seed=42, zeros=False, cfg=cfg)
    
    print(f"Graph created: {data.x.shape[0]} nodes")
    print(f"Features: {data.x.shape[1]} (u, v, f)")
    print(f"Edges: {data.edge_index.shape[1]}")
    print(f"Laplacian eigenvectors: {data.eigenvectors.shape}")
    print(f"Spectral modes available: {data.eigenvectors.shape[1]}")
    
    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Test all three phases
    model1, out1 = test_phase1(cfg, data, device)
    model2, out2 = test_phase2(cfg, data, device)
    model3, out3 = test_phase3(cfg, data, device)
    
    # Compare outputs
    compare_models((out1, out2, out3))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nPhase 1 (Global Node):")
    print("  ✓ Simple virtual global node for broadcasting")
    print("  ✓ No graph structure information in global communication")
    print("  ✓ Baseline approach")
    
    print("\nPhase 2 (Pure Spectral):")
    print("  ✓ Uses Laplacian eigenbasis (Fourier transform on graph)")
    print("  ✓ Principled global communication based on graph structure")
    print("  ✓ Processes different frequency modes separately")
    print("  ✓ No explicit edge-based message passing")
    
    print("\nPhase 3 (Hybrid):")
    print("  ✓ Combines local message passing (spatial) and spectral processing")
    print("  ✓ Spatial branch: captures local structure via edges")
    print("  ✓ Spectral branch: captures global patterns via Fourier")
    print("  ✓ Best of both worlds")
    
    print("\n" + "="*70)
    print("To use in training:")
    print("  1. Phase 1: from try_gno import WaveGNN")
    print("  2. Phase 2: from spectral_models import SpectralWaveGNN")
    print("  3. Phase 3: from spectral_models import HybridWaveGNN")
    print("="*70)


if __name__ == "__main__":
    main()
