"""
Helper utilities for spectral GNN models.

This module provides utility functions for:
- Model creation and configuration
- Eigendecomposition management
- Visualization helpers
- Performance analysis
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple


def get_model_info(model) -> Dict[str, any]:
    """
    Get comprehensive information about a model.
    
    Args:
        model: PyTorch model instance
    
    Returns:
        Dictionary with model statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_type': model.__class__.__name__,
    }
    
    # Add model-specific info
    if hasattr(model, 'num_spectral_modes'):
        info['num_spectral_modes'] = model.num_spectral_modes
        info['uses_spectral'] = True
    else:
        info['uses_spectral'] = False
    
    if hasattr(model, 'num_layers'):
        info['num_layers'] = model.num_layers
    
    if hasattr(model, 'hidden_dim'):
        info['hidden_dim'] = model.hidden_dim
    
    return info


def print_model_summary(model, verbose=True):
    """
    Print a formatted summary of the model.
    
    Args:
        model: PyTorch model
        verbose: If True, print layer-by-layer details
    """
    info = get_model_info(model)
    
    print("=" * 70)
    print(f"Model: {info['model_type']}")
    print("=" * 70)
    print(f"Total parameters: {info['total_parameters']:,}")
    print(f"Trainable parameters: {info['trainable_parameters']:,}")
    
    if 'hidden_dim' in info:
        print(f"Hidden dimension: {info['hidden_dim']}")
    
    if 'num_layers' in info:
        print(f"Number of layers: {info['num_layers']}")
    
    if info.get('uses_spectral', False):
        print(f"Spectral modes: {info['num_spectral_modes']}")
        print("Uses Laplacian eigenbasis: ✓")
    
    if verbose:
        print("\nLayer-by-layer breakdown:")
        print("-" * 70)
        for name, module in model.named_children():
            num_params = sum(p.numel() for p in module.parameters())
            print(f"{name:30s} | {num_params:>10,} parameters")
    
    print("=" * 70)


def analyze_eigenspectrum(eigenvalues: torch.Tensor) -> Dict[str, float]:
    """
    Analyze the eigenvalue spectrum of the Laplacian.
    
    Args:
        eigenvalues: [K] tensor of eigenvalues
    
    Returns:
        Dictionary with spectral statistics
    """
    eigenvalues = eigenvalues.cpu().numpy()
    
    # Basic statistics
    stats = {
        'min_eigenvalue': float(eigenvalues.min()),
        'max_eigenvalue': float(eigenvalues.max()),
        'mean_eigenvalue': float(eigenvalues.mean()),
        'std_eigenvalue': float(eigenvalues.std()),
        'num_modes': len(eigenvalues),
    }
    
    # Spectral gaps (differences between consecutive eigenvalues)
    gaps = np.diff(eigenvalues)
    stats['mean_gap'] = float(gaps.mean())
    stats['max_gap'] = float(gaps.max())
    stats['max_gap_index'] = int(gaps.argmax())
    
    # Frequency distribution
    stats['zero_eigenvalues'] = int(np.sum(np.abs(eigenvalues) < 1e-6))
    
    return stats


def print_eigenspectrum_summary(eigenvalues: torch.Tensor):
    """
    Print a summary of the eigenvalue spectrum.
    
    Args:
        eigenvalues: [K] tensor of eigenvalues
    """
    stats = analyze_eigenspectrum(eigenvalues)
    
    print("=" * 70)
    print("Laplacian Eigenspectrum Summary")
    print("=" * 70)
    print(f"Number of modes: {stats['num_modes']}")
    print(f"Eigenvalue range: [{stats['min_eigenvalue']:.6f}, {stats['max_eigenvalue']:.6f}]")
    print(f"Mean eigenvalue: {stats['mean_eigenvalue']:.6f} ± {stats['std_eigenvalue']:.6f}")
    print(f"\nSpectral gaps:")
    print(f"  Mean gap: {stats['mean_gap']:.6f}")
    print(f"  Max gap: {stats['max_gap']:.6f} (at index {stats['max_gap_index']})")
    
    if stats['zero_eigenvalues'] > 0:
        print(f"\nNote: {stats['zero_eigenvalues']} near-zero eigenvalues")
        print("(Indicates connected components in graph)")
    
    print("=" * 70)


def estimate_mode_energy(
    signal: torch.Tensor, 
    eigenvectors: torch.Tensor,
    energy_threshold: float = 0.95
) -> int:
    """
    Estimate number of modes needed to capture a given energy threshold.
    
    Args:
        signal: [N] or [N, C] signal to analyze
        eigenvectors: [N, K] eigenvector matrix
        energy_threshold: Fraction of energy to capture (default: 0.95)
    
    Returns:
        Number of modes needed to capture energy_threshold of total energy
    """
    if signal.dim() == 1:
        signal = signal.unsqueeze(1)
    
    # Project signal onto eigenvectors
    coeffs = eigenvectors.T @ signal  # [K, C]
    energy = (coeffs ** 2).sum(dim=1)  # [K]
    
    # Compute cumulative energy
    total_energy = energy.sum()
    cumulative = torch.cumsum(energy, dim=0) / total_energy
    
    # Find number of modes needed
    num_modes = int((cumulative >= energy_threshold).nonzero()[0].item()) + 1
    
    return num_modes


def recommend_num_modes(
    data,
    feature_index: int = 0,
    energy_thresholds: list = [0.90, 0.95, 0.99]
) -> Dict[str, int]:
    """
    Recommend number of spectral modes based on energy analysis.
    
    Args:
        data: Graph data object with x and eigenvectors
        feature_index: Which feature to analyze (0=u, 1=v, 2=f)
        energy_thresholds: Energy levels to analyze
    
    Returns:
        Dictionary mapping threshold to recommended modes
    """
    signal = data.x[:, feature_index]
    eigenvectors = data.eigenvectors
    
    recommendations = {}
    for threshold in energy_thresholds:
        num_modes = estimate_mode_energy(signal, eigenvectors, threshold)
        recommendations[f"{int(threshold*100)}%"] = num_modes
    
    return recommendations


def compare_model_outputs(
    outputs: Dict[str, torch.Tensor],
    metric: str = 'mse'
) -> Dict[str, float]:
    """
    Compare outputs from different models.
    
    Args:
        outputs: Dictionary mapping model names to output tensors
        metric: Comparison metric ('mse', 'mae', 'max_diff')
    
    Returns:
        Dictionary with pairwise comparisons
    """
    model_names = list(outputs.keys())
    comparisons = {}
    
    for i, name1 in enumerate(model_names):
        for name2 in model_names[i+1:]:
            out1 = outputs[name1]
            out2 = outputs[name2]
            
            if metric == 'mse':
                diff = ((out1 - out2) ** 2).mean().item()
            elif metric == 'mae':
                diff = (out1 - out2).abs().mean().item()
            elif metric == 'max_diff':
                diff = (out1 - out2).abs().max().item()
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            comparisons[f"{name1}_vs_{name2}"] = diff
    
    return comparisons


def validate_graph_data(data) -> Tuple[bool, list]:
    """
    Validate that graph data has all required fields for spectral models.
    
    Args:
        data: Graph data object
    
    Returns:
        (is_valid, list_of_issues)
    """
    issues = []
    
    # Check basic fields
    required_fields = ['x', 'edge_index', 'bc_mask']
    for field in required_fields:
        if not hasattr(data, field):
            issues.append(f"Missing required field: {field}")
    
    # Check spectral fields
    spectral_fields = ['eigenvectors', 'eigenvalues', 'laplacian']
    for field in spectral_fields:
        if not hasattr(data, field):
            issues.append(f"Missing spectral field: {field} (required for Phase 2 & 3)")
    
    # Check shapes
    if hasattr(data, 'x') and hasattr(data, 'eigenvectors'):
        num_nodes = data.x.shape[0]
        if data.eigenvectors.shape[0] != num_nodes:
            issues.append(f"Eigenvector shape mismatch: {data.eigenvectors.shape[0]} != {num_nodes}")
    
    if hasattr(data, 'eigenvectors') and hasattr(data, 'eigenvalues'):
        if data.eigenvectors.shape[1] != data.eigenvalues.shape[0]:
            issues.append(f"Eigenvalue count mismatch")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def print_graph_summary(data):
    """
    Print a summary of graph data.
    
    Args:
        data: Graph data object
    """
    print("=" * 70)
    print("Graph Data Summary")
    print("=" * 70)
    
    # Basic info
    print(f"Number of nodes: {data.x.shape[0]}")
    print(f"Number of features: {data.x.shape[1]}")
    
    if hasattr(data, 'edge_index'):
        print(f"Number of edges: {data.edge_index.shape[1]}")
    
    if hasattr(data, 'bc_mask'):
        num_boundary = data.bc_mask.sum().item()
        print(f"Boundary nodes: {num_boundary}")
        print(f"Interior nodes: {data.x.shape[0] - num_boundary}")
    
    # Spectral info
    if hasattr(data, 'eigenvectors'):
        print(f"\nSpectral information:")
        print(f"  Eigenvectors: {data.eigenvectors.shape}")
        print(f"  Number of modes: {data.eigenvectors.shape[1]}")
        
        if hasattr(data, 'eigenvalues'):
            print(f"  Eigenvalue range: [{data.eigenvalues.min():.6f}, {data.eigenvalues.max():.6f}]")
    
    # Validation
    is_valid, issues = validate_graph_data(data)
    if is_valid:
        print("\n✓ Graph data is valid for all phases")
    else:
        print("\n⚠ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    
    print("=" * 70)


# Export all utilities
__all__ = [
    'get_model_info',
    'print_model_summary',
    'analyze_eigenspectrum',
    'print_eigenspectrum_summary',
    'estimate_mode_energy',
    'recommend_num_modes',
    'compare_model_outputs',
    'validate_graph_data',
    'print_graph_summary',
]
