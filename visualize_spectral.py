"""
Visualization utilities for spectral graph methods.

This script helps visualize:
1. Laplacian eigenvectors (Fourier basis on the graph)
2. Frequency spectrum (eigenvalues)
3. How different modes capture different patterns
"""

import numpy as np
import matplotlib.pyplot as plt
from dataset import create_graph
from omegaconf import OmegaConf


def plot_eigenvectors(data, num_to_plot=8, save_path=None):
    """
    Plot the first few Laplacian eigenvectors.
    
    Low-index eigenvectors = smooth, global patterns
    High-index eigenvectors = oscillating, local patterns
    
    Args:
        data: Graph data object with eigenvectors and eigenvalues
        num_to_plot: Number of eigenvectors to visualize
        save_path: Optional path to save the figure
    """
    eigenvectors = data.eigenvectors.numpy()
    eigenvalues = data.eigenvalues.numpy()
    coords = data.coords.numpy().flatten()
    
    num_modes = min(num_to_plot, eigenvectors.shape[1])
    
    fig, axes = plt.subplots(2, num_modes // 2, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(num_modes):
        ax = axes[i]
        ax.plot(coords, eigenvectors[:, i], 'b-', linewidth=2)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_title(f'Mode {i}\nλ = {eigenvalues[i]:.4f}')
        ax.set_xlabel('Position')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    return fig


def plot_eigenvalue_spectrum(data, save_path=None):
    """
    Plot the eigenvalue spectrum.
    
    Shows which frequencies are present in the graph Laplacian.
    Small eigenvalues = global modes
    Large eigenvalues = local modes
    
    Args:
        data: Graph data object with eigenvalues
        save_path: Optional path to save the figure
    """
    eigenvalues = data.eigenvalues.numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Eigenvalue magnitudes
    ax1.plot(eigenvalues, 'bo-', linewidth=2, markersize=4)
    ax1.set_xlabel('Mode Index')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title('Laplacian Eigenvalue Spectrum')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 2: Eigenvalue gaps (differences)
    gaps = np.diff(eigenvalues)
    ax2.plot(gaps, 'ro-', linewidth=2, markersize=4)
    ax2.set_xlabel('Mode Index')
    ax2.set_ylabel('Eigenvalue Gap')
    ax2.set_title('Spectral Gaps\n(larger gaps = distinct frequency bands)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    return fig


def visualize_frequency_filtering(data, feature_index=0, num_modes_list=[5, 10, 20, 50]):
    """
    Demonstrate how different numbers of modes affect signal reconstruction.
    
    Shows what happens when we:
    1. Transform signal to frequency domain
    2. Keep only first K modes
    3. Transform back to spatial domain
    
    Args:
        data: Graph data object
        feature_index: Which feature to visualize (0=u, 1=v, 2=f)
        num_modes_list: List of mode counts to compare
    """
    import torch
    
    signal = data.x[:, feature_index].numpy()
    coords = data.coords.numpy().flatten()
    eigenvectors = data.eigenvectors
    
    fig, axes = plt.subplots(len(num_modes_list) + 1, 1, figsize=(10, 10))
    
    # Original signal
    axes[0].plot(coords, signal, 'k-', linewidth=2, label='Original')
    axes[0].set_title('Original Signal')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Reconstruct with different numbers of modes
    for i, num_modes in enumerate(num_modes_list, 1):
        U = eigenvectors[:, :num_modes]
        
        # Project to frequency domain and back
        signal_torch = torch.tensor(signal, dtype=torch.float32)
        coeffs = U.T @ signal_torch.unsqueeze(1)  # Frequency coefficients
        reconstructed = (U @ coeffs).squeeze().numpy()  # Back to spatial
        
        # Plot
        axes[i].plot(coords, signal, 'k-', linewidth=1, alpha=0.3, label='Original')
        axes[i].plot(coords, reconstructed, 'b-', linewidth=2, label=f'{num_modes} modes')
        
        # Compute error
        error = np.abs(signal - reconstructed).mean()
        axes[i].set_title(f'Reconstruction with {num_modes} modes (MAE: {error:.6f})')
        axes[i].set_ylabel('Amplitude')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    axes[-1].set_xlabel('Position')
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_mode_energy(data, feature_index=0):
    """
    Plot energy in different frequency modes.
    
    Shows which frequencies contain the most signal energy.
    
    Args:
        data: Graph data object
        feature_index: Which feature to analyze
    """
    import torch
    
    signal = data.x[:, feature_index]
    eigenvectors = data.eigenvectors
    
    # Compute frequency coefficients
    coeffs = eigenvectors.T @ signal.unsqueeze(1)
    energy = (coeffs ** 2).squeeze().numpy()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Energy per mode
    ax1.bar(range(len(energy)), energy, alpha=0.7)
    ax1.set_xlabel('Mode Index')
    ax1.set_ylabel('Energy')
    ax1.set_title('Energy Distribution Across Frequency Modes')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Cumulative energy
    cumulative = np.cumsum(energy) / np.sum(energy) * 100
    ax2.plot(cumulative, 'b-', linewidth=2)
    ax2.axhline(90, color='r', linestyle='--', label='90% energy')
    ax2.axhline(95, color='g', linestyle='--', label='95% energy')
    ax2.axhline(99, color='orange', linestyle='--', label='99% energy')
    ax2.set_xlabel('Number of Modes')
    ax2.set_ylabel('Cumulative Energy (%)')
    ax2.set_title('Cumulative Energy Capture')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Find modes needed for different energy thresholds
    for threshold in [90, 95, 99]:
        idx = np.argmax(cumulative >= threshold)
        print(f"{threshold}% energy captured by {idx} modes (out of {len(energy)})")
    
    plt.tight_layout()
    plt.show()
    
    return fig


def main():
    """Main visualization demo."""
    print("Spectral Graph Methods Visualization")
    print("=" * 50)
    
    # Create config
    cfg = OmegaConf.create({
        'model': {
            'num_spectral_modes': 50,
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
    
    # Create graph with eigendecomposition
    print("\nCreating graph and computing eigendecomposition...")
    data = create_graph(seed=42, cfg=cfg)
    
    print(f"Graph: {data.x.shape[0]} nodes")
    print(f"Eigenvectors computed: {data.eigenvectors.shape[1]} modes")
    print(f"Eigenvalue range: [{data.eigenvalues.min():.4f}, {data.eigenvalues.max():.4f}]")
    
    # Visualization 1: First few eigenvectors
    print("\n1. Plotting Laplacian eigenvectors...")
    plot_eigenvectors(data, num_to_plot=8)
    
    # Visualization 2: Eigenvalue spectrum
    print("\n2. Plotting eigenvalue spectrum...")
    plot_eigenvalue_spectrum(data)
    
    # Visualization 3: Frequency filtering
    print("\n3. Demonstrating frequency-domain filtering...")
    visualize_frequency_filtering(data, feature_index=0, num_modes_list=[5, 10, 20, 50])
    
    # Visualization 4: Mode energy
    print("\n4. Analyzing energy distribution across modes...")
    plot_mode_energy(data, feature_index=0)
    
    print("\n" + "=" * 50)
    print("Visualization complete!")
    print("\nKey Observations:")
    print("- Low-index modes (0, 1, 2...): Smooth, global patterns")
    print("- High-index modes: Oscillating, local patterns")
    print("- Most energy typically in first few modes")
    print("- This is why spectral methods work well!")


if __name__ == "__main__":
    # Check if matplotlib is available
    try:
        import matplotlib
        matplotlib.use('TkAgg')  # or 'Qt5Agg' depending on your system
        main()
    except ImportError as e:
        print("Error: This script requires matplotlib")
        print("Install with: pip install matplotlib")
        print(f"Error details: {e}")
