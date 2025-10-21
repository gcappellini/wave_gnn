"""
Utility functions for plotting mesh-based features as animations.
"""

import numpy as np
import matplotlib.pyplot as plt



def plot_features_2d(
    nodes,
    histories,
    feature_names=['Deformation', 'Velocity', 'Force'],
    output_file='features.png',
    figsize=None,
    colormaps=None,
    show_colorbar=True,
    dpi=150,
    dt =0.01
):
    """
    Plot full 2D distributions (space vs time) for each feature.
    Automatically arranges subplots in rows of 3 when more than 3 features.

    Parameters
    ----------
    nodes : np.ndarray
        1D array of spatial coordinates, shape [N_nodes]
    histories : np.ndarray
        Feature values over time, shape [N_features, N_timesteps, N_nodes]
    feature_names : list of str, optional
        Names for each feature. If None, uses 'Feature 1', 'Feature 2', etc.
    output_file : str, default='features.png'
        Output filename for the figure
    figsize : tuple, optional
        Figure size as (width, height). If None, auto-computed based on layout.
    colormaps : list of str, optional
        Colormap for each feature. If None, all use 'viridis'
    show_colorbar : bool, default=True
        Whether to show colorbars for each subplot
    dpi : int, default=150
        Resolution of the saved figure
    """
    n_features, n_timesteps, n_nodes = histories.shape

    # Default feature names
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(n_features)]

    # Default colormaps
    if colormaps is None:
        colormaps = ['viridis'] * n_features
    elif len(colormaps) != n_features:
        raise ValueError("Number of colormaps must match number of features")

    # Calculate grid layout: max 3 columns per row
    n_cols = min(3, n_features)
    n_rows = int(np.ceil(n_features / n_cols))
    
    # Auto-compute figure size if not provided
    if figsize is None:
        # Base width per subplot (4 inches) and height (3.5 inches)
        figsize = (n_cols * 4, n_rows * 3.5)

    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    # Flatten axes for easy indexing
    axes = axes.flatten()

    # Convert time steps to seconds
    time_end = (n_timesteps - 1) * dt
    # Choose up to 6 ticks for readability
    tick_count = min(6, n_timesteps)
    time_ticks = np.linspace(0.0, time_end, tick_count)

    for i in range(n_features):
        im = axes[i].imshow(
            histories[i, :, :],
            extent=[nodes.min(), nodes.max(), 0.0, time_end],
            origin='lower',
            aspect='auto',
            cmap=colormaps[i]
        )
        axes[i].set_title(feature_names[i])
        axes[i].set_xlabel("Space (x)")
        axes[i].set_ylabel("Time (s)")
        # Set y-ticks in seconds
        axes[i].set_yticks(time_ticks)
        axes[i].set_yticklabels([f"{t:.3f}" for t in time_ticks])
        if show_colorbar:
            fig.colorbar(im, ax=axes[i], orientation='vertical', shrink=0.8)
    
    # Hide unused subplots if n_features < total grid size
    for i in range(n_features, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved static 2D feature map with {n_features} features ({n_rows}x{n_cols} layout) as '{output_file}'")

