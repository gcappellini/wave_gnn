"""
Utility functions for plotting mesh-based features as animations.
"""

import numpy as np
import matplotlib.pyplot as plt



def plot_features_2d(
    nodes,
    histories,
    dt,
    feature_names=['Deformation', 'Velocity', 'Force'],
    output_file='features.png',
    figsize=None,
    colormaps=None,
    show_colorbar=True,
    dpi=150,
    xlabel='Space (x)',
    ylabel='Time (s)',
    ylabel_as_int=False,
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
    xlabel : str, default='Space (x)'
        Label for the x-axis
    ylabel : str, default='Time (s)'
        Label for the y-axis
    ylabel_as_int : bool, default=False
        If True, display y-axis values as integers
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
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel(ylabel)
        # Set y-ticks in seconds or integers
        axes[i].set_yticks(time_ticks)
        if ylabel_as_int:
            axes[i].set_yticklabels([f"{int(t)}" for t in time_ticks])
        else:
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


def plot_loss_history(
    loss_history,
    output_file='loss_history.png',
    figsize=(14, 10),
    dpi=150,
    show_weights=True,
):
    """
    Plot training and validation loss history with all loss components.
    
    Parameters
    ----------
    loss_history : list of dict
        Loss history records from training, each dict contains:
        - epoch, train_total, train_PI1, train_PI2, val_total, val_PI1, val_PI2
        - Optional: train_Energy, train_RK4_1, train_RK4_2
        - Optional: weight_PI_loss1, weight_PI_loss2, etc. (if adaptive weights)
    output_file : str, default='loss_history.png'
        Output filename for the figure
    figsize : tuple, default=(14, 10)
        Figure size as (width, height)
    dpi : int, default=150
        Resolution of the saved figure
    show_weights : bool, default=True
        Whether to plot adaptive weights if available
    """
    if not loss_history:
        print("Warning: Empty loss history, skipping plot")
        return
    
    # Extract data
    epochs = [r['epoch'] for r in loss_history]
    
    # Check which loss components are available
    has_energy = 'train_Energy' in loss_history[0] and loss_history[0]['train_Energy'] is not None
    has_rk4 = 'train_RK4_1' in loss_history[0] and loss_history[0]['train_RK4_1'] is not None
    has_weights = any(k.startswith('weight_') for k in loss_history[0].keys())
    
    # Determine subplot layout
    n_plots = 2  # Always have train and val plots
    if has_weights and show_weights:
        n_plots = 3
    
    fig = plt.figure(figsize=figsize)
    
    # ========== Plot 1: Training Losses ==========
    ax1 = plt.subplot(n_plots, 1, 1)
    
    train_total = [r['train_total'] for r in loss_history]
    train_PI1 = [r['train_PI1'] for r in loss_history]
    train_PI2 = [r['train_PI2'] for r in loss_history]
    
    ax1.plot(epochs, train_total, 'k-', linewidth=2, label='Total', alpha=0.8)
    ax1.plot(epochs, train_PI1, 'b--', linewidth=1.5, label='PI1', alpha=0.7)
    ax1.plot(epochs, train_PI2, 'r--', linewidth=1.5, label='PI2', alpha=0.7)
    
    if has_energy:
        train_energy = [r['train_Energy'] for r in loss_history if r['train_Energy'] is not None]
        epochs_energy = [r['epoch'] for r in loss_history if r['train_Energy'] is not None]
        if train_energy:
            ax1.plot(epochs_energy, train_energy, 'g-.', linewidth=1.5, label='Energy', alpha=0.7)
    
    if has_rk4:
        train_rk4_1 = [r['train_RK4_1'] for r in loss_history if r['train_RK4_1'] is not None]
        train_rk4_2 = [r['train_RK4_2'] for r in loss_history if r['train_RK4_2'] is not None]
        epochs_rk4 = [r['epoch'] for r in loss_history if r['train_RK4_1'] is not None]
        if train_rk4_1:
            ax1.plot(epochs_rk4, train_rk4_1, 'c:', linewidth=1.5, label='RK4_1', alpha=0.7)
            ax1.plot(epochs_rk4, train_rk4_2, 'm:', linewidth=1.5, label='RK4_2', alpha=0.7)
    
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Training Loss', fontsize=11)
    ax1.set_title('Training Loss Components', fontsize=13, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', framealpha=0.9, fontsize=9)
    
    # ========== Plot 2: Validation Losses ==========
    ax2 = plt.subplot(n_plots, 1, 2)
    
    val_total = [r['val_total'] for r in loss_history]
    val_PI1 = [r['val_PI1'] for r in loss_history]
    val_PI2 = [r['val_PI2'] for r in loss_history]
    
    ax2.plot(epochs, val_total, 'k-', linewidth=2, label='Total', alpha=0.8)
    ax2.plot(epochs, val_PI1, 'b--', linewidth=1.5, label='PI1', alpha=0.7)
    ax2.plot(epochs, val_PI2, 'r--', linewidth=1.5, label='PI2', alpha=0.7)
    
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Validation Loss', fontsize=11)
    ax2.set_title('Validation Loss Components', fontsize=13, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', framealpha=0.9, fontsize=9)
    
    # ========== Plot 3: Adaptive Weights (if available) ==========
    if has_weights and show_weights:
        ax3 = plt.subplot(n_plots, 1, 3)
        
        # Collect all weight keys
        weight_keys = [k for k in loss_history[0].keys() if k.startswith('weight_')]
        
        colors = ['b', 'r', 'g', 'c', 'm', 'y']
        linestyles = ['-', '--', '-.', ':']
        
        for i, key in enumerate(weight_keys):
            weights = [r[key] for r in loss_history]
            color = colors[i % len(colors)]
            linestyle = linestyles[i % len(linestyles)]
            label = key.replace('weight_', '')
            ax3.plot(epochs, weights, color=color, linestyle=linestyle, 
                    linewidth=1.5, label=label, alpha=0.8)
        
        ax3.set_xlabel('Epoch', fontsize=11)
        ax3.set_ylabel('Weight Value', fontsize=11)
        ax3.set_title('Adaptive Loss Weights Evolution', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.legend(loc='best', framealpha=0.9, fontsize=9, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved loss history plot to '{output_file}'")