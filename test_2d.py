import torch
import numpy as np
from scipy.interpolate import griddata

def rollout_test_2d(model, gt_data, n_intervals=10, dt_interval=1.0, nt_per_interval=10, 
                    self_feeding=False):
    """
    Perform rollout testing for 2D wave equation with self-feeding capability
    
    Args:
        model: Trained PINNDeepONet_Wave2D instance
        gt_data: MATLAB ground truth with columns [x, y, t, f, u, v]
        n_intervals: Number of prediction intervals
        dt_interval: Time step between intervals (s)
        nt_per_interval: Number of time steps per interval for prediction
        self_feeding: If True, use previous prediction as initial condition for next interval
    
    Returns:
        results: Dictionary with predictions and errors for each interval
    """
    
    print("\n" + "="*60)
    print("STARTING 2D ROLLOUT TEST")
    print("="*60)
    print(f"Number of intervals: {n_intervals}")
    print(f"Time step between intervals: {dt_interval}s")
    print(f"Time steps per interval: {nt_per_interval}")
    print(f"Total simulation time: {n_intervals * dt_interval}s")
    print(f"Self-feeding: {self_feeding}")
    
    # Extract ground truth columns
    x_gt = gt_data[:, 0]
    y_gt = gt_data[:, 1]
    t_gt = gt_data[:, 2]
    f_gt = gt_data[:, 3]
    u_gt = gt_data[:, 4]
    v_gt = gt_data[:, 5]
    
    # Get unique coordinate values
    t_unique = np.unique(t_gt)
    x_unique = np.unique(x_gt)
    y_unique = np.unique(y_gt)
    
    print(f"\nGround truth time range: [{t_unique.min():.2f}, {t_unique.max():.2f}]s")
    print(f"Ground truth spatial grid: {len(x_unique)} x {len(y_unique)} points")
    
    # Get sensor grid coordinates from model
    x_sensors = model.sensor_x_ic.numpy()
    y_sensors = model.sensor_y_ic.numpy()
    
    # Storage for results
    results = {
        'times': [],
        'u_pred': [],
        'u_gt_interp': [],
        'errors': [],
        'rel_errors': [],
        'x_coords': [],
        'y_coords': []
    }
    
    model.eval()
    
    # Rollout loop
    for i in range(n_intervals):
        t_current = i * dt_interval
        print(f"\n--- Interval {i+1}/{n_intervals} (t = {t_current:.2f}s) ---")
        
        # Find closest time in ground truth
        t_idx = np.argmin(np.abs(t_unique - t_current))
        t_selected = t_unique[t_idx]
        print(f"Using ground truth at t = {t_selected:.4f}s")
        
        # Extract data at current time
        mask = np.abs(t_gt - t_selected) < 1e-6
        x_current = x_gt[mask]
        y_current = y_gt[mask]
        f_current = f_gt[mask]
        u_current = u_gt[mask]
        v_current = v_gt[mask]
        
        print(f"Found {len(x_current)} spatial points at this time")
        
        # Interpolate to sensor locations
        points = np.column_stack([x_current, y_current])
        sensor_points = np.column_stack([x_sensors, y_sensors])
        
        if self_feeding and i > 0:
            # Use previous prediction as initial condition
            print("Using self-feeding: previous prediction as IC")
            u0_sensors = torch.tensor(u_pred_last, dtype=torch.float32).numpy()
            v0_sensors = torch.tensor(v_pred_last, dtype=torch.float32).numpy()
        else:
            # Use ground truth as initial condition
            u0_sensors = griddata(points, u_current, sensor_points, method='linear')
            v0_sensors = griddata(points, v_current, sensor_points, method='linear')
            u0_sensors = np.nan_to_num(u0_sensors, nan=0.0)
            v0_sensors = np.nan_to_num(v0_sensors, nan=0.0)
        
        # Source is always from ground truth
        src_sensors = griddata(points, f_current, sensor_points, method='linear')
        src_sensors = np.nan_to_num(src_sensors, nan=0.0)
        
        print(f"u0_sensors range: [{u0_sensors.min():.6f}, {u0_sensors.max():.6f}]")
        print(f"v0_sensors range: [{v0_sensors.min():.6f}, {v0_sensors.max():.6f}]")
        print(f"src_sensors range: [{src_sensors.min():.6f}, {src_sensors.max():.6f}]")
        
        # Convert to tensors (batch size 1)
        u0_sensors_tensor = torch.tensor(u0_sensors, dtype=torch.float32).unsqueeze(0)
        v0_sensors_tensor = torch.tensor(v0_sensors, dtype=torch.float32).unsqueeze(0)
        src_sensors_tensor = torch.tensor(src_sensors, dtype=torch.float32).unsqueeze(0)
        
        # Create spatiotemporal prediction grid for this interval
        t_pred = np.linspace(0, dt_interval, nt_per_interval)
        X_pred, Y_pred, T_pred = np.meshgrid(x_unique, y_unique, t_pred, indexing='ij')
        xyt_pred = torch.tensor(
            np.column_stack([X_pred.flatten(), Y_pred.flatten(), T_pred.flatten()]),
            dtype=torch.float32
        )
        
        # Model prediction
        with torch.no_grad():
            u_pred = model.forward(u0_sensors_tensor, v0_sensors_tensor, 
                                  src_sensors_tensor, xyt_pred)
            u_pred_np = u_pred.squeeze().numpy()
        
        # Get prediction at the end of interval (last time step)
        u_pred_grid = u_pred_np.reshape(len(x_unique), len(y_unique), nt_per_interval)
        u_pred_last = u_pred_grid[:, :, -1].flatten()  # For next iteration
        u_pred_final = u_pred_last  # Use only final time step for comparison
        
        # Get ground truth at end of this interval
        t_next = t_current + dt_interval
        t_next_idx = np.argmin(np.abs(t_unique - t_next))
        t_next_gt = t_unique[t_next_idx]
        
        mask_next = np.abs(t_gt - t_next_gt) < 1e-6
        x_next = x_gt[mask_next]
        y_next = y_gt[mask_next]
        u_next_gt = u_gt[mask_next]
        
        print(f"u_pred (final time) range: [{u_pred_final.min():.6f}, {u_pred_final.max():.6f}]")
        print(f"u_gt range: [{u_next_gt.min():.6f}, {u_next_gt.max():.6f}]")
        
        # Compute error
        error = np.abs(u_pred_final - u_next_gt)
        rel_error = np.linalg.norm(error) / (np.linalg.norm(u_next_gt) + 1e-10)
        
        print(f"Max absolute error: {np.max(error):.6e}")
        print(f"Mean absolute error: {np.mean(error):.6e}")
        print(f"Relative L2 error: {rel_error:.6e}")
        
        # Store results
        results['times'].append(t_next)
        results['u_pred'].append(u_pred_final)
        results['u_gt_interp'].append(u_next_gt)
        results['errors'].append(error)
        results['rel_errors'].append(rel_error)
        results['x_coords'].append(x_next)
        results['y_coords'].append(y_next)
        
        # Compute velocity prediction for next iteration (if using self-feeding)
        if self_feeding and i < n_intervals - 1:
            print("Computing velocity for next interval...")
            xyt_grad = xyt_pred.clone().requires_grad_(True)
            u_grad = model.forward(u0_sensors_tensor, v0_sensors_tensor, 
                                  src_sensors_tensor, xyt_grad)
            v_pred = torch.autograd.grad(u_grad, xyt_grad, 
                                        torch.ones_like(u_grad), create_graph=False)[0][:, 2]
            v_pred_grid = v_pred.numpy().reshape(len(x_unique), len(y_unique), nt_per_interval)
            v_pred_last = v_pred_grid[:, :, -1].flatten()  # For next iteration
            print("Velocity computation done.")
    
    # Summary statistics
    print("\n" + "="*60)
    print("ROLLOUT TEST SUMMARY")
    print("="*60)
    print(f"Average relative L2 error: {np.mean(results['rel_errors']):.6e}")
    print(f"Max relative L2 error: {np.max(results['rel_errors']):.6e}")
    print(f"Min relative L2 error: {np.min(results['rel_errors']):.6e}")
    print("="*60 + "\n")
    
    return results


def plot_rollout_errors_2d(results):
    """Plot rollout error evolution over time"""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Error over time
    ax1.semilogy(results['times'], results['rel_errors'], 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Relative L2 Error', fontsize=12)
    ax1.set_title('2D Rollout Test: Error Evolution', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Max error per interval
    max_errors = [np.max(e) for e in results['errors']]
    ax2.semilogy(results['times'], max_errors, 'rs-', linewidth=2, markersize=8)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Max Absolute Error', fontsize=12)
    ax2.set_title('2D Rollout Test: Max Error per Interval', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_rollout_snapshots_2d(results, snapshot_indices=[0, -1]):
    """
    Plot snapshots of predictions vs ground truth at selected intervals.
    Optimized: skip interpolation, direct scatter plot on original points.
    
    Args:
        results: Dictionary from rollout_test_2d
        snapshot_indices: List of interval indices to plot (default: first and last)
    """
    import matplotlib.pyplot as plt
    
    n_snapshots = min(len(snapshot_indices), len(results['times']))
    fig = plt.figure(figsize=(16, 5*n_snapshots))
    
    for plot_idx, interval_idx in enumerate(snapshot_indices):
        # Handle negative indices
        if interval_idx < 0:
            interval_idx = len(results['times']) + interval_idx
        
        if interval_idx >= len(results['times']) or interval_idx < 0:
            continue
        
        t_val = results['times'][interval_idx]
        u_pred = results['u_pred'][interval_idx]
        u_gt = results['u_gt_interp'][interval_idx]
        error = results['errors'][interval_idx]
        x_coords = results['x_coords'][interval_idx]
        y_coords = results['y_coords'][interval_idx]
        rel_l2 = results['rel_errors'][interval_idx]
        
        # Determine color scale
        u_min = min(np.min(u_pred), np.min(u_gt))
        u_max = max(np.max(u_pred), np.max(u_gt))
        
        # Direct scatter plots (no interpolation)
        ax1 = plt.subplot(n_snapshots, 3, plot_idx*3 + 1)
        sc1 = ax1.scatter(x_coords, y_coords, c=u_pred, s=20, cmap='RdBu_r', 
                         vmin=u_min, vmax=u_max, edgecolors='none')
        ax1.set_title(f'PINN Prediction (t={t_val:.2f}s)', fontsize=11, fontweight='bold')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_aspect('equal')
        cbar1 = plt.colorbar(sc1, ax=ax1)
        cbar1.set_label('u')
        
        # Ground Truth
        ax2 = plt.subplot(n_snapshots, 3, plot_idx*3 + 2)
        sc2 = ax2.scatter(x_coords, y_coords, c=u_gt, s=20, cmap='RdBu_r',
                         vmin=u_min, vmax=u_max, edgecolors='none')
        ax2.set_title(f'Ground Truth (t={t_val:.2f}s)', fontsize=11, fontweight='bold')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_aspect('equal')
        cbar2 = plt.colorbar(sc2, ax=ax2)
        cbar2.set_label('u')
        
        # Error
        ax3 = plt.subplot(n_snapshots, 3, plot_idx*3 + 3)
        sc3 = ax3.scatter(x_coords, y_coords, c=error, s=20, cmap='hot', edgecolors='none')
        ax3.set_title(f'Absolute Error\n(Rel L2: {rel_l2:.2e})', fontsize=11, fontweight='bold')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_aspect('equal')
        cbar3 = plt.colorbar(sc3, ax=ax3)
        cbar3.set_label('|error|')
    
    plt.tight_layout()
    return fig


def create_animation_2d(results, output_file='rollout_animation.mp4', fps=2, interval_indices=None):
    """
    Create animation of 2D wave propagation over rollout intervals.
    Optimized: skip interpolation, use scatter plots on original data points.
    
    Args:
        results: Dictionary from rollout_test_2d
        output_file: Path to save animation (mp4 or gif)
        fps: Frames per second (default: 2 for faster generation)
        interval_indices: List of indices to animate (default: all)
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import Normalize
    
    if interval_indices is None:
        interval_indices = list(range(len(results['times'])))
    
    print(f"Creating animation with {len(interval_indices)} frames...")
    
    # Get consistent color scale
    all_u_pred = np.concatenate(results['u_pred'])
    all_u_gt = np.concatenate(results['u_gt_interp'])
    u_min = min(np.min(all_u_pred), np.min(all_u_gt))
    u_max = max(np.max(all_u_pred), np.max(all_u_gt))
    norm = Normalize(vmin=u_min, vmax=u_max)
    
    # Setup figure with 3 subplots
    fig = plt.figure(figsize=(15, 5))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)
    
    def animate(frame_idx):
        """Update function for animation - only plot, no interpolation"""
        interval_idx = interval_indices[frame_idx]
        t_val = results['times'][interval_idx]
        u_pred = results['u_pred'][interval_idx]
        u_gt = results['u_gt_interp'][interval_idx]
        error = results['errors'][interval_idx]
        x_coords = results['x_coords'][interval_idx]
        y_coords = results['y_coords'][interval_idx]
        rel_l2 = results['rel_errors'][interval_idx]
        
        # Clear previous
        ax1.clear()
        ax2.clear()
        ax3.clear()
        
        # Direct scatter plots (no interpolation)
        sc1 = ax1.scatter(x_coords, y_coords, c=u_pred, s=15, cmap='RdBu_r', 
                         norm=norm, edgecolors='none')
        ax1.set_title(f'PINN Prediction\n(t={t_val:.2f}s)', fontsize=11, fontweight='bold')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_aspect('equal')
        
        # Plot ground truth
        sc2 = ax2.scatter(x_coords, y_coords, c=u_gt, s=15, cmap='RdBu_r',
                         norm=norm, edgecolors='none')
        ax2.set_title(f'Ground Truth\n(t={t_val:.2f}s)', fontsize=11, fontweight='bold')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_aspect('equal')
        
        # Plot error
        sc3 = ax3.scatter(x_coords, y_coords, c=error, s=15, cmap='hot', edgecolors='none')
        ax3.set_title(f'Error\n(Rel L2: {rel_l2:.2e})', fontsize=10, fontweight='bold')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_aspect('equal')
        
        # Print progress
        if frame_idx % max(1, len(interval_indices)//5) == 0:
            print(f"  Frame {frame_idx+1}/{len(interval_indices)}")
        
        return [sc1, sc2, sc3]
    
    anim = animation.FuncAnimation(fig, animate, frames=len(interval_indices), 
                                   interval=1000/fps, repeat=True, blit=False)
    
    # Save animation
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=1200)
        anim.save(output_file, writer=writer)
        print(f"✓ Animation saved to {output_file}")
    except RuntimeError as e:
        print(f"Warning: Could not save as mp4: {e}")
        print(f"Saving as gif instead...")
        gif_file = output_file.replace('.mp4', '.gif')
        anim.save(gif_file, writer='pillow', dpi=80)
        print(f"✓ Animation saved to {gif_file}")
    
    return fig, anim