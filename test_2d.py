import torch
import numpy as np
from scipy.interpolate import griddata

def rollout_test_2d(model, gt_data, n_intervals=10, dt_interval=1.0):
    """
    Perform rollout testing for 2D wave equation using ground truth sensor extraction
    
    Args:
        model: Trained PINNDeepONet_Wave2D instance
        gt_data: MATLAB ground truth with columns [x, y, t, f, u, v]
        n_intervals: Number of prediction intervals
        dt_interval: Time step between intervals (s)
    
    Returns:
        results: Dictionary with predictions and errors for each interval
    """
    
    print("\n" + "="*60)
    print("STARTING 2D ROLLOUT TEST")
    print("="*60)
    print(f"Number of intervals: {n_intervals}")
    print(f"Time step: {dt_interval}s")
    print(f"Total simulation time: {n_intervals * dt_interval}s")
    
    # Extract ground truth columns
    x_gt = gt_data[:, 0]
    y_gt = gt_data[:, 1]
    t_gt = gt_data[:, 2]
    f_gt = gt_data[:, 3]
    u_gt = gt_data[:, 4]
    v_gt = gt_data[:, 5]
    
    # Get unique time values
    t_unique = np.unique(t_gt)
    print(f"\nGround truth time range: [{t_unique.min():.2f}, {t_unique.max():.2f}]s")
    print(f"Number of time snapshots: {len(t_unique)}")
    
    # Get sensor grid coordinates from model
    x_sensors = model.x_sensors_ic.numpy()
    y_sensors = model.y_sensors_ic.numpy()
    
    # Storage for results
    results = {
        'times': [],
        'u_pred': [],
        'u_gt_interp': [],
        'errors': [],
        'rel_errors': []
    }
    
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
        
        u0_sensors = griddata(points, u_current, sensor_points, method='linear')
        v0_sensors = griddata(points, v_current, sensor_points, method='linear')
        src_sensors = griddata(points, f_current, sensor_points, method='linear')
        
        # Handle potential NaN from extrapolation
        u0_sensors = np.nan_to_num(u0_sensors, nan=0.0)
        v0_sensors = np.nan_to_num(v0_sensors, nan=0.0)
        src_sensors = np.nan_to_num(src_sensors, nan=0.0)
        
        # Convert to tensors
        u0_sensors_tensor = torch.tensor(u0_sensors, dtype=torch.float32).unsqueeze(0)
        v0_sensors_tensor = torch.tensor(v0_sensors, dtype=torch.float32).unsqueeze(0)
        src_sensors_tensor = torch.tensor(src_sensors, dtype=torch.float32).unsqueeze(0)
        
        print(f"u0_sensors range: [{u0_sensors.min():.6f}, {u0_sensors.max():.6f}]")
        print(f"v0_sensors range: [{v0_sensors.min():.6f}, {v0_sensors.max():.6f}]")
        print(f"src_sensors range: [{src_sensors.min():.6f}, {src_sensors.max():.6f}]")
        
        # Predict at next time step
        t_next = (i + 1) * dt_interval
        
        # Create prediction grid (use ground truth spatial points)
        t_next_idx = np.argmin(np.abs(t_unique - t_next))
        t_next_gt = t_unique[t_next_idx]
        
        mask_next = np.abs(t_gt - t_next_gt) < 1e-6
        x_next = x_gt[mask_next]
        y_next = y_gt[mask_next]
        u_next_gt = u_gt[mask_next]
        
        # Create input tensor for prediction
        xyt_pred = torch.tensor(
            np.column_stack([x_next, y_next, np.full_like(x_next, dt_interval)]),
            dtype=torch.float32
        )
        
        # Model prediction
        with torch.no_grad():
            u_pred = model.forward(u0_sensors_tensor, v0_sensors_tensor, 
                                  src_sensors_tensor, xyt_pred)
            u_pred_np = u_pred.squeeze().numpy()
        
        print(f"u_pred range: [{u_pred_np.min():.6f}, {u_pred_np.max():.6f}]")
        print(f"u_gt range: [{u_next_gt.min():.6f}, {u_next_gt.max():.6f}]")
        
        # Compute error
        error = np.abs(u_pred_np - u_next_gt)
        rel_error = np.linalg.norm(error) / (np.linalg.norm(u_next_gt) + 1e-10)
        
        print(f"Max absolute error: {np.max(error):.6e}")
        print(f"Mean absolute error: {np.mean(error):.6e}")
        print(f"Relative L2 error: {rel_error:.6e}")
        
        # Store results
        results['times'].append(t_next)
        results['u_pred'].append(u_pred_np)
        results['u_gt_interp'].append(u_next_gt)
        results['errors'].append(error)
        results['rel_errors'].append(rel_error)
    
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
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.semilogy(results['times'], results['rel_errors'], 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Relative L2 Error', fontsize=12)
    ax.set_title('2D Rollout Test: Error Evolution', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_rollout_snapshots_2d(model, results, gt_data, snapshot_indices=[0, 4, 9]):
    """Plot snapshots of predictions vs ground truth at selected intervals"""
    import matplotlib.pyplot as plt
    
    n_snapshots = len(snapshot_indices)
    fig = plt.figure(figsize=(18, 4*n_snapshots))
    
    # Extract ground truth grid info
    x_gt = gt_data[:, 0]
    y_gt = gt_data[:, 1]
    t_gt = gt_data[:, 2]
    u_gt = gt_data[:, 4]
    
    t_unique = np.unique(t_gt)
    x_unique = np.unique(x_gt)
    y_unique = np.unique(y_gt)
    
    for plot_idx, interval_idx in enumerate(snapshot_indices):
        if interval_idx >= len(results['times']):
            continue
        
        t_val = results['times'][interval_idx]
        u_pred = results['u_pred'][interval_idx]
        u_gt_interp = results['u_gt_interp'][interval_idx]
        error = results['errors'][interval_idx]
        
        # Find ground truth at this time
        t_idx = np.argmin(np.abs(t_unique - t_val))
        mask = np.abs(t_gt - t_unique[t_idx]) < 1e-6
        
        x_snap = x_gt[mask]
        y_snap = y_gt[mask]
        
        # Create grids for plotting
        X, Y = np.meshgrid(x_unique, y_unique, indexing='ij')
        points = np.column_stack([x_snap, y_snap])
        
        u_pred_grid = griddata(points, u_pred, (X, Y), method='linear')
        u_gt_grid = griddata(points, u_gt_interp, (X, Y), method='linear')
        error_grid = griddata(points, error, (X, Y), method='linear')
        
        # Plot row
        ax1 = plt.subplot(n_snapshots, 3, plot_idx*3 + 1)
        im1 = ax1.contourf(X, Y, u_pred_grid, levels=20, cmap='RdBu_r')
        ax1.set_title(f'PINN Prediction (t={t_val:.1f}s)')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_aspect('equal')
        plt.colorbar(im1, ax=ax1)
        
        ax2 = plt.subplot(n_snapshots, 3, plot_idx*3 + 2)
        im2 = ax2.contourf(X, Y, u_gt_grid, levels=20, cmap='RdBu_r')
        ax2.set_title(f'Ground Truth (t={t_val:.1f}s)')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_aspect('equal')
        plt.colorbar(im2, ax=ax2)
        
        ax3 = plt.subplot(n_snapshots, 3, plot_idx*3 + 3)
        im3 = ax3.contourf(X, Y, error_grid, levels=20, cmap='hot')
        rel_l2 = results['rel_errors'][interval_idx]
        ax3.set_title(f'Error (Rel L2: {rel_l2:.2e})')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_aspect('equal')
        plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    return fig
