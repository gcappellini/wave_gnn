import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_solution_2d(model, a_test=0.5, b_test=0.0, source_type='zero', 
                     source_amplitude=7.5, center_x=0.5, center_y=0.5, T_max=2.0, gt_data=None):
    """Visualize 2D wave solution at different time snapshots"""
    
    # Generate test case
    u0_sensors = model.generate_ic_sine_series(a_test)
    v0_sensors = model.generate_ic_sine_series(b_test)
    src_sensors = model.generate_source(source_type, source_amplitude, center_x, center_y)
    
    # Create spatiotemporal grid
    nx, ny, nt = 50, 50, 5  # 5 time snapshots
    x_plot = torch.linspace(model.domain[0], model.domain[1], nx)
    y_plot = torch.linspace(model.domain[0], model.domain[1], ny)
    t_plot = torch.linspace(0, T_max, nt)
    
    # Predict solution at each time
    u_pred_snapshots = []
    with torch.no_grad():
        for t_val in t_plot:
            X, Y = torch.meshgrid(x_plot, y_plot, indexing='ij')
            T = torch.full_like(X, t_val)
            xyt_grid = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)
            
            u_pred = model.forward(u0_sensors, v0_sensors, src_sensors, xyt_grid)
            u_pred_grid = u_pred.reshape(nx, ny).numpy()
            u_pred_snapshots.append(u_pred_grid)
    
    # Plot snapshots
    fig = plt.figure(figsize=(20, 4))
    
    for i, (u_snap, t_val) in enumerate(zip(u_pred_snapshots, t_plot)):
        ax = plt.subplot(1, nt, i+1)
        im = ax.contourf(x_plot.numpy(), y_plot.numpy(), u_snap, levels=20, cmap='RdBu_r')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f't = {t_val:.2f}s')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    return fig


def plot_solution_2d_comparison(model, a_test=0.5, b_test=0.0, source_type='zero',
                                 source_amplitude=7.5, center_x=0.5, center_y=0.5, 
                                 T_max=2.0, gt_data=None):
    """Compare PINN vs ground truth at multiple time snapshots"""
    
    if gt_data is None:
        return plot_solution_2d(model, a_test, b_test, source_type, 
                               source_amplitude, center_x, center_y, T_max)
    
    # Generate test case
    u0_sensors = model.generate_ic_sine_series(a_test)
    v0_sensors = model.generate_ic_sine_series(b_test)
    src_sensors = model.generate_source(source_type, source_amplitude, center_x, center_y)
    
    # Extract ground truth data
    x_gt = gt_data[:, 0]
    y_gt = gt_data[:, 1]
    t_gt = gt_data[:, 2]
    u_gt = gt_data[:, 4]
    
    # Get unique coordinates
    t_unique = np.unique(t_gt)
    x_unique_all = np.unique(x_gt)
    y_unique_all = np.unique(y_gt)
    
    # Define time snapshots: 0%, 25%, 50%, 75%, 100%
    time_fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
    time_values = [T_max * frac for frac in time_fractions]
    n_times = len(time_values)
    
    # Create figure with 5 rows x 3 columns
    fig = plt.figure(figsize=(18, 25))
    
    from scipy.interpolate import griddata
    
    # First pass: compute global min/max across all time snapshots
    global_vmin = float('inf')
    global_vmax = float('-inf')
    all_grids = []  # Store grids for second pass
    
    for row_idx, t_target in enumerate(time_values):
        # Find closest time in ground truth
        t_idx = np.argmin(np.abs(t_unique - t_target))
        t_selected = t_unique[t_idx]
        
        # Extract data at selected time
        mask = np.abs(t_gt - t_selected) < 1e-6
        x_selected = x_gt[mask]
        y_selected = y_gt[mask]
        u_selected = u_gt[mask]
        
        # Get spatial grid info
        x_unique = np.unique(x_selected)
        y_unique = np.unique(y_selected)
        nx_gt, ny_gt = len(x_unique), len(y_unique)
        
        # Interpolate ground truth to regular grid
        X_gt, Y_gt = np.meshgrid(x_unique, y_unique, indexing='ij')
        points = np.column_stack([x_selected, y_selected])
        u_gt_grid = griddata(points, u_selected, (X_gt, Y_gt), method='linear')
        
        # PINN prediction at same grid
        x_plot = torch.tensor(x_unique, dtype=torch.float32)
        y_plot = torch.tensor(y_unique, dtype=torch.float32)
        X_pred, Y_pred = torch.meshgrid(x_plot, y_plot, indexing='ij')
        T_pred = torch.full_like(X_pred, t_selected)
        xyt_grid = torch.stack([X_pred.flatten(), Y_pred.flatten(), T_pred.flatten()], dim=1)
        
        with torch.no_grad():
            u_pred = model.forward(u0_sensors, v0_sensors, src_sensors, xyt_grid)
            u_pred_grid = u_pred.reshape(nx_gt, ny_gt).numpy()
        
        # Update global min/max
        global_vmin = min(global_vmin, np.nanmin(u_gt_grid), np.nanmin(u_pred_grid))
        global_vmax = max(global_vmax, np.nanmax(u_gt_grid), np.nanmax(u_pred_grid))
        
        # Store grids for plotting
        all_grids.append({
            't_selected': t_selected,
            'X_gt': X_gt,
            'Y_gt': Y_gt,
            'u_gt_grid': u_gt_grid,
            'X_pred': X_pred,
            'Y_pred': Y_pred,
            'u_pred_grid': u_pred_grid
        })

    metrics = {}
    
    # Second pass: plot with global color limits
    for row_idx, grid_data in enumerate(all_grids):
        t_selected = grid_data['t_selected']
        X_gt = grid_data['X_gt']
        Y_gt = grid_data['Y_gt']
        u_gt_grid = grid_data['u_gt_grid']
        X_pred = grid_data['X_pred']
        Y_pred = grid_data['Y_pred']
        u_pred_grid = grid_data['u_pred_grid']
        
        # Compute error
        error = np.abs(u_pred_grid - u_gt_grid)
        rel_l2 = np.linalg.norm(error) / (np.linalg.norm(u_gt_grid) + 1e-10)
        
        # Column 1: Ground Truth
        ax1 = plt.subplot(n_times, 3, row_idx*3 + 1)
        im1 = ax1.contourf(X_gt, Y_gt, u_gt_grid, levels=20, cmap='RdBu_r', vmin=global_vmin, vmax=global_vmax)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(f'Ground Truth (t={t_selected:.2f}s)')
        ax1.set_aspect('equal')
        plt.colorbar(im1, ax=ax1)
        
        # Column 2: PINN Prediction
        ax2 = plt.subplot(n_times, 3, row_idx*3 + 2)
        im2 = ax2.contourf(X_pred.numpy(), Y_pred.numpy(), u_pred_grid, levels=20, cmap='RdBu_r', vmin=global_vmin, vmax=global_vmax)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title(f'PINN Prediction (t={t_selected:.2f}s)')
        ax2.set_aspect('equal')
        plt.colorbar(im2, ax=ax2)
        
        # Column 3: Error
        ax3 = plt.subplot(n_times, 3, row_idx*3 + 3)
        im3 = ax3.contourf(X_pred.numpy(), Y_pred.numpy(), error, levels=20, cmap='hot')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_title(f'Error (Rel L2: {rel_l2:.2e})')
        ax3.set_aspect('equal')
        plt.colorbar(im3, ax=ax3)
        
        # Print error statistics for this time
        print(f"\n=== Error Statistics at t={t_selected:.2f}s ({time_fractions[row_idx]*100:.0f}%) ===")
        print(f"Max error: {np.nanmax(error):.6e}")
        print(f"Mean error: {np.nanmean(error):.6e}")
        print(f"Relative L2: {rel_l2:.6e}")
        
        metrics[t_selected] = {
            'max_error': np.nanmax(error),
            'mean_error': np.nanmean(error),
            'rel_l2': rel_l2,
            'error_grid': error.flatten()  # Store flattened error for global L2 calculation
        }

    
    plt.tight_layout()
    
    # Transform metrics dictionary
    time_errors = {t: m for t, m in metrics.items() if isinstance(m, dict)}
    
    metrics['global_mean_error'] = np.nanmean([m['mean_error'] for m in time_errors.values()])
    metrics['global_max_error'] = np.nanmax([m['max_error'] for m in time_errors.values()])
    
    # Compute global L2 error across all time snapshots
    all_errors = np.concatenate([m['error_grid'] for m in time_errors.values() if 'error_grid' in m])
    metrics['global_l2_error'] = np.linalg.norm(all_errors)
    
    # Find errors at t=0 and t=1
    t_values_list = sorted(time_errors.keys())
    if len(t_values_list) > 0:
        t_0_error = time_errors[t_values_list[0]]
        metrics['t0_mean_error'] = t_0_error['mean_error']
        metrics['t0_max_error'] = t_0_error['max_error']
    
    if len(t_values_list) > 0:
        t_1_error = time_errors[t_values_list[-1]]
        metrics['t1_mean_error'] = t_1_error['mean_error']
        metrics['t1_max_error'] = t_1_error['max_error']
    
    return fig, metrics


def plot_training_history(history, val_interval=100):
    """Plot all training loss histories on a single figure with pastel colors"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.semilogy(history['total'], color='#A3C1DA', label='Total Loss', linewidth=2)
    ax.semilogy(history['pde'], color='#B5EAD7', label='PDE Residual Loss', linewidth=2)
    ax.semilogy(history['ic_u'], color='#FFDAC1', label='Displacement IC Loss', linewidth=2)
    ax.semilogy(history['ic_v'], color='#FFB7B2', label='Velocity IC Loss', linewidth=2)
    
    # Plot test metric if available with proper epoch alignment
    if 'test_metric' in history and len(history['test_metric']) > 0:
        test_epochs = history['test_epochs'] if 'test_epochs' in history else [(i+1) * val_interval for i in range(len(history['test_metric']))]
        ax.semilogy(test_epochs, history['test_metric'], color='#E6B89C', label='Test Metric', linewidth=2, marker='o', markersize=4)
    
    ax.set_title('Training Loss History')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig
