import matplotlib.pyplot as plt
import torch
import numpy as np
import logging

log = logging.getLogger(__name__)

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


def plot_training_with_pretraining(history, pretrain_history, val_interval=100):
    """
    Plot training history with pretraining phase separately highlighted.
    Shows pretraining IC_u and IC_v losses + test metric alongside main training losses.
    
    Handles two cases:
    1. Both Phase 1 and Phase 2 executed: Shows separate plots for each phase
    2. Only Phase 1 executed: Shows only pretraining losses in left panel, empty right panel
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Check if Phase 2 was actually run (has 'total' and 'pde' keys)
    has_phase2 = history is not None and 'total' in history and 'pde' in history
    
    # ========== Left panel: Pretraining phase ==========
    if pretrain_history is not None and len(pretrain_history) > 0:
        pretrain_epochs = list(range(1, len(pretrain_history.get('loss_ic_u', [])) + 1))
        if 'loss_ic_u' in pretrain_history and len(pretrain_history['loss_ic_u']) > 0:
            ax1.semilogy(pretrain_epochs, pretrain_history['loss_ic_u'], color='#FFDAC1', label='IC_u Loss', linewidth=2.5, marker='.')
        if 'loss_ic_v' in pretrain_history and len(pretrain_history['loss_ic_v']) > 0:
            ax1.semilogy(pretrain_epochs, pretrain_history['loss_ic_v'], color='#FFB7B2', label='IC_v Loss', linewidth=2.5, marker='.')
        
        # Plot test metric if available
        if 'test_metric' in pretrain_history and len(pretrain_history['test_metric']) > 0:
            test_epochs_pre = pretrain_history.get('test_epochs', 
                                                    [(i+1) * val_interval for i in range(len(pretrain_history['test_metric']))])
            ax1.semilogy(test_epochs_pre, pretrain_history['test_metric'], color='#E6B89C', 
                         label='Test Metric', linewidth=2.5, marker='o', markersize=5)
        
        ax1.set_title('Phase 1: Pretraining IC Reconstruction', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (log scale)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
    else:
        ax1.text(0.5, 0.5, 'No pretraining data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Phase 1: Pretraining (Not Run)', fontsize=12, fontweight='bold')
    
    # ========== Right panel: Main training phase ==========
    if has_phase2:
        # Phase 2 was run: plot main training losses
        if 'total' in history and len(history['total']) > 0:
            ax2.semilogy(history['total'], color='#A3C1DA', label='Total Loss', linewidth=2)
        if 'pde' in history and len(history['pde']) > 0:
            ax2.semilogy(history['pde'], color='#B5EAD7', label='PDE Residual Loss', linewidth=2)
        if 'ic_u' in history and len(history['ic_u']) > 0:
            ax2.semilogy(history['ic_u'], color='#FFDAC1', label='Displacement IC Loss', linewidth=2)
        if 'ic_v' in history and len(history['ic_v']) > 0:
            ax2.semilogy(history['ic_v'], color='#FFB7B2', label='Velocity IC Loss', linewidth=2)
        
        # Plot test metric if available with proper epoch alignment
        if 'test_metric' in history and len(history['test_metric']) > 0:
            test_epochs = history['test_epochs'] if 'test_epochs' in history else [(i+1) * val_interval for i in range(len(history['test_metric']))]
            ax2.semilogy(test_epochs, history['test_metric'], color='#E6B89C', label='Test Metric', linewidth=2, marker='o', markersize=4)
        
        ax2.set_title('Phase 2: Main Training (Adaptive Weights)', fontsize=12, fontweight='bold')
    else:
        # Phase 2 was not run: show only pretraining message
        ax2.text(0.5, 0.5, 'Phase 2 not executed', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Phase 2: Main Training (Not Run)', fontsize=12, fontweight='bold')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (log scale)')
    ax2.grid(True, alpha=0.3)
    if has_phase2:
        ax2.legend(fontsize=11)
    
    plt.tight_layout()
    return fig

def plot_ic_reconstruction(model, test_case, n_grid=100, save_path="ic_reconstruction.png", gt_data=None):
    """
    Visualizes IC reconstruction against MATLAB ground truth at t=0.
    If gt_data is available, extracts u0 and v0 from MATLAB solution at t=0.
    Falls back to generated test case if gt_data is not available.
    Compact layout: 3 columns for displacement and velocity rows.
    """
    model.eval() # Set model to evaluation mode
    
    # Determine device from model parameters
    device = next(model.parameters()).device
    
    # 1. Generate Evaluation Grid (100x100 points)
    domain = model.domain # Assuming domain is [0, 1]
    x_1d = torch.linspace(domain[0], domain[1], n_grid, device=device)
    y_1d = torch.linspace(domain[0], domain[1], n_grid, device=device)
    X, Y = torch.meshgrid(x_1d, y_1d, indexing='ij')
    
    # Collocation grid at t=0
    xyt_eval = torch.stack([X.flatten(), Y.flatten(), torch.zeros_like(X.flatten())], dim=1)
    
    # 2. Generate Input Sensors and Get Ground Truth or Test Case
    with torch.no_grad():
        if gt_data is not None:
            # Extract IC from MATLAB ground truth at t=0
            x_gt = gt_data[:, 0]
            y_gt = gt_data[:, 1]
            t_gt = gt_data[:, 2]
            u_gt = gt_data[:, 4]
            v_gt = gt_data[:, 5]
            
            # Get unique time values
            t_unique = np.unique(t_gt)
            
            # Find closest time to t=0
            t_idx = np.argmin(np.abs(t_unique - 0.0))
            t_selected = t_unique[t_idx]
            
            # Extract data at selected time
            mask = np.abs(t_gt - t_selected) < 1e-6
            x_selected = x_gt[mask]
            y_selected = y_gt[mask]
            u_selected = u_gt[mask]
            v_selected = v_gt[mask]
            
            if len(x_selected) == 0:
                log.warning("No ground truth data at t=0, using test case instead")
                # Fallback to test case
                a_coeffs = test_case['a_coeffs']
                b_coeffs = test_case['b_coeffs']
                u0_sensors = model.generate_ic_sine_series(a_coeffs)
                v0_sensors = model.generate_ic_sine_series(b_coeffs)
                u0_true_field = model.generate_ic_sine_series(a_coeffs, X.flatten(), Y.flatten())
                v0_true_field = model.generate_ic_sine_series(b_coeffs, X.flatten(), Y.flatten())
                gt_available = False
            else:
                # Get spatial grid info from MATLAB data at t=0
                x_unique = np.unique(x_selected)
                y_unique = np.unique(y_selected)
                
                # Create meshgrid for MATLAB data
                X_gt, Y_gt = np.meshgrid(x_unique, y_unique, indexing='ij')
                
                # Interpolate MATLAB GT to evaluation grid using griddata
                from scipy.interpolate import griddata
                points = np.column_stack([x_selected, y_selected])
                points_eval = np.column_stack([X.flatten().cpu().numpy(), Y.flatten().cpu().numpy()])
                
                u0_true_field = torch.tensor(
                    griddata(points, u_selected, points_eval, method='linear', fill_value=0.0),
                    device=device, dtype=torch.float32
                )
                v0_true_field = torch.tensor(
                    griddata(points, v_selected, points_eval, method='linear', fill_value=0.0),
                    device=device, dtype=torch.float32
                )
                
                # For sensors, extract values at sensor locations from interpolated field
                u0_sensors = u0_true_field[:model.n_sensors_ic**2].clone()
                v0_sensors = v0_true_field[:model.n_sensors_ic**2].clone()
                gt_available = True
        else:
            # Use test case
            a_coeffs = test_case['a_coeffs']
            b_coeffs = test_case['b_coeffs']
            u0_sensors = model.generate_ic_sine_series(a_coeffs)
            v0_sensors = model.generate_ic_sine_series(b_coeffs)
            u0_true_field = model.generate_ic_sine_series(a_coeffs, X.flatten(), Y.flatten())
            v0_true_field = model.generate_ic_sine_series(b_coeffs, X.flatten(), Y.flatten())
            gt_available = False
        
        # Model Output (Reconstruction)
        u0_pred_field, v0_pred_field = model.predict_ic(u0_sensors, v0_sensors, xyt_eval)

    # 3. Reshape fields for plotting
    u0_true = u0_true_field.reshape(n_grid, n_grid).cpu().numpy()
    u0_pred = u0_pred_field.reshape(n_grid, n_grid).cpu().numpy()
    v0_true = v0_true_field.reshape(n_grid, n_grid).cpu().numpy()
    v0_pred = v0_pred_field.reshape(n_grid, n_grid).cpu().numpy()
    
    # Compute signed errors
    u0_error = u0_pred - u0_true
    v0_error = v0_pred - v0_true
    
    # Color scaling (95th percentile for better contrast)
    vmax_u = max(np.percentile(np.abs(u0_true), 95), np.percentile(np.abs(u0_pred), 95))
    vmax_v = max(np.percentile(np.abs(v0_true), 95), np.percentile(np.abs(v0_pred), 95))
    err_vmax_u = np.percentile(np.abs(u0_error), 95)
    err_vmax_v = np.percentile(np.abs(v0_error), 95)

    # 4. Compact 2x3 layout: displacement row, velocity row
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    
    source_label = "MATLAB GT" if gt_available else "Generated"
    
    # ===== DISPLACEMENT ROW =====
    # PINN Displacement
    c00 = axes[0, 0].contourf(X.cpu().numpy(), Y.cpu().numpy(), u0_pred, levels=50, cmap='RdBu_r', vmax=vmax_u, vmin=-vmax_u)
    axes[0, 0].set_ylabel('y', fontsize=11)
    axes[0, 0].set_title(r'$u_0$ PINN', fontsize=11, fontweight='bold')
    fig.colorbar(c00, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # Ground Truth Displacement
    c01 = axes[0, 1].contourf(X.cpu().numpy(), Y.cpu().numpy(), u0_true, levels=50, cmap='RdBu_r', vmax=vmax_u, vmin=-vmax_u)
    axes[0, 1].set_title(f'$u_0$ {source_label}', fontsize=11, fontweight='bold')
    fig.colorbar(c01, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Displacement Error (signed)
    c02 = axes[0, 2].contourf(X.cpu().numpy(), Y.cpu().numpy(), u0_error, levels=50, cmap='RdBu_r', vmax=err_vmax_u, vmin=-err_vmax_u)
    axes[0, 2].set_title(f'Error $u_0$ (MAE={np.mean(np.abs(u0_error)):.2e})', fontsize=11, fontweight='bold')
    fig.colorbar(c02, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # ===== VELOCITY ROW =====
    # PINN Velocity
    c10 = axes[1, 0].contourf(X.cpu().numpy(), Y.cpu().numpy(), v0_pred, levels=50, cmap='viridis', vmax=vmax_v, vmin=-vmax_v)
    axes[1, 0].set_xlabel('x', fontsize=11)
    axes[1, 0].set_ylabel('y', fontsize=11)
    axes[1, 0].set_title(r'$v_0$ PINN', fontsize=11, fontweight='bold')
    fig.colorbar(c10, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Ground Truth Velocity
    c11 = axes[1, 1].contourf(X.cpu().numpy(), Y.cpu().numpy(), v0_true, levels=50, cmap='viridis', vmax=vmax_v, vmin=-vmax_v)
    axes[1, 1].set_xlabel('x', fontsize=11)
    axes[1, 1].set_title(f'$v_0$ {source_label}', fontsize=11, fontweight='bold')
    fig.colorbar(c11, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Velocity Error (signed)
    c12 = axes[1, 2].contourf(X.cpu().numpy(), Y.cpu().numpy(), v0_error, levels=50, cmap='RdBu_r', vmax=err_vmax_v, vmin=-err_vmax_v)
    axes[1, 2].set_xlabel('x', fontsize=11)
    axes[1, 2].set_title(f'Error $v_0$ (MAE={np.mean(np.abs(v0_error)):.2e})', fontsize=11, fontweight='bold')
    fig.colorbar(c12, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.08, wspace=0.4, hspace=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"IC Reconstruction plot saved to {save_path}")
    log.info(f"Ground truth source: {'MATLAB' if gt_available else 'Generated test case'}")
    log.info(f"Displacement error - MAE: {np.mean(np.abs(u0_error)):.6e}, Max: {np.max(np.abs(u0_error)):.6e}")
    log.info(f"Velocity error     - MAE: {np.mean(np.abs(v0_error)):.6e}, Max: {np.max(np.abs(v0_error)):.6e}")
    model.train() # Restore training mode