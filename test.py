import numpy as np
import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')



def rollout_test(model, gt_data, T_total=10.0, dt_interval=1.0, 
                 nx=100, nt_per_interval=50):
    """
    Rollout test with changing source center every dt_interval seconds
    Uses ground truth data to extract sensor values at each interval
    
    Args:
        model: trained PINN-DeepONet model
        gt_data: ground truth data [x, t, f, u, v] (REQUIRED)
        T_total: total simulation time (default: 10s)
        dt_interval: time interval for source center change (default: 1s)
        nx: number of spatial points
        nt_per_interval: number of time points per interval
    
    Returns:
        u_rollout: complete solution (nx, nt_total)
        t_rollout: time vector
        x_rollout: spatial vector
        fig: matplotlib figure
    """
    
    if gt_data is None:
        raise ValueError("Ground truth data is required for rollout test!")
    
    # Extract ground truth - format: [x, t, f, u, v]
    x_gt = gt_data[:, 0]
    t_gt = gt_data[:, 1]
    f_gt = gt_data[:, 2]
    u_gt = gt_data[:, 3]
    v_gt = gt_data[:, 4]
    
    # Get unique coordinates
    x_unique = np.unique(x_gt)
    t_unique = np.unique(t_gt)
    nt_gt = len(t_unique)
    nx_gt = len(x_unique)
    
    # Reshape to grids
    u_gt_grid = u_gt.reshape(nt_gt, nx_gt).T  # (nx_gt, nt_gt)
    v_gt_grid = v_gt.reshape(nt_gt, nx_gt).T
    f_gt_grid = f_gt.reshape(nt_gt, nx_gt).T
    
    # Setup
    n_intervals = int(T_total / dt_interval)
    x_plot = torch.linspace(model.domain[0], model.domain[1], nx)
    t_interval = torch.linspace(0, dt_interval, nt_per_interval)
    
    # Storage for complete rollout
    u_rollout = []
    
    # Create fixed xt_grid for each interval
    X, T = torch.meshgrid(x_plot, t_interval, indexing='ij')
    xt_grid = torch.stack([X.flatten(), T.flatten()], dim=1)
    
    model.eval()
    with torch.no_grad():
        for interval_idx in range(n_intervals):
            # Find the time index in ground truth for start of this interval
            t_start = interval_idx * dt_interval
            t_idx = np.argmin(np.abs(t_unique - t_start))
            
            # Extract u, v, f from ground truth at this time step
            u_current_gt = u_gt_grid[:, t_idx]  # (nx_gt,)
            v_current_gt = v_gt_grid[:, t_idx]
            f_current_gt = f_gt_grid[:, t_idx]
            
            # Interpolate to sensor locations if needed
            u0_sensors = torch.tensor(np.interp(model.sensor_x_ic.numpy(), 
                                                 x_unique, u_current_gt), dtype=torch.float32)
            v0_sensors = torch.tensor(np.interp(model.sensor_x_ic.numpy(), 
                                                 x_unique, v_current_gt), dtype=torch.float32)
            src_sensors = torch.tensor(np.interp(model.sensor_x_src.numpy(), 
                                                  x_unique, f_current_gt), dtype=torch.float32)
            
            # Predict for this interval
            u_pred = model.forward(u0_sensors, v0_sensors, src_sensors, xt_grid)
            u_pred_grid = u_pred.reshape(nx, nt_per_interval)
            
            # Store results
            u_rollout.append(u_pred_grid.numpy())
    
    # Concatenate all intervals
    u_rollout = np.concatenate(u_rollout, axis=1)
    
    # Create full time and space arrays
    t_rollout = np.linspace(0, T_total, n_intervals * nt_per_interval)
    x_rollout = x_plot.numpy()
    
    # Interpolate ground truth to match rollout grid
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((x_unique, t_unique), u_gt_grid)
    X_new, T_new = np.meshgrid(x_rollout, t_rollout, indexing='ij')
    points = np.column_stack([X_new.ravel(), T_new.ravel()])
    u_gt_interp = interp(points).reshape(u_rollout.shape)
    
    # Plot results
    fig = plt.figure(figsize=(18, 10))
    
    # 3 subplots: PINN, Ground Truth, Error
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.contourf(t_rollout, x_rollout, u_rollout, levels=50, cmap='viridis')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('x')
    ax1.set_title('PINN Rollout Prediction')
    plt.colorbar(im1, ax=ax1)
    
    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.contourf(t_rollout, x_rollout, u_gt_interp, levels=50, cmap='viridis')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('x')
    ax2.set_title('MATLAB Ground Truth')
    plt.colorbar(im2, ax=ax2)
    
    ax3 = plt.subplot(1, 3, 3)
    error = np.abs(u_rollout - u_gt_interp)
    rel_l2_error = np.linalg.norm(error)/np.linalg.norm(u_gt_interp)
    im3 = ax3.contourf(t_rollout, x_rollout, error, levels=50, cmap='hot')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('x')
    ax3.set_title(f'Absolute Error\n(Rel L2: {rel_l2_error:.2e})')
    plt.colorbar(im3, ax=ax3)
    
    print(f"\n=== Rollout Error Statistics ===")
    print(f"Max absolute error: {np.max(error):.6e}")
    print(f"Mean absolute error: {np.mean(error):.6e}")
    print(f"RMS error: {np.sqrt(np.mean(error**2)):.6e}")
    print(f"Relative L2 error: {rel_l2_error:.6e}")
    
    plt.tight_layout()
    return u_rollout, t_rollout, x_rollout, fig