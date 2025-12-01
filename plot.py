import matplotlib.pyplot as plt
import torch
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def plot_solution(model, a_test=0.5, b_test=2.0, source_type='zero', 
                  source_amplitude=1.0, source_center=0.5, source_t=None, T_max=1.0, gt_data=None):
    """Visualize the trained solution"""
    
    # Generate test case
    u0_sensors = model.generate_ic_sine_series(a_test)
    v0_sensors = model.generate_ic_sine_series(b_test)
    
    if model.time_varying_source:
        # Generate spatiotemporal source on sensor grid
        src_grid = model.generate_source(source_type, source_amplitude, source_center, 
                                         x=model.sensor_x_src, time_varying=True, 
                                         t=model.sensor_t_src, center_t=source_t)
        src_sensors = src_grid.flatten()
    else:
        src_sensors = model.generate_source(source_type, source_amplitude, source_center)
    
    # Create spatiotemporal grid
    nx, nt = 100, 50
    x_plot = torch.linspace(model.domain[0], model.domain[1], nx)
    t_plot = torch.linspace(0, T_max, nt)
    
    X, T = torch.meshgrid(x_plot, t_plot, indexing='ij')
    xt_grid = torch.stack([X.flatten(), T.flatten()], dim=1)
    
    # Predict solution
    with torch.no_grad():
        u_pred = model.forward(u0_sensors, v0_sensors, src_sensors, xt_grid)
        u_pred = u_pred.reshape(nx, nt).numpy()
    
    # Source and ICs for plotting
    if model.time_varying_source:
        # For time-varying, plot at specific time source_t
        src_plot = model.generate_source(source_type, source_amplitude, source_center, 
                                        x=x_plot, time_varying=True, 
                                        t=torch.tensor([source_t]), center_t=source_t).squeeze().numpy()
    else:
        src_plot = model.generate_source(source_type, source_amplitude, source_center, x=x_plot).numpy()
    
    u0_plot = (model.generate_ic_sine_series(a_test, x_plot)).numpy()
    v0_plot = (model.generate_ic_sine_series(b_test, x_plot)).numpy()

    # Reshape ground truth data to match prediction grid
    # gt_data format: [x, t, f, u, v] - reshape to (nt, nx) for u and v
    x_gt = gt_data[:, 0]
    t_gt = gt_data[:, 1]
    f_gt = gt_data[:, 2]
    u_gt = gt_data[:, 3]
    v_gt = gt_data[:, 4]
    
    # Determine grid size from gt_data
    nt_gt = len(np.unique(t_gt))
    nx_gt = len(np.unique(x_gt))
    u_gt_grid = u_gt.reshape(nt_gt, nx_gt)  # Shape: (nt, nx)
    
    # Interpolate gt to match prediction grid if needed
    if (nt_gt != nt) or (nx_gt != nx):
        from scipy.interpolate import RegularGridInterpolator
        t_gt_unique = np.unique(t_gt)
        x_gt_unique = np.unique(x_gt)
        interp = RegularGridInterpolator((t_gt_unique, x_gt_unique), u_gt_grid)
        points = np.column_stack([T.numpy().ravel(), X.numpy().ravel()])
        u_gt_interp = interp(points).reshape(nx, nt)
    else:
        u_gt_interp = u_gt_grid.T  # Transpose to (nx, nt) to match u_pred
    
    # Compute absolute error
    abs_error = np.abs(u_pred - u_gt_interp)

    fig = plt.figure(figsize=(18, 10))
    

    # Plot 1: MATLAB Ground Truth
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.contourf(T.numpy(), X.numpy(), u_gt_interp, levels=20, cmap='RdBu_r')
    ax1.set_xlabel('t', fontsize=12)
    ax1.set_ylabel('x', fontsize=12)
    ax1.set_title('MATLAB: u(x,t)', fontsize=12)
    plt.colorbar(im1, ax=ax1)
    
    # Plot 2: PINN Spatiotemporal heatmap
    ax2 = plt.subplot(2, 3, 2)
    im = ax2.contourf(T.numpy(), X.numpy(), u_pred, levels=20, cmap='RdBu_r')
    ax2.set_xlabel('t', fontsize=12)
    ax2.set_ylabel('x', fontsize=12)
    ax2.set_title('PINN: u(x,t)', fontsize=12)
    plt.colorbar(im, ax=ax2)

    # Plot 3: Absolute Error
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.contourf(T.numpy(), X.numpy(), abs_error, levels=20, cmap='hot')
    ax3.set_xlabel('t', fontsize=12)
    ax3.set_ylabel('x', fontsize=12)
    max_err = np.max(abs_error)
    mean_err = np.mean(abs_error)
    ax3.set_title(f'Absolute Error\nmax={max_err:.2e}, mean={mean_err:.2e}', fontsize=12)
    plt.colorbar(im3, ax=ax3)
    
    # Plot 4: ICs and Source
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(x_plot.numpy(), u0_plot, 'b-', linewidth=2, label='IC: u(x,0)')
    ax4.plot(x_plot.numpy(), v0_plot, 'g--', linewidth=2, label='IC: u_t(x,0)')
    source_label = f'Source f(x,t={source_t:.2f})' if model.time_varying_source else 'Source f(x)'
    ax4.plot(x_plot.numpy(), src_plot, 'r-.', linewidth=2, label=source_label)
    ax4.set_xlabel('x', fontsize=12)
    ax4.set_ylabel('Value', fontsize=12)
    ax4.set_title('Initial Conditions & Forcing', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Solution evolution (snapshots)
    ax5 = plt.subplot(2, 3, 5)
    time_indices = [0, nt//4, nt//2, 3*nt//4, -1]
    for idx in time_indices:
        t_val = t_plot[idx].item()
        # PINN prediction
        line_pred, = ax5.plot(x_plot.numpy(), u_pred[:, idx], label=f'PINN t={t_val:.3f}')
        # MATLAB ground truth (dotted, same color as PINN)
        if u_gt_interp is not None:
            ax5.plot(x_plot.numpy(), u_gt_interp[:, idx], linestyle=':', color=line_pred.get_color(), label=f'MATLAB t={t_val:.3f}')
    ax5.set_xlabel('x', fontsize=12)
    ax5.set_ylabel('u(x,t)', fontsize=12)
    ax5.set_title(f'PINN Solution (a={a_test}, b={b_test})', fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Error statistics
    ax6 = plt.subplot(2, 3, 6)
    # Plot error over time (spatial average)
    error_vs_time = np.mean(abs_error, axis=0)
    ax6.plot(t_plot.numpy(), error_vs_time, 'r-', linewidth=2)
    ax6.set_xlabel('t', fontsize=12)
    ax6.set_ylabel('Mean Absolute Error', fontsize=12)
    ax6.set_title('Error Evolution', fontsize=12)
    ax6.grid(True, alpha=0.3)
    
    # Print error statistics
    print(f"\n=== Error Statistics vs MATLAB ===")
    print(f"Max absolute error: {max_err:.6e}")
    print(f"Mean absolute error: {mean_err:.6e}")
    print(f"RMS error: {np.sqrt(np.mean(abs_error**2)):.6e}")
    print(f"Relative L2 error: {np.linalg.norm(abs_error)/np.linalg.norm(u_gt_interp):.6e}")
    
    plt.tight_layout()
    return fig


def plot_training_history(history):
    """Plot training loss history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].semilogy(history['total'], 'k-', linewidth=1.5)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].semilogy(history['pde'], 'b-', linewidth=1.5)
    axes[0, 1].set_title('PDE Residual Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].semilogy(history['ic_u'], 'r-', linewidth=1.5)
    axes[1, 0].set_title('Displacement IC Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].semilogy(history['ic_v'], 'g-', linewidth=1.5)
    axes[1, 1].set_title('Velocity IC Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig