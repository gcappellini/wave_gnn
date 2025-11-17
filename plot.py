import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_solution(model, a_test=1.5, b_test=0.0, source_type='zero', 
                  source_amplitude=0.0, source_center=0.5, T_max=1.0, gt_data=None):
    """Visualize the trained solution"""
    
    # Generate test case
    u0_sensors = model.generate_ic_displacement(a_test)
    v0_sensors = model.generate_ic_velocity(b_test)
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
    src_plot = model.source_function(x_plot, source_type, source_amplitude, source_center).numpy()
    u0_plot = (a_test * torch.sin(np.pi * x_plot)).numpy()
    v0_plot = (b_test * torch.sin(np.pi * x_plot)).numpy()
    
    # Determine figure layout based on whether gt_data is provided
    if gt_data is not None:
        fig = plt.figure(figsize=(18, 10))
        nrows = 2
        
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
        
    else:
        fig = plt.figure(figsize=(18, 5))
        nrows = 1
    
    # Plot 1: Solution evolution (snapshots)
    ax1 = plt.subplot(nrows, 3, 1)
    time_indices = [0, nt//4, nt//2, 3*nt//4, -1]
    for idx in time_indices:
        t_val = t_plot[idx].item()
        ax1.plot(x_plot.numpy(), u_pred[:, idx], label=f't={t_val:.3f}')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('u(x,t)', fontsize=12)
    ax1.set_title(f'PINN Solution (a={a_test}, b={b_test})', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: PINN Spatiotemporal heatmap
    ax2 = plt.subplot(nrows, 3, 2)
    im = ax2.contourf(T.numpy(), X.numpy(), u_pred, levels=20, cmap='RdBu_r')
    ax2.set_xlabel('t', fontsize=12)
    ax2.set_ylabel('x', fontsize=12)
    ax2.set_title('PINN: u(x,t)', fontsize=12)
    plt.colorbar(im, ax=ax2)
    
    # Plot 3: ICs and Source
    ax3 = plt.subplot(nrows, 3, 3)
    ax3.plot(x_plot.numpy(), u0_plot, 'b-', linewidth=2, label='IC: u(x,0)')
    ax3.plot(x_plot.numpy(), v0_plot, 'g--', linewidth=2, label='IC: u_t(x,0)')
    ax3.plot(x_plot.numpy(), src_plot, 'r-.', linewidth=2, label='Source f(x)')
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('Value', fontsize=12)
    ax3.set_title('Initial Conditions & Forcing', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Additional plots if ground truth is provided
    if gt_data is not None:
        # Plot 4: MATLAB Ground Truth
        ax4 = plt.subplot(2, 3, 4)
        im4 = ax4.contourf(T.numpy(), X.numpy(), u_gt_interp, levels=20, cmap='RdBu_r')
        ax4.set_xlabel('t', fontsize=12)
        ax4.set_ylabel('x', fontsize=12)
        ax4.set_title('MATLAB: u(x,t)', fontsize=12)
        plt.colorbar(im4, ax=ax4)
        
        # Plot 5: Absolute Error
        ax5 = plt.subplot(2, 3, 5)
        im5 = ax5.contourf(T.numpy(), X.numpy(), abs_error, levels=20, cmap='hot')
        ax5.set_xlabel('t', fontsize=12)
        ax5.set_ylabel('x', fontsize=12)
        max_err = np.max(abs_error)
        mean_err = np.mean(abs_error)
        ax5.set_title(f'Absolute Error\nmax={max_err:.2e}, mean={mean_err:.2e}', fontsize=12)
        plt.colorbar(im5, ax=ax5)
        
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