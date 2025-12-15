import matplotlib.pyplot as plt
import torch
import numpy as np
import logging
import os

log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def plot_solution(model, a_test=0.5, b_test=2.0, source_type='zero', 
                  source_amplitude=1.0, source_center=0.5, source_t=None, T_max=1.0, gt_data=None):
    """Visualize the trained solution with displacement and velocity"""
    
    # Generate test case
    u0_sensors = model.generate_ic_sine_series(a_test)
    v0_sensors = model.generate_ic_sine_series(b_test)
    
    # Always generate spatiotemporal source
    src_grid = model.generate_source(source_type, source_amplitude, source_center, 
                                     x=model.sensor_x_src, time_varying=True, 
                                     t=model.sensor_t_src, center_t=source_t)
    src_sensors = src_grid.flatten()
    
    # Create spatiotemporal grid
    nx, nt = 100, 50
    x_plot = torch.linspace(model.domain[0], model.domain[1], nx)
    t_plot = torch.linspace(0, T_max, nt)
    
    X, T = torch.meshgrid(x_plot, t_plot, indexing='ij')
    xt_grid = torch.stack([X.flatten(), T.flatten()], dim=1)
    
    # Predict solution and velocity
    with torch.no_grad():
        u_pred = model.forward(u0_sensors, v0_sensors, src_sensors, xt_grid)
        u_pred = u_pred.reshape(nx, nt).numpy()
        v_pred = model.get_velocity(u0_sensors, v0_sensors, src_sensors, xt_grid)
        v_pred = v_pred.reshape(nx, nt).numpy()
    
    # Source and ICs for plotting
    src_plot = model.generate_source(source_type, source_amplitude, source_center, 
                                    x=x_plot, time_varying=True, 
                                    t=torch.tensor([source_t]), center_t=source_t).squeeze().numpy()
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
    v_gt_grid = v_gt.reshape(nt_gt, nx_gt)  # Shape: (nt, nx)
    
    # Interpolate gt to match prediction grid if needed
    if (nt_gt != nt) or (nx_gt != nx):
        from scipy.interpolate import RegularGridInterpolator
        t_gt_unique = np.unique(t_gt)
        x_gt_unique = np.unique(x_gt)
        interp_u = RegularGridInterpolator((t_gt_unique, x_gt_unique), u_gt_grid)
        interp_v = RegularGridInterpolator((t_gt_unique, x_gt_unique), v_gt_grid)
        points = np.column_stack([T.numpy().ravel(), X.numpy().ravel()])
        u_gt_interp = interp_u(points).reshape(nx, nt)
        v_gt_interp = interp_v(points).reshape(nx, nt)
    else:
        u_gt_interp = u_gt_grid.T  # Transpose to (nx, nt) to match u_pred
        v_gt_interp = v_gt_grid.T  # Transpose to (nx, nt) to match v_pred
    
    # Compute signed errors
    u_error = u_pred - u_gt_interp
    v_error = v_pred - v_gt_interp
    
    # Color scaling
    u_vmax = max(np.abs(np.percentile(u_pred, 95)), np.abs(np.percentile(u_gt_interp, 95)))
    v_vmax = max(np.abs(np.percentile(v_pred, 95)), np.abs(np.percentile(v_gt_interp, 95)))
    err_vmax_u = np.percentile(np.abs(u_error), 95)
    err_vmax_v = np.percentile(np.abs(v_error), 95)

    fig = plt.figure(figsize=(16, 10))
    
    # ===== DISPLACEMENT ROW =====
    # Plot 1: PINN Displacement
    ax1 = plt.subplot(2, 4, 1)
    im1 = ax1.contourf(T.numpy(), X.numpy(), u_pred, levels=20, cmap='RdBu_r', vmin=-u_vmax, vmax=u_vmax)
    ax1.set_ylabel('x', fontsize=11)
    ax1.set_title('PINN: u(x,t)', fontsize=11, fontweight='bold')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Plot 2: Ground Truth Displacement
    ax2 = plt.subplot(2, 4, 2)
    im2 = ax2.contourf(T.numpy(), X.numpy(), u_gt_interp, levels=20, cmap='RdBu_r', vmin=-u_vmax, vmax=u_vmax)
    ax2.set_title('Ground: u(x,t)', fontsize=11, fontweight='bold')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Plot 3: Displacement Error
    ax3 = plt.subplot(2, 4, 3)
    im3 = ax3.contourf(T.numpy(), X.numpy(), u_error, levels=20, cmap='RdBu_r', vmin=-err_vmax_u, vmax=err_vmax_u)
    ax3.set_title(f'Error: u (MAE={np.mean(np.abs(u_error)):.2e})', fontsize=11, fontweight='bold')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # Plot 4: ICs and Source
    ax4 = plt.subplot(2, 4, 4)
    ax4.plot(x_plot.numpy(), u0_plot, 'b-', linewidth=2.5, label='u₀(x,0)')
    ax4.plot(x_plot.numpy(), v0_plot, 'g--', linewidth=2.5, label='v₀(x,0)')
    ax4.plot(x_plot.numpy(), src_plot, 'r-.', linewidth=2.5, label=f'f(x,t={source_t:.2f})')
    ax4.set_ylabel('Value', fontsize=11)
    ax4.set_title('ICs & Forcing', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # ===== VELOCITY ROW =====
    # Plot 5: PINN Velocity
    ax5 = plt.subplot(2, 4, 5)
    im5 = ax5.contourf(T.numpy(), X.numpy(), v_pred, levels=20, cmap='viridis', vmin=-v_vmax, vmax=v_vmax)
    ax5.set_xlabel('t', fontsize=11)
    ax5.set_ylabel('x', fontsize=11)
    ax5.set_title('PINN: v(x,t)', fontsize=11, fontweight='bold')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    
    # Plot 6: Ground Truth Velocity
    ax6 = plt.subplot(2, 4, 6)
    im6 = ax6.contourf(T.numpy(), X.numpy(), v_gt_interp, levels=20, cmap='viridis', vmin=-v_vmax, vmax=v_vmax)
    ax6.set_xlabel('t', fontsize=11)
    ax6.set_title('Ground: v(x,t)', fontsize=11, fontweight='bold')
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    
    # Plot 7: Velocity Error
    ax7 = plt.subplot(2, 4, 7)
    im7 = ax7.contourf(T.numpy(), X.numpy(), v_error, levels=20, cmap='RdBu_r', vmin=-err_vmax_v, vmax=err_vmax_v)
    ax7.set_xlabel('t', fontsize=11)
    ax7.set_title(f'Error: v (MAE={np.mean(np.abs(v_error)):.2e})', fontsize=11, fontweight='bold')
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
    
    # Plot 8: Time evolution
    ax8 = plt.subplot(2, 4, 8)
    time_indices = [0, nt//4, nt//2, 3*nt//4, -1]
    for idx in time_indices:
        t_val = t_plot[idx].item()
        ax8.plot(x_plot.numpy(), u_pred[:, idx], label=f't={t_val:.2f}', linewidth=2)
    ax8.set_xlabel('x', fontsize=11)
    ax8.set_ylabel('u(x,t)', fontsize=11)
    ax8.set_title('Solution Snapshots', fontsize=11, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)
    
    # Print error statistics
    print(f"\n=== Error Statistics vs Ground Truth ===")
    print(f"Displacement (u) - Max: {np.max(np.abs(u_error)):.6e}, Mean: {np.mean(np.abs(u_error)):.6e}, RMS: {np.sqrt(np.mean(u_error**2)):.6e}")
    print(f"Velocity (v)     - Max: {np.max(np.abs(v_error)):.6e}, Mean: {np.mean(np.abs(v_error)):.6e}, RMS: {np.sqrt(np.mean(v_error**2)):.6e}")
    print(f"Relative L2 error (u): {np.linalg.norm(u_error)/np.linalg.norm(u_gt_interp):.6e}")
    print(f"Relative L2 error (v): {np.linalg.norm(v_error)/np.linalg.norm(v_gt_interp):.6e}")
    
    plt.subplots_adjust(left=0.07, right=0.98, top=0.96, bottom=0.08, wspace=0.35, hspace=0.35)
    metrics = {
        'u_max_error': np.max(np.abs(u_error)),
        'u_mean_error': np.mean(np.abs(u_error)),
        'v_max_error': np.max(np.abs(v_error)),
        'v_mean_error': np.mean(np.abs(v_error))}
    return fig, metrics


def plot_training_history(history):
    """Plot all training loss histories on a single figure with pastel colors"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.semilogy(history['total'], color='#A3C1DA', label='Total Loss', linewidth=2)
    ax.semilogy(history['pde'], color='#B5EAD7', label='PDE Residual Loss', linewidth=2)
    ax.semilogy(history['ic_u'], color='#FFDAC1', label='Displacement IC Loss', linewidth=2)
    ax.semilogy(history['ic_v'], color='#FFB7B2', label='Velocity IC Loss', linewidth=2)
    
    ax.set_title('Training Loss History')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_ic_reconstruction(model, test_case, n_grid=100, save_path="ic_reconstruction.png"):
    """
    Visualizes true vs. reconstructed u0 and v0 fields with signed errors.
    Compact layout: 3 columns for displacement and velocity rows.
    """
    model.eval() # Set model to evaluation mode
    
    # 1. Generate Evaluation Grid (100x100 points)
    domain = model.domain # Assuming domain is [0, 1]
    x_1d = torch.linspace(domain[0], domain[1], n_grid)
    y_1d = torch.linspace(domain[0], domain[1], n_grid)
    X, Y = torch.meshgrid(x_1d, y_1d, indexing='ij')
    
    # Collocation grid at t=0
    xyt_eval = torch.stack([X.flatten(), Y.flatten(), torch.zeros_like(X.flatten())], dim=1)
    
    # 2. Generate Input Sensors and Ground Truth
    with torch.no_grad():
        a_coeffs = test_case['a_coeffs']
        b_coeffs = test_case['b_coeffs']
        
        # Inputs (Sensors)
        u0_sensors = model.generate_ic_sine_series(a_coeffs)
        v0_sensors = model.generate_ic_sine_series(b_coeffs)
        
        # Targets (Ground Truth Field at all points)
        u0_true_field = model.generate_ic_sine_series(a_coeffs, X.flatten(), Y.flatten())
        v0_true_field = model.generate_ic_sine_series(b_coeffs, X.flatten(), Y.flatten())
        
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
    
    # ===== DISPLACEMENT ROW =====
    # PINN Displacement
    c00 = axes[0, 0].contourf(X.cpu().numpy(), Y.cpu().numpy(), u0_pred, levels=50, cmap='RdBu_r', vmax=vmax_u, vmin=-vmax_u)
    axes[0, 0].set_ylabel('y', fontsize=11)
    axes[0, 0].set_title(r'$u_0$ PINN', fontsize=11, fontweight='bold')
    fig.colorbar(c00, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # Ground Truth Displacement
    c01 = axes[0, 1].contourf(X.cpu().numpy(), Y.cpu().numpy(), u0_true, levels=50, cmap='RdBu_r', vmax=vmax_u, vmin=-vmax_u)
    axes[0, 1].set_title(r'$u_0$ Ground', fontsize=11, fontweight='bold')
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
    axes[1, 1].set_title(r'$v_0$ Ground', fontsize=11, fontweight='bold')
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
    log.info(f"Displacement error - MAE: {np.mean(np.abs(u0_error)):.6e}, Max: {np.max(np.abs(u0_error)):.6e}")
    log.info(f"Velocity error     - MAE: {np.mean(np.abs(v0_error)):.6e}, Max: {np.max(np.abs(v0_error)):.6e}")
    model.train() # Restore training mode