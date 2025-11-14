"""Backend supported: tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np
import os
from datetime import datetime
from plot import plot_loss_components, plot_comparison
import matplotlib.pyplot as plt

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join('logs_pideeponet', timestamp)
os.makedirs(log_dir, exist_ok=True)

seed = 2
np.random.seed(seed)
dde.config.set_random_seed(seed)

c = 1.0
k = 1.0

training = True
load_from = None  # 'logs_pideeponet/20251112-162220_warmstart'  #

# PDE
def pde(x, y, v):

    grad_y = dde.zcs.LazyGrad(x, y)
    dy_t = grad_y.compute((0, 1))
    dy_tt = grad_y.compute((0, 2))
    dy_xx = grad_y.compute((2, 0))
    return dy_tt - c * dy_xx + k * dy_t - 10 * v

def output_transform(x, y):
    """
    Enforce boundary conditions through output transformation.
    
    In DeepONet, x can be:
    - A tuple (x_branch, x_trunk) during training
    - Just x_trunk array during prediction
    
    We need x_trunk which has shape (n_points, 2) where columns are [x_space, t_time]
    """
    # Handle both tuple and array cases
    if isinstance(x, tuple):
        x_trunk = x[1]  # Extract trunk coordinates from tuple
    else:
        x_trunk = x
    
    x_coord = x_trunk[:, 0:1]  # Spatial coordinate (first column)
    
    # Boundary transform: x*(1-x) vanishes at x=0 and x=1
    bc_transform = x_coord * (1.0 - x_coord)
    
    # Apply transform: u_transformed = u_net * x * (1-x)
    return y * bc_transform.T

if __name__ == "__main__":
    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # bc = dde.icbc.DirichletBC(geomtime, lambda _: 0, lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(geomtime, lambda _: 0, lambda _, on_initial: on_initial)

    ic_2 = dde.icbc.OperatorBC(
        geomtime,
        lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=1),
        lambda _, on_initial: on_initial
    )

    pde = dde.data.TimePDE(
        geomtime,
        pde,
        [ic, ic_2],
        num_domain=200,
        num_boundary=40,
        num_initial=20,
        num_test=500,
    )

    # Function space
    func_space = dde.data.GRF(length_scale=0.2)

    # Data
    eval_pts = np.linspace(0, 1, num=50)[:, None]
    data = dde.zcs.PDEOperatorCartesianProd(
        pde, func_space, eval_pts, 1000, function_variables=[0], num_test=100, batch_size=50
    )

    # Net
    net = dde.nn.DeepONetCartesianProd(
        [50, 128, 128, 32],
        [2, 128, 128, 32],
        "tanh",
        "Glorot normal",
    )
    net.apply_output_transform(output_transform)
    model = dde.zcs.Model(data, net)
    model.compile("adam", lr=0.0005)

    if load_from is not None:
        model_files = [f for f in os.listdir(load_from) if f.startswith("model.ckpt")]
        if not model_files:
            raise FileNotFoundError(f"No model checkpoint found in {load_from}")
        model_path = os.path.join(load_from, model_files[0])
        model.restore(model_path, verbose=1)

    if training:
        start = datetime.now()
        losshistory, train_state = model.train(iterations=20000, model_save_path=f"{log_dir}/model.ckpt")
        end = datetime.now()
        tr_t = end - start
        plot_loss_components(losshistory, log_dir)
    else:
        tr_t=0

    v_branch = np.loadtxt('./v_branch.csv', delimiter=',')

    # Reshape v_branch to (1, 50) for DeepONet branch network
    # DeepONet expects shape (batch_size, num_sensors)
    if v_branch.ndim == 1:
        v_branch = v_branch.reshape(1, -1)  # (50,) -> (1, 50)

    xv = np.loadtxt('gt_wave1D.csv', delimiter=',', usecols=0)
    tv = np.loadtxt('gt_wave1D.csv', delimiter=',', usecols=1)
    u_true = np.loadtxt('gt_wave1D.csv', delimiter=',', usecols=2)

    u_true = u_true.reshape((len(np.unique(tv)), len(np.unique(xv))))

    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(0, 1, num=50), v_branch.ravel(), 'b-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('v(x)')
    plt.title('Force v(x)')
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'v_branch.png'), dpi=300, bbox_inches='tight')
    plt.close()

    x_trunk = np.vstack((np.ravel(xv), np.ravel(tv))).T
    u_pred = model.predict((v_branch, x_trunk))
    u_pred = u_pred.reshape((len(np.unique(tv)), len(np.unique(xv)))).T  # Shape: (time, space) - no transpose!
    l2_err = dde.metrics.l2_relative_error(u_true, u_pred)

    plot_comparison(u_true, u_pred, l2_err, log_dir)

    # Save inputs and predictions to CSV
    np.savetxt(os.path.join(log_dir, 'v_branch.csv'), v_branch, delimiter=',')
    np.savetxt(os.path.join(log_dir, 'x_trunk.csv'), x_trunk, delimiter=',')
    np.savetxt(os.path.join(log_dir, 'u_pred.csv'), u_pred, delimiter=',')

    with open(os.path.join(log_dir, 'training_time.txt'), 'w') as f:
        f.write(f"Training time: {tr_t}\n")

    rollout = np.loadtxt('gt_wave1D_rollout.csv', delimiter=',')
    
    # Extract columns from MATLAB CSV: [x, t, force, u]
    xv_rollout = rollout[:, 0]  # Spatial coordinates
    tv_rollout = rollout[:, 1]  # Time coordinates
    f_rollout = rollout[:, 2]   # Force values
    u_true_rollout = rollout[:, 3]  # Solution values
    
    # Get dimensions
    n_time = len(np.unique(tv_rollout))  # Number of unique time points
    n_space = len(np.unique(xv_rollout))  # Number of unique spatial points
    
    # Reshape from 1D to 2D: (time, space)
    # MATLAB loop: outer=time, inner=space → fills space (columns) first
    # NumPy reshape with C-order (default): fills last dimension (columns) first
    f_rollout_2d = f_rollout.reshape((n_time, n_space))  # Shape: (time, space)
    u_true_rollout_2d = u_true_rollout.reshape((n_time, n_space))  # Shape: (time, space)
    
    # Plot force rollout
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get time and space bounds for extent
    t_min, t_max = tv_rollout.min(), tv_rollout.max()
    x_min, x_max = xv_rollout.min(), xv_rollout.max()
    
    # imshow expects (rows, cols) = (y-axis, x-axis)
    # Our data is (time, space), so we plot with:
    # - x-axis = space (horizontal)
    # - y-axis = time (vertical)
    im = ax.imshow(f_rollout_2d, 
                   extent=[x_min, x_max, t_min, t_max],
                   origin='lower',  # t=0 at bottom
                   aspect='auto',
                   cmap='RdBu_r',
                   interpolation='bilinear')
    
    plt.colorbar(im, ax=ax, label='Force f(x,t)')
    ax.set_xlabel('x (space)', fontsize=12)
    ax.set_ylabel('t (time)', fontsize=12)
    ax.set_title('Force Distribution f(x,t) - Rollout', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'f_rollout.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Force rollout plot saved to {log_dir}/f_rollout.png")


    x_trunk_rollout = np.vstack((xv_rollout, tv_rollout)).T
    
    # Get unique x and t values (must form a regular grid)
    x_unique = np.unique(xv_rollout)
    t_unique = np.unique(tv_rollout)

    # Reshape f_rollout into a 2D array: shape (nx, nt)
    # where nx = number of spatial points, nt = number of time instants
    nx = len(x_unique)
    nt = len(t_unique)
    f_grid = f_rollout.reshape(nx, nt, order='F' if tv_rollout[:nx].var() == 0 else 'C')

    # Interpolate along the x dimension to get f_branch
    # np.interp operates row-wise, so we broadcast properly:
    f_branch = np.array([
        np.interp(eval_pts.flatten(), x_unique, f_grid[:, j])
        for j in range(nt)
    ]).T

    u_pred = []

    for i in range(len(t_unique)):
        t_pred = t_unique[i] if t_unique[i] <= 1 else t_unique[i] % 1
        f_branch_i = f_branch[:, i].reshape(1, -1)  # Shape: (1, 50)
        t_i = np.full((nx, 1), t_pred)  # Shape: (nx, 1)
        x_trunk_i = np.hstack((x_unique.reshape(-1, 1), t_i))  # Shape: (nx, 2)
        u_pred_i = model.predict((f_branch_i, x_trunk_i)).flatten()  # Shape: (nx, 1)

        u_pred.append(u_pred_i)

    u_pred_rollout = np.array(u_pred)
    u_pred_rollout_2d = u_pred_rollout.reshape((len(np.unique(tv_rollout)), len(np.unique(xv_rollout))))
    l2_err = dde.metrics.l2_relative_error(u_true_rollout_2d, u_pred_rollout_2d)  # Shape: (time, space)

    plot_comparison(u_true_rollout_2d, u_pred_rollout_2d, l2_err, log_dir, roll=True, t_max=t_max)