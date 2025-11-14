"""Backend supported: tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np
import os
from datetime import datetime
from plot import plot_loss_components, plot_comparison
import matplotlib.pyplot as plt
from wave1D_operator import SineSeries
import torch

# Get script directory for absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class CombinedFunctionSpace:
    def __init__(self, func_space_1, func_space_2, func_space_3=None):
        self.func_space_1 = func_space_1
        self.func_space_2 = func_space_2
        self.func_space_3 = func_space_3
        self.n1 = func_space_1.N
        self.n2 = func_space_2.N
        self.n3 = func_space_3.N if func_space_3 is not None else 0
        
    def random(self, size):
        features_1 = self.func_space_1.random(size)
        features_2 = self.func_space_2.random(size)
        if self.func_space_3 is not None:
            features_3 = self.func_space_3.random(size)
            return np.hstack([features_1, features_2, features_3]).astype(np.float32)
        return np.hstack([features_1, features_2]).astype(np.float32)
    
    def eval_one(self, feature, x):
        f1 = self.func_space_1.eval_one(feature[:self.n1], x)
        f2 = self.func_space_2.eval_one(feature[self.n1:self.n1+self.n2], x)
        if self.func_space_3 is not None:
            f3 = self.func_space_3.eval_one(feature[self.n1+self.n2:self.n1+self.n2+self.n3], x)
            result = np.concatenate([f1.ravel(), f2.ravel(), f3.ravel()])
        else:
            result = np.concatenate([f1.ravel(), f2.ravel()])
        return result.astype(np.float32)
    
    def eval_batch(self, features, xs):
        f1 = self.func_space_1.eval_batch(features[:, :self.n1], xs)
        f2 = self.func_space_2.eval_batch(features[:, self.n1:self.n1+self.n2], xs)
        if self.func_space_3 is not None:
            f3 = self.func_space_3.eval_batch(features[:, self.n1+self.n2:self.n1+self.n2+self.n3], xs)
            return np.hstack([f1, f2, f3]).astype(np.float32)
        return np.hstack([f1, f2]).astype(np.float32)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(SCRIPT_DIR, 'logs_pideeponet', timestamp)
os.makedirs(log_dir, exist_ok=True)

seed = 2
np.random.seed(seed)
dde.config.set_random_seed(seed)

c = 1.0
k = 1.0

training = True 
load_from = None #os.path.join(SCRIPT_DIR, 'logs_pideeponet/20251113-123841_u0_f')  #

# def compute_second_derivative_fd(func_values, x_coords):
#     """
#     Compute second derivative of a function via finite differences.
    
#     Args:
#         func_values: Function values at x_coords (shape: batch_size x n_points)
#         x_coords: Coordinates where function is evaluated (1D array)
    
#     Returns:
#         Second derivatives at same points (same shape as func_values)
#     """
#     # Use PyTorch or NumPy depending on backend
#     backend_name = dde.backend.backend_name
    
#     if backend_name == "pytorch":
#         import torch
#         # Convert to numpy for finite differences, then back to torch
#         func_np = func_values.detach().cpu().numpy()
        
#         # Compute finite differences on the grid
#         dx = x_coords[1] - x_coords[0]  # Uniform grid spacing
#         d2f_dx2 = np.zeros_like(func_np)
        
#         # Central difference for interior points: f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
#         d2f_dx2[:, 1:-1] = (func_np[:, 2:] - 2*func_np[:, 1:-1] + func_np[:, :-2]) / (dx**2)
        
#         # Forward/backward differences for boundaries
#         d2f_dx2[:, 0] = (func_np[:, 2] - 2*func_np[:, 1] + func_np[:, 0]) / (dx**2)
#         d2f_dx2[:, -1] = (func_np[:, -1] - 2*func_np[:, -2] + func_np[:, -3]) / (dx**2)
        
#         return torch.from_numpy(d2f_dx2).to(func_values.device).float()
#     else:
#         # NumPy backend
#         dx = x_coords[1] - x_coords[0]
#         d2f_dx2 = np.zeros_like(func_values)
        
#         d2f_dx2[:, 1:-1] = (func_values[:, 2:] - 2*func_values[:, 1:-1] + func_values[:, :-2]) / (dx**2)
#         d2f_dx2[:, 0] = (func_values[:, 2] - 2*func_values[:, 1] + func_values[:, 0]) / (dx**2)
#         d2f_dx2[:, -1] = (func_values[:, -1] - 2*func_values[:, -2] + func_values[:, -3]) / (dx**2)
        
#         return d2f_dx2

# PDE
def pde_op(x, y, v):
    f = v[:, 0:1]       # First force value
    u0 = v[:, 50:100]   # Initial displacement function values
    v0 = v[:, 100:150]  # Initial velocity function values
    
    grad_y = dde.zcs.LazyGrad(x, y)
    dy_t = grad_y.compute((0, 1))
    dy_tt = grad_y.compute((0, 2))
    dy_xx = grad_y.compute((2, 0))
    
    # Extract trunk coordinates from x
    if isinstance(x, tuple):
        x_trunk = x[1]
    else:
        x_trunk = x
    
    t_coords = 0.1 #x_trunk[:, 1:2]  # Time coordinate
    
    # Compute second derivatives of u0 and v0 via finite differences
    grad_v = dde.zcs.LazyGrad(x, v)
    dv_xx = grad_v.compute((2, 0))
    du0_xx = dv_xx[:, 50:100]
    dv0_xx = dv_xx[:, 100:150]

    # PDE with IC terms as you had them
    return dy_tt - c * dy_xx + k * dy_t - 10 * f - c*(du0_xx + t_coords*dv0_xx) + k*v0

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

    ic = dde.icbc.IC(geomtime, lambda _: 0, lambda _, on_initial: on_initial)
    # ic = dde.icbc.OperatorBC(
    #     geomtime,
    #     lambda x, y, v: 0,  # u(x,0) should match u0
    #     lambda _, on_initial: on_initial
    # )

    ic_2 = dde.icbc.OperatorBC(
        geomtime,
        lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=1),
        lambda _, on_initial: on_initial
    )

    pde = dde.data.TimePDE(
        geomtime,
        pde_op,
        [ic, ic_2],
        num_domain=200,
        num_boundary=40,
        num_initial=20,
        num_test=500,
    )

    # Function space
    func_space_f = dde.data.GRF(length_scale=0.2)
    func_space_u0 = SineSeries(N=3)
    func_space_v0 = SineSeries(N=3)
    func_space_combined = CombinedFunctionSpace(func_space_f, func_space_u0, func_space_v0)

    # Data
    eval_pts = np.linspace(0, 1, num=50)[:, None]
    data = dde.zcs.PDEOperatorCartesianProd(
        pde, 
        func_space_combined, 
        eval_pts, 
        1000, 
        function_variables=[0], 
        num_test=100, 
        batch_size=50
    )

    # Net
    net = dde.nn.DeepONetCartesianProd(
        [150, 64, 64, 64],  # Branch: 50 (force) + 50 (u0) + 50 (v0) = 150
        [2, 64, 64, 64],
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

    v_branch = np.loadtxt(os.path.join(SCRIPT_DIR, 'v_branch.csv'), delimiter=',')

    if v_branch.ndim == 1:
        v_branch = v_branch.reshape(1, -1)

    u0_branch = np.zeros_like(v_branch)
    v0_branch = np.zeros_like(v_branch)
    
    combined_branch = np.hstack([v_branch, u0_branch, v0_branch])

    xv = np.loadtxt(os.path.join(SCRIPT_DIR, 'gt_wave1D.csv'), delimiter=',', usecols=0)
    tv = np.loadtxt(os.path.join(SCRIPT_DIR, 'gt_wave1D.csv'), delimiter=',', usecols=1)
    u_true = np.loadtxt(os.path.join(SCRIPT_DIR, 'gt_wave1D.csv'), delimiter=',', usecols=2)

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
    u_pred = model.predict((combined_branch, x_trunk))
    u_pred = u_pred.reshape((len(np.unique(tv)), len(np.unique(xv)))).T  
    l2_err = dde.metrics.l2_relative_error(u_true, u_pred)

    plot_comparison(u_true, u_pred, l2_err, log_dir)

    # Save inputs and predictions to CSV
    np.savetxt(os.path.join(log_dir, 'v_branch.csv'), v_branch, delimiter=',')
    np.savetxt(os.path.join(log_dir, 'x_trunk.csv'), x_trunk, delimiter=',')
    np.savetxt(os.path.join(log_dir, 'u_pred.csv'), u_pred, delimiter=',')

    with open(os.path.join(log_dir, 'training_time.txt'), 'w') as f:
        f.write(f"Training time: {tr_t}\n")

    rollout = np.loadtxt(os.path.join(SCRIPT_DIR, 'gt_wave1D_rollout.csv'), delimiter=',')
    
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

    
    # Get unique x and t values (must form a regular grid)
    x_unique = np.unique(xv_rollout)
    t_unique = np.unique(tv_rollout)

    nx = len(x_unique)
    nt = len(t_unique)

    # Create trunk input for prediction on [0,1] time range
    x_trunk_pred = np.vstack((np.ravel(xv), np.ravel(tv))).T

    # Prepare storage for predictions
    look_up_every = 100
    u_pred_rollout = np.zeros((int(n_time/look_up_every), n_space))

    u_store = []
    for idx, t in enumerate(t_unique):
        if idx % look_up_every == 0:
            print(f"Updating u0 at time step {idx}, t={t:.2f}")
            u0_t0 = rollout[rollout[:, 1]==t, 3]
            u0_t0_interp = np.interp(eval_pts.ravel(), x_unique, u0_t0)

            f_t0 = rollout[rollout[:, 1]==t, 2]
            f_t0_interp = np.interp(eval_pts.ravel(), x_unique, f_t0)

            # v0 set to zero (initial velocity = 0)
            v0_t0_interp = np.zeros_like(u0_t0_interp)

            branch_t0 = np.hstack([
                f_t0_interp.reshape(-1, 1).T,
                u0_t0_interp.reshape(-1, 1).T,
                v0_t0_interp.reshape(-1, 1).T
            ])

            u_pred_t = model.predict((branch_t0, x_trunk_pred))
            u_store.extend(u_pred_t.ravel())
 
    u_pred_rollout_2d = np.array(u_store).reshape(n_time, n_space)
    
    l2_err = dde.metrics.l2_relative_error(u_true_rollout_2d, u_pred_rollout_2d)
    plot_comparison(u_true_rollout_2d, u_pred_rollout_2d, l2_err, log_dir, roll=True, t_max=t_unique[-1])