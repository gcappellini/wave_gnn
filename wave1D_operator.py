"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import os
from plot import plot_loss_components, plot_weights_NTK, plot_eigenvalues_spectra, plot_comparison
from datetime import datetime
import json
import torch

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join('logs_pideeponet', timestamp)
os.makedirs(log_dir, exist_ok=True)

seed = 2
np.random.seed(seed)
dde.config.set_random_seed(seed)

c = 1.0
a = 2.0
k = 0.0

iters = 5000  # Number of training iterations (40000)

load_from = None #"./logs_pideeponet/20251111-094705"

lr_0 = 0.001
decay = 0.3
epochs_decay = 5000

num_domain = 600
num_boundary = 0
num_initial = 20  # Total points = 600 + 100 = 700; 700/50 = 14 batches exactly
num_test = 500

eval_fcts = 200
batch_size = 50

ic_points = 20
hidden_feats = 64

ic_scale=1.75

# PDE (homogeneous wave equation - no forcing needed)
def pde_op(x, y, v):
    """Wave equation PDE: u_tt - c^2 * u_xx = 0
    No forcing term since func_space is used for initial conditions only.
    """
    u_t = dde.grad.jacobian(y, x, i=0, j=1)
    u_xx = dde.grad.hessian(y, x, i=0, j=0)
    u_tt = dde.grad.hessian(y, x, i=1, j=1)
    
    return u_tt - c**2 * u_xx

# Define u0_func from the sampled function for analytical solution
def u0_func(x):
    """Initial condition sampled from func_space with boundary compatibility.
    u0(x) = v(x) * x * (1-x) ensures u0(0) = u0(1) = 0
    """
    x_space = x[:, 0:1] if x.ndim > 1 else x
    return np.sin(np.pi * x_space) + np.sin(a * np.pi * x_space)

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

def analytical_solution(ins, u0_func, c):
    """
    d'Alembert solution for 1D wave equation with zero initial velocity.
    
    u(x,t) = 1/2 * [u0(x - ct) + u0(x + ct)]
    
    Args:
        x: spatial coordinate (N,)
        t: temporal coordinate (N,)
        u0_func: initial condition function
        c: wave speed
        
    Returns:
        u: solution at (x,t)
    """
    # Ensure inputs are arrays
    x = ins[:,0:1]
    t = ins[:,1:2]
    
    # Evaluate at shifted positions
    x_left = x - c * t
    x_right = x + c * t
    
    # # Apply periodic extension for points outside [0,1]
    # x_left = x_left % 1.0
    # x_right = x_right % 1.0
    
    u_left = u0_func(x_left.reshape(-1, 1))
    u_right = u0_func(x_right.reshape(-1, 1))
    
    return 0.5 * (u_left + u_right)

# Initial conditions now depend on the function from func_space
# func_space will provide u0(x) - the initial displacement
# Apply boundary transform to ensure u0(0) = u0(1) = 0
def ic_func(x, v):
    """Initial condition with boundary compatibility.
    In IC/BC context, x is typically just the coordinates (not a tuple).
    x has shape (n_points, 2) where columns are [x_space, t_time].
    v has shape (n_points, 2) where columns are [x_space, function_values].
    Works with both numpy arrays and PyTorch tensors.
    """
    # For initial conditions, x is directly the coordinate array, not a tuple
    x_spatial = x[:, 0:1]  # Extract spatial coordinate (first column)
    
    # FIXED: Extract only function values from v (second column)
    # v[:, 0] are the spatial coordinates, v[:, 1] are the function values
    if torch.is_tensor(v):
        v_vals = v[:, 1:2]  # Extract function values (second column)
    else:
        v_vals = v[:, 1:2] if v.ndim == 2 else v.reshape(-1, 1)
    
    # Apply boundary transform: u0(x) = v(x) * x * (1-x)
    # This ensures u0(0) = u0(1) = 0
    
    # Check if inputs are tensors or numpy arrays
    if torch.is_tensor(x_spatial):
        x_spatial = x_spatial.squeeze()
        if not torch.is_tensor(v_vals):
            v_vals = torch.tensor(v_vals, dtype=x_spatial.dtype, device=x_spatial.device)
        v_vals = v_vals.squeeze()

        shape = v_vals * x_spatial * (1.0 - x_spatial)
        shape = ic_scale * shape / torch.max(torch.abs(shape))
    else:
        # NumPy path
        v_arr = np.asarray(v_vals).squeeze()
        x_spatial = np.asarray(x_spatial).squeeze()
        
        shape = v_arr * x_spatial * (1.0 - x_spatial)
        shape = ic_scale * shape / np.max(np.abs(shape))
    
    return shape

def compute_ntk_weights_from_matrices(K_pde, K_ic1, K_ic2):
    """
    Compute adaptive weights from full NTK matrices.
    Uses the trace of each matrix for weight computation.
    
    Args:
        K_pde: NTK Gram matrix for PDE residual (n_pde x n_pde)
        K_ic1: NTK Gram matrix for IC1 (n_ic x n_ic)
        K_ic2: NTK Gram matrix for IC2 (n_ic x n_ic)
    
    Returns:
        weights: [lambda_pde, lambda_ic1, lambda_ic2]
        traces: [trace_pde, trace_ic1, trace_ic2]
    """
    # Compute traces
    trace_pde = np.trace(K_pde)
    trace_ic1 = np.trace(K_ic1)
    trace_ic2 = np.trace(K_ic2)
    
    # Total trace
    total_trace = trace_pde + trace_ic1 + trace_ic2
    
    # Compute adaptive weights using Algorithm 1 formula
    lambda_pde = total_trace / (trace_pde + 1e-10)
    lambda_ic1 = total_trace / (trace_ic1 + 1e-10)
    lambda_ic2 = total_trace / (trace_ic2 + 1e-10)
    
    weights = np.array([lambda_pde, lambda_ic1, lambda_ic2])
    traces = np.array([trace_pde, trace_ic1, trace_ic2])
    
    return weights, traces


def compute_ntk_matrices(model, data):
    """
    Compute full NTK Gram matrices for operator learning with DeepONet.
    
    For operator learning, the data structure is different:
    - train_x: tuple (v_branch, x_trunk) where v_branch are function samples
    - train_aux_vars: v(x), function values at coordinates
    
    We compute NTK for:
    - K_pde: PDE residual at interior spatiotemporal points
    - K_ic1: Initial condition u(x,0) = u0(x) 
    - K_ic2: Initial velocity u_t(x,0) = 0
    
    Returns matrices K_pde, K_ic1, K_ic2 where K[i,j] = <J_i, J_j>
    """
    
    # Get training data
    v_branch_train, x_trunk_train = data.train_x  # Tuple of (branch_input, trunk_input)
    v_aux_train = data.train_aux_vars  # Function values v(x)
    
    # Get parameters from the PDE data object
    num_domain = data.pde.num_domain
    num_initial = data.pde.num_initial
    
    # Get trainable parameters
    params = [p for p in model.net.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in params)
    
    print(f"  Computing NTK matrices for operator learning")
    print(f"  Branch input shape: {v_branch_train.shape}, Trunk input shape: {x_trunk_train.shape}")
    print(f"  Aux vars shape: {v_aux_train.shape}, Num params: {num_params}")
    print(f"  PDE points: {num_domain}, IC points: {num_initial}")
    
    # For operator learning, we need to sample a few functions from training set
    # and compute NTK for their PDE/IC losses
    # Use a subset of functions to make computation tractable
    n_functions_sample = min(10, v_branch_train.shape[0])
    func_indices = np.random.choice(v_branch_train.shape[0], n_functions_sample, replace=False)
    
    # Initialize Jacobian matrices
    # We'll compute for sampled functions and concatenate
    J_pde_list = []
    J_ic1_list = []
    J_ic2_list = []
    
    for func_idx in func_indices:
        # Get the function sample (branch input)
        v_branch = torch.tensor(v_branch_train[func_idx:func_idx+1], dtype=torch.float32)
        
        # Separate trunk points into PDE (domain) and IC points
        # Assume first num_domain points are interior, next num_initial are IC
        # (This may need adjustment based on how DeepXDE orders points)
        pde_trunk_indices = np.where(x_trunk_train[:, 1] > 1e-6)[0][:num_domain]  # t > 0
        ic_trunk_indices = np.where(np.abs(x_trunk_train[:, 1]) < 1e-6)[0][:num_initial]  # t ≈ 0
        
        # === Compute PDE Jacobian ===
        for trunk_idx in pde_trunk_indices:
            x_trunk = torch.tensor(x_trunk_train[trunk_idx:trunk_idx+1], dtype=torch.float32, requires_grad=True)
            
            # Forward pass through DeepONet
            y = model.net((v_branch, x_trunk))
            
            # Get auxiliary variable for this function at this point
            v_val = v_aux_train[func_idx, trunk_idx]
            v_tensor = torch.tensor([[v_val]], dtype=torch.float32)
            
            # Compute PDE residual
            residual = pde_op(x_trunk, y, v_tensor)
            residual_scalar = residual.sum()
            
            # Compute gradients w.r.t. parameters
            grads = torch.autograd.grad(
                outputs=residual_scalar,
                inputs=params,
                create_graph=False,
                retain_graph=False,
                allow_unused=True
            )
            
            grad_vec = []
            for g in grads:
                if g is not None:
                    grad_vec.append(g.detach().cpu().numpy().flatten())
                else:
                    grad_vec.append(np.zeros(1))
            J_pde_list.append(np.concatenate(grad_vec))
        
        # === Compute IC1 Jacobian (u at t=0) ===
        for trunk_idx in ic_trunk_indices:
            x_trunk = torch.tensor(x_trunk_train[trunk_idx:trunk_idx+1], dtype=torch.float32, requires_grad=True)
            
            # Forward pass
            y = model.net((v_branch, x_trunk))
            
            # Target: ic_func(x, v) with the function sample
            x_np = x_trunk.detach().cpu().numpy()
            v_np = v_aux_train[func_idx, trunk_idx]
            target_np = ic_func(x_np, v_np)
            target = torch.tensor(target_np, dtype=torch.float32)
            
            loss = (y - target).sum()
            
            grads = torch.autograd.grad(
                outputs=loss,
                inputs=params,
                create_graph=False,
                retain_graph=False,
                allow_unused=True
            )
            
            grad_vec = []
            for g in grads:
                if g is not None:
                    grad_vec.append(g.detach().cpu().numpy().flatten())
                else:
                    grad_vec.append(np.zeros(1))
            J_ic1_list.append(np.concatenate(grad_vec))
        
        # === Compute IC2 Jacobian (u_t at t=0) ===
        for trunk_idx in ic_trunk_indices:
            x_trunk = torch.tensor(x_trunk_train[trunk_idx:trunk_idx+1], dtype=torch.float32, requires_grad=True)
            
            # Forward pass
            y = model.net((v_branch, x_trunk))
            
            # Compute u_t
            u_t = dde.grad.jacobian(y, x_trunk, i=0, j=1)
            u_t_scalar = u_t.sum()
            
            grads = torch.autograd.grad(
                outputs=u_t_scalar,
                inputs=params,
                create_graph=False,
                retain_graph=False,
                allow_unused=True
            )
            
            grad_vec = []
            for g in grads:
                if g is not None:
                    grad_vec.append(g.detach().cpu().numpy().flatten())
                else:
                    grad_vec.append(np.zeros(1))
            J_ic2_list.append(np.concatenate(grad_vec))
    
    # Convert lists to arrays
    J_pde = np.array(J_pde_list) if J_pde_list else np.zeros((1, num_params))
    J_ic1 = np.array(J_ic1_list) if J_ic1_list else np.zeros((1, num_params))
    J_ic2 = np.array(J_ic2_list) if J_ic2_list else np.zeros((1, num_params))
    
    print(f"  Jacobian shapes: J_pde={J_pde.shape}, J_ic1={J_ic1.shape}, J_ic2={J_ic2.shape}")
    
    # Compute Gram matrices K = J @ J^T
    K_pde = J_pde @ J_pde.T
    K_ic1 = J_ic1 @ J_ic1.T
    K_ic2 = J_ic2 @ J_ic2.T
    
    return K_pde, K_ic1, K_ic2


class AdaptiveWeightCallback(dde.callbacks.Callback):
    def __init__(self, model, data, update_every=100, plot_checkpoints=[0, 10000, 20000, 30000, 40000]):
        super().__init__()
        self.model = model
        self.data = data
        self.update_every = update_every
        self.plot_checkpoints = plot_checkpoints  # Epochs where we save matrices for plotting
        
        # Initialize logging lists
        self.epochs_log = []
        self.K_pde_log = []
        self.K_ic1_log = []
        self.K_ic2_log = []
        self.lambda_pde_log = []
        self.lambda_ic1_log = []
        self.lambda_ic2_log = []
        
        # Storage for full NTK matrices at plot checkpoints only
        self.checkpoint_K_pde = []
        self.checkpoint_K_ic1 = []
        self.checkpoint_K_ic2 = []
        self.checkpoint_epochs = []
        
    def on_epoch_end(self):
        epoch = self.model.train_state.epoch
        
        if epoch % self.update_every == 0:
            print(f"\n[Epoch {epoch}] Computing full NTK matrices for weight adaptation...")
            
            # Compute full NTK matrices
            K_pde, K_ic1, K_ic2 = compute_ntk_matrices(self.model, self.data)
            
            # Compute adaptive weights from the matrices
            weights, traces = compute_ntk_weights_from_matrices(K_pde, K_ic1, K_ic2)
            
            # Log epoch, traces, and weights
            self.epochs_log.append(epoch)
            self.K_pde_log.append(traces[0])
            self.K_ic1_log.append(traces[1])
            self.K_ic2_log.append(traces[2])
            self.lambda_pde_log.append(weights[0])
            self.lambda_ic1_log.append(weights[1])
            self.lambda_ic2_log.append(weights[2])
            
            print(f"  Traces: K_pde={traces[0]:.2e}, K_ic1={traces[1]:.2e}, K_ic2={traces[2]:.2e}")
            print(f"  Weights: λ_pde={weights[0]:.2e}, λ_ic1={weights[1]:.2e}, λ_ic2={weights[2]:.2e}")
            
            # Store full matrices only at plot checkpoints
            if epoch in self.plot_checkpoints:
                print(f"  -> Storing full matrices for eigenvalue plotting (checkpoint)")
                self.checkpoint_K_pde.append(K_pde)
                self.checkpoint_K_ic1.append(K_ic1)
                self.checkpoint_K_ic2.append(K_ic2)
                self.checkpoint_epochs.append(epoch)
            
            # Get current learning rate from PyTorch optimizer
            current_lr = self.model.opt.param_groups[0]['lr']
            print(f"  Current LR: {current_lr:.2e}")
            
            # Update loss weights - preserve current learning rate
            self.model.compile(
                "adam",
                lr=current_lr,
                loss_weights=weights,
                decay=("inverse time", epochs_decay, decay),
            )

if __name__ == "__main__":
    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    ic = dde.icbc.OperatorBC(
        geomtime, 
        lambda x, y, v: y - ic_func(x, v), 
        lambda _, on_initial: on_initial
    )

    # Initial velocity remains zero
    ic_2 = dde.icbc.OperatorBC(
        geomtime,
        lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=1),
        lambda _, on_initial: on_initial
        # lambda x, _: dde.utils.isclose(x[1], 0),
    )

    pde = dde.data.TimePDE(
        geomtime,
        pde_op,
        [ic, ic_2],
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
        num_test=num_test,
    )

    # Function space - represents the distribution of initial displacements u0(x)
    # GRF generates smooth random functions with correlation length_scale=0.2
    func_space = dde.data.GRF(length_scale=0.4)

    # Data - now learning operator from initial conditions u0(x) to solution u(x,t)
    eval_pts = np.linspace(0, 1, num=ic_points)[:, None]
    data = dde.data.PDEOperatorCartesianProd(
        pde, func_space, eval_pts, eval_fcts, 
        function_variables=[0],  # Function variable is used in IC, not PDE
        num_test=num_test, 
        batch_size=batch_size
    )

    # Net
    net = dde.nn.DeepONetCartesianProd(
        [ic_points, hidden_feats, hidden_feats, hidden_feats],
        [2, hidden_feats, hidden_feats, hidden_feats],
        "tanh",
        "Glorot normal",
    )

    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)
    model.compile("adam", lr=lr_0, decay=("inverse time", epochs_decay, decay))

    if iters>0:
        print("Computing initial NTK matrices and weights...")
        K_pde_init, K_ic1_init, K_ic2_init = compute_ntk_matrices(model, data)
        initial_weights, initial_traces = compute_ntk_weights_from_matrices(K_pde_init, K_ic1_init, K_ic2_init)
        print(f"Initial adaptive weights: {initial_weights}")
        print(f"Initial traces: {initial_traces}")

        model.compile("adam", lr=lr_0,     
                    # metrics=["l2 relative error"],
                    decay=("inverse time", epochs_decay, decay),
                    loss_weights=initial_weights)
        
    if load_from is not None:
        model_files = [f for f in os.listdir(load_from) if f.startswith("model.ckpt")]
        if not model_files:
            raise FileNotFoundError(f"No model checkpoint found in {load_from}")
        model_path = os.path.join(load_from, model_files[0])
        model.restore(model_path, verbose=1)
        print(f"✓ Model loaded successfully from {model_path}")

    if iters > 0:
        plot_checkpoints = [0, iters // 4, iters // 2, 3 * iters // 4, iters]
        print(f"Eigenvalue plots will be saved at epochs: {plot_checkpoints}")
        
        adaptive_weight_callback = AdaptiveWeightCallback(model, data, update_every=100, plot_checkpoints=plot_checkpoints)
        
        # Store initial matrices (epoch 0) in the callback
        adaptive_weight_callback.checkpoint_K_pde.append(K_pde_init)
        adaptive_weight_callback.checkpoint_K_ic1.append(K_ic1_init)
        adaptive_weight_callback.checkpoint_K_ic2.append(K_ic2_init)
        adaptive_weight_callback.checkpoint_epochs.append(0)

        start = datetime.now()
        losshistory, train_state = model.train(iterations=iters, model_save_path=f"{log_dir}/model.ckpt", callbacks=[adaptive_weight_callback])
        end = datetime.now()
        training_time = end - start
        print(f"Training time: {training_time}")
    else:
        training_time = 0


    func_feats = func_space.random(1)
    xs = np.linspace(0, 1, num=100)[:, None]
    v = func_space.eval_batch(func_feats, xs)[0]

    # Sample multiple functions from func_space for comparison
    n_samples = 5
    func_feats_samples = func_space.random(n_samples)
    xs = np.linspace(0, 1, num=100)[:, None]

    # Evaluate u0_func (the target initial condition)
    u0_values = u0_func(xs)

    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(xs, u0_values, 'k-', linewidth=2, label='u0_func (target)', zorder=10)
 
    # Plot sampled functions from func_space
    for i in range(n_samples):
        v_sample = func_space.eval_batch(func_feats_samples[i:i+1], xs)[0]
        # Apply ic_func transform to match IC
        v_transformed = ic_func(xs, v_sample)
        plt.plot(xs, v_transformed, '--', alpha=0.6, label=f'GRF sample {i+1}')

    plt.xlabel('x')
    plt.ylabel('u(x, 0)')
    plt.title('Comparison: Target Initial Condition vs GRF Samples')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(log_dir, 'ic_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Initial condition comparison plot saved to {log_dir}/ic_comparison.png")


    N_x = 100
    N_t = 100
    x_vis = np.linspace(0, 1, N_x)
    t_vis = np.linspace(0, 1, N_t)
    X, T = np.meshgrid(x_vis, t_vis)
    eval_pts_vis = np.hstack([X.reshape(-1, 1), T.reshape(-1, 1)])

    u_true = analytical_solution(eval_pts_vis, u0_func, c)
    u_true = u_true.reshape((N_x, N_t))#.T


    # For prediction, sample the SAME initial condition u0_func at eval_pts
    # This creates the branch input that corresponds to the true solution
    u0_samples = u0_func(eval_pts)  # Sample u0(x) at the 50 sensor locations
    # Apply boundary transform to match what IC expects
    u0_samples_transformed = u0_samples * eval_pts * (1.0 - eval_pts)
    v_branch = u0_samples_transformed.T  # Shape: (1, 50)

    xv, tv = np.meshgrid(x_vis, t_vis)
    x_trunk = np.vstack((np.ravel(xv), np.ravel(tv))).T
    u_pred = model.predict((v_branch, x_trunk))
    u_pred = u_pred.reshape((N_x, N_t))
    l2_err = dde.metrics.l2_relative_error(u_true, u_pred)

    print(l2_err)
    plot_comparison(u_true, u_pred, l2_err, log_dir)

    params = {
        'seed': seed,
        'c': c,
        'k': k,
        'num_domain': num_domain,
        'num_boundary': num_boundary,
        'num_initial': num_initial,
        'num_test': num_test,
        # 'm': m,
        'batch_size': batch_size,
        'lr_0': lr_0,
        'decay': decay,
        'epochs_decay': epochs_decay,
        'ic_points': ic_points,
        'eval_fcts': eval_fcts,
        'hidden_feats': hidden_feats,
        'iters': iters,
        'load_from': load_from,
        'l2_relative_error': float(l2_err),
        'tr_time': str(training_time),
        'tr_time_seconds': training_time.total_seconds() if training_time != 0 else 0
    }
    with open(os.path.join(log_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=2)

    if iters > 0:
            # Plot all loss components separately
            plot_loss_components(losshistory, log_dir)


            if len(adaptive_weight_callback.epochs_log) > 0:
                plot_weights_NTK(adaptive_weight_callback, log_dir)
            
            # Plot eigenvalue spectra at checkpoints
            if len(adaptive_weight_callback.checkpoint_epochs) > 0:
                plot_eigenvalues_spectra(adaptive_weight_callback, iters, log_dir)