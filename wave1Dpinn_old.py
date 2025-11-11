"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle
original code: https://github.com/lululxvi/deepxde/blob/master/examples/operator/diff_rec_aligned_pideeponet.py"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import json
import torch
from plot import plot_loss_components, plot_weights_NTK, plot_eigenvalues_spectra, plot_comparison
from fft import FourierFeatureTransform

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join('logs_pinn', timestamp)
os.makedirs(log_dir, exist_ok=True)

# Set random seeds for reproducibility
seed = 2
np.random.seed(seed)
dde.config.set_random_seed(seed)

c = 10.0
k = 0.0  # Damping coefficient
a = 2.0

load_from = None  # "./logs_pinn/20251110-184906_undamped_c1_plateau"

num_domain = 600
num_boundary = 0
num_initial = 100
num_test = 500

# Number of Fourier features per scale
# m = 4 # (16)
iters = 1000  # Number of training iterations (40000)

lr_0 = 0.001
epochs_decay = 5000
decay = 0.3

hidden_feats = 100
hidden_layers = 3

m_spatial=32       
m_temporal=32      
sigma_spatial=5.0  # Moderate spatial frequencies
sigma_temporal_list=[1.0, 10.0]  # Low and high temporal frequencies

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
    Compute full NTK Gram matrices (not just traces) for eigenvalue analysis.
    Returns matrices K_pde, K_ic1, K_ic2 where K[i,j] = <J_i, J_j>
    J_i is the Jacobian of loss component at point i w.r.t. parameters
    """
    
    # Get training points
    train_x = data.train_x_all
    num_domain = data.num_domain
    num_initial = data.num_initial
    
    pde_points = train_x[:num_domain]
    ic_points = train_x[num_domain:num_domain + num_initial]
    
    # Convert to PyTorch tensors
    pde_points_pt = torch.tensor(pde_points, dtype=torch.float32, requires_grad=True)
    ic_points_pt = torch.tensor(ic_points, dtype=torch.float32, requires_grad=True)
    
    # Get trainable parameters
    params = [p for p in model.net.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in params)
    
    print(f"  Computing NTK matrices: {num_domain} PDE pts, {num_initial} IC pts, {num_params} params")
    
    # Compute Jacobian matrix for PDE residuals
    J_pde = np.zeros((pde_points.shape[0], num_params))
    for i in range(pde_points.shape[0]):
        point_tensor = pde_points_pt[i:i+1].clone().detach().requires_grad_(True)
        print(model.net)
        y = model.net(point_tensor)
        residual = pde(point_tensor, y)
        residual_scalar = residual.sum()
        
        grads = torch.autograd.grad(
            outputs=residual_scalar,
            inputs=params,
            create_graph=False,
            retain_graph=False,
            allow_unused=True
        )
        
        # Flatten gradients into a single vector
        grad_vec = []
        for g in grads:
            if g is not None:
                grad_vec.append(g.detach().cpu().numpy().flatten())
            else:
                grad_vec.append(np.zeros(0))
        J_pde[i, :] = np.concatenate(grad_vec)
    
    # Compute Jacobian matrix for IC1 (u)
    J_ic1 = np.zeros((ic_points.shape[0], num_params))
    for i in range(ic_points.shape[0]):
        point_tensor = ic_points_pt[i:i+1].clone().detach().requires_grad_(True)
        y = model.net(point_tensor)
        target = torch.tensor(u0(point_tensor.detach().cpu().numpy()), dtype=torch.float32)
        if y.is_cuda:
            target = target.cuda()
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
                grad_vec.append(np.zeros(0))
        J_ic1[i, :] = np.concatenate(grad_vec)
    
    # Compute Jacobian matrix for IC2 (u_t)
    J_ic2 = np.zeros((ic_points.shape[0], num_params))
    for i in range(ic_points.shape[0]):
        point_tensor = ic_points_pt[i:i+1].clone().detach().requires_grad_(True)
        y = model.net(point_tensor)
        
        # Compute u_t using jacobian
        u_t = dde.grad.jacobian(y, point_tensor, i=0, j=1)
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
                grad_vec.append(np.zeros(0))
        J_ic2[i, :] = np.concatenate(grad_vec)
    
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
                metrics=["l2 relative error"],
                loss_weights=weights,
                decay=("inverse time", epochs_decay, decay),
            )

def get_initial_loss(model):
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(0)
    return losshistory.loss_train[0]


# PDE: wave equation without forcing (homogeneous)
def pde(x, u):
    """
    Compute PDE residual: u_tt - c^2 * u_xx + k * u_t = 0
    
    Args:
        x: input coordinates [x_space, t_time] (batch_size, 2)
            x[:, 0] = spatial coordinate (x)
            x[:, 1] = temporal coordinate (t)
        u: network output (batch_size, 1)
    """
    # DeepXDE's GeometryXTime convention: x[:, 0] = space, x[:, 1] = time
    # First derivatives: j parameter indicates which input column to differentiate w.r.t.
    u_t = dde.grad.jacobian(u, x, i=0, j=1)  # ∂u/∂t (j=1 for time)
    
    # Second derivatives (Hessian)
    u_xx = dde.grad.hessian(u, x, i=0, j=0)  # ∂²u/∂x² (j=0 for space)
    u_tt = dde.grad.hessian(u, x, i=1, j=1)  # ∂²u/∂t² (j=1 for time)
    
    # PDE residual (homogeneous wave equation)
    residual = u_tt - c**2 * u_xx + k * u_t
    return residual

# Apply output transform to enforce boundary conditions: u(0,t) = u(1,t) = 0
def output_transform(x, y):
    """
    Enforce boundary conditions through output transformation.
    
    Args:
        x: input coordinates (n_points, 2) 
            NOTE: Based on analytical solution u(), columns are [t, x] NOT [x, t]
        y: network output (n_points, 1)
    Returns:
        transformed output that satisfies BCs
    """
    x_coord = x[:, 0:1]  # Spatial coordinate is column 1, shape (n_points, 1)
    
    # Boundary transform: x*(1-x) vanishes at x=0 and x=1
    bc_transform = x_coord * (1.0 - x_coord)
    
    # Apply transform: u_transformed = u_net * x * (1-x)
    return y * bc_transform

def u0(x, a=a):
    """
    Initial condition function u(x,0) = sin(pi*x) + sin(a*pi*x)
    """
    x_space = x[:, 0:1] if x.ndim > 1 else x
    return np.sin(np.pi * x_space) + np.sin(a * np.pi * x_space)

def v0(x, a=a, c=c):
    """
    Initial velocity function v(x,0) = ∂u/∂t at t=0
    """
    x_space = x[:, 0:1] if x.ndim > 1 else x
    return np.zeros_like(x_space)

def u(ins, u0):
    """
    :param x: x = (t, x)
    """
    x = ins[:,0:1]
    t = ins[:,1:2]
    return 0.5*(u0(x - c * t) + u0(x + c * t)) #* np.exp(-k * t / 2)

if __name__ == "__main__":
    # NOTE: If you get AttributeError about 'B_temporal_sigma_1.0', 
    # Python has cached the old fft.py module. Restart your Python process.
    
    feature_transform = FourierFeatureTransform(
        input_dim=2,
        m_spatial=m_spatial,       
        m_temporal=m_temporal,      
        sigma_spatial=sigma_spatial,  # Moderate spatial frequencies
        sigma_temporal_list=sigma_temporal_list,  # Low and high temporal frequencies
        seed=seed
    )
    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # bc = dde.icbc.DirichletBC(geomtime, lambda _: 0, lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(geomtime, lambda x: u0(x, a), lambda _, on_initial: on_initial)
    ic_2 = dde.icbc.OperatorBC(
        geomtime,
        lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=1),
        lambda x, _: dde.utils.isclose(x[1], 0),
    )

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [ic, ic_2],
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
        num_test=num_test,
        solution=lambda ins: u(ins, u0)
    )

    # Network with Fourier features: input 4*m features → hidden layers → output
    layer_size = [2] + [hidden_feats] * hidden_layers + [1]
    activation = "tanh"
    initializer = "Glorot uniform"

    # Create FNN
    net = dde.nn.FNN(
        layer_size,
        activation,
        initializer,
    )

    # Apply feature transform before network
    net.apply_feature_transform(feature_transform)
    net.apply_output_transform(output_transform)

    # Note: We're NOT using apply_feature_transform because it causes dimension mismatch
    # during PDE evaluation. Instead, we'll use the standard trunk network and rely on
    # the network to learn appropriate features from the 2D (x,t) input.

    model = dde.Model(data, net)

    model.compile("adam", lr=lr_0,     
                # metrics=["l2 relative error"],
        loss_weights=[1.0, 1.0, 1.0],
        decay=("inverse time", epochs_decay, decay),)
    
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

    if iters>0:
        # Define checkpoints: 0%, 25%, 50%, 75%, 100% of training
        plot_checkpoints = [0, iters // 4, iters // 2, 3 * iters // 4, iters]
        print(f"Eigenvalue plots will be saved at epochs: {plot_checkpoints}")
        
        adaptive_weight_callback = AdaptiveWeightCallback(model, data, update_every=100, plot_checkpoints=plot_checkpoints)
        
        # Store initial matrices (epoch 0) in the callback
        adaptive_weight_callback.checkpoint_K_pde.append(K_pde_init)
        adaptive_weight_callback.checkpoint_K_ic1.append(K_ic1_init)
        adaptive_weight_callback.checkpoint_K_ic2.append(K_ic2_init)
        adaptive_weight_callback.checkpoint_epochs.append(0)

        start = datetime.now()
        losshistory, train_state = model.train(iterations=iters, 
                                            display_every=500, 
                                            model_save_path=f"{log_dir}/model.ckpt",
                                            callbacks=[adaptive_weight_callback])
        end = datetime.now()
        tr_time = end - start
        print(f"Training time: {tr_time}")
    else:
        tr_time = 0
    # dde.utils.plot_loss_history(losshistory)


    # Create 2D evaluation grid for visualization (100x100)
    N_x = 100
    N_t = 100
    x_vis = np.linspace(0, 1, num=N_x)
    t_vis = np.linspace(0, 1, num=N_t)
    x_grid, t_grid = np.meshgrid(x_vis, t_vis, indexing='ij')
    eval_pts_vis = np.hstack([x_grid.reshape(-1, 1), t_grid.reshape(-1, 1)])  # shape (10000, 2): [x, t]


    # For trunk input: use the same visualization grid
    x_trunk = eval_pts_vis

    # Predict solution u(t,x)
    u_pred = model.predict(x_trunk)
    u_pred = u_pred.reshape((N_t, N_x)).T

    # Compute true solution using WaveGNN1D solver (pass 1D arrays, not meshgrids)
    # solve() returns (u_xt, U_nt) where u_xt has shape (Nt, Nx)
    # L = build_laplacian_matrix(N_x, x_vis[1]-x_vis[0])
    # u_pred, _ = WaveGNN1D(L=L, c=c, k=k, dt=0.005).solve(x_vis, t_vis, u0, v0)

    u_true = u(eval_pts_vis, u0)
    u_true = u_true.reshape((N_t, N_x)).T

    l2_err = dde.metrics.l2_relative_error(u_true, u_pred)

    print(f"L2 relative error on visualization grid: {l2_err:.3e}")
    plot_comparison(u_true, u_pred, l2_err, log_dir)

    params = {
        'seed': seed,
        'c': c,
        'k': k,
        'num_domain': num_domain,
        'num_boundary': num_boundary,
        'num_initial': num_initial,
        'num_test': num_test,
        'hidden_feats': hidden_feats,
        'hidden_layers': hidden_layers,
        'm_spatial': m_spatial,
        'm_temporal': m_temporal,
        'sigma_spatial': sigma_spatial,  # Moderate spatial frequencies
        'sigma_temporal_0': sigma_temporal_list[0],
        'sigma_temporal_1': sigma_temporal_list[1],
        'iters': iters,
        'load_from': load_from,
        'l2_relative_error': float(l2_err),
        'tr_time': str(tr_time),
        'tr_time_seconds': tr_time.total_seconds() if tr_time != 0 else 0
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


