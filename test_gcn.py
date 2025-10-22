import numpy as np
from datetime import datetime
from scipy.io import savemat
import torch
from dataset import create_graph, WaveGNN1D
from plot import plot_features_2d
from omegaconf import OmegaConf
from scaler import DataScaler
from pathlib import Path



def load_best_model(path='./best_model.pt', device=None, model_cls=None, model_kwargs=None):
    """Load the best saved checkpoint and return a model instance + checkpoint dict.

    Args:
        path: path to checkpoint (default 'best_model.pt')
        device: torch device or None to auto-select
        model_cls: optional model class to instantiate; if None, will import DeepGCN from dataset
        model_kwargs: kwargs to pass to model constructor (optional, will use saved config if available)

    Returns:
        model (torch.nn.Module), checkpoint (dict)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(path, map_location=device)
    # ckpt = torch.load(path, map_location=device, weights_only=True)

    if model_cls is None:
        try:
            from dataset import DeepGCN as DefaultModel
            model_cls = DefaultModel
        except Exception:
            raise ImportError("Could not import DeepGCN from dataset. Please pass model_cls or ensure dataset.DeepGCN is importable.")

    # Use saved model_config if available (from newer checkpoints), otherwise fall back to model_kwargs
    if model_kwargs is None:
        if 'model_config' in ckpt:
            model_kwargs = ckpt['model_config']
        else:
            # Fallback for old checkpoints without model_config
            model_kwargs = {'in_channels': 3, 'hidden_channels': 128, 'out_channels': 2, 'dropout': 0.5}
            print("Warning: Checkpoint doesn't contain 'model_config'. Using default architecture.")
            print("If loading fails, please provide model_kwargs manually or retrain with updated train.py")

    model = model_cls(**model_kwargs).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    return model, ckpt


def stringforce(coords, t, t_f, f_min=-3, x_f_1=0.2, x_f_2=0.8):
    """
    2D membrane forcing - Gaussian pulses in space and time.
    
    Parameters
    ----------
    coords : np.ndarray or torch.Tensor
        Spatial coordinates. If torch.Tensor, will be converted to numpy.
    t : float
        Time value.
    t_f : float
        Final time constant.
    f_min : float
        Minimum force amplitude.
    x_f_1, y_f_1 : float
        First force center position.
    x_f_2, y_f_2 : float
        Second force center position.
    
    Returns
    -------
    f : np.ndarray
        Computed 2D force field values.
    """
    # Convert torch tensor to numpy if needed
    if hasattr(coords, 'detach'):
        X = coords.detach().cpu().numpy()
    else:
        X = np.asarray(coords)
    
    t_1 = 0.2 * t_f
    t_2 = 0.8 * t_f
    h = f_min

    # First pulse at (x_f_1, y_f_1)
    z1 = h * np.exp(-400 * ((X - x_f_1)**2)) * \
         np.exp(-((t - t_1)**2) / (2 * 0.5**2))
    
    # Second pulse at (x_f_2, y_f_2)
    z2 = h * np.exp(-400 * ((X - x_f_2)**2)) * \
         np.exp(-((t - t_2)**2) / (2 * 0.5**2))

    f = z1 + z2
    # Use standard numpy squeeze to avoid array_wrap issues
    if f.ndim > 1 and f.shape[-1] == 1:
        f = f.squeeze(-1)
    return f


def simulate_wave(gnn, data, t_f, dt=0.01, gt=True, scaler=None, device=None):
    """
    Simulate 2D wave equation using GNN with integrated time stepping.
    
    Args:
        gnn: WaveGNN2D instance (contains dt)
        data: Initial graph data
        t_f: Final simulation time
        dt: Time step
        gt: Whether to compute ground truth solution
        scaler: DataScaler for input/output scaling (optional)
        device: torch.device to use for computation (default: None, auto-detect from gnn)
        
    Returns:
        t_history: Time points
        u_history: Position history, shape (num_steps, Nx, Ny)
        v_history: Velocity history, shape (num_steps, Nx, Ny)
        f_history: Force history, shape (num_steps, Nx, Ny)
        u_gt: Ground truth positions (if gt=True)
        v_gt: Ground truth velocities (if gt=True)
    """
    # Detect device from model if not provided
    if device is None:
        device = next(gnn.parameters()).device
    
    # Move data to device
    data = data.to(device)
    if hasattr(data, 'laplacian') and torch.is_tensor(data.laplacian):
        data.laplacian = data.laplacian.to(device)
    
    # Initialize node features
    t = 0.0
    
    # Storage
    num_steps = int(t_f / dt) + 1
    t_history = np.zeros(num_steps)
    u_history = np.zeros((num_steps, data.x[:, 0].shape[0]))
    v_history = np.zeros((num_steps, data.x[:, 0].shape[0]))
    f_history = np.zeros((num_steps, data.x[:, 0].shape[0]))
    if gt:
        u_gt = np.zeros((num_steps, data.x[:, 0].shape[0]))
        v_gt = np.zeros((num_steps, data.x[:, 0].shape[0]))
        gn = WaveGNN1D(data.laplacian)
    
    # Store initial condition
    t_history[0] = t
    u_history[0] = data.x[:, 0].cpu().numpy()
    v_history[0] = data.x[:, 1].cpu().numpy()
    f_history[0] = data.x[:, 2].cpu().numpy()
    if gt:
        u_gt[0] = data.x[:, 0].cpu().numpy()
        v_gt[0] = data.x[:, 1].cpu().numpy()

    features = data.x
    if gt:
        features_gt = data.x.clone()
    
    # Time integration loop
    for step in range(1, num_steps):
        # Apply scaling if provided
        if scaler is not None:
            # Scale directly using tensor API; keeps device and avoids explicit numpy round-trip
            features_scaled = scaler.transform_input(features)
            preds = gnn(features_scaled, data.edge_index, data.bc_mask)

            # Check for numerical instability
            preds_np = preds.detach().cpu().numpy()
            if np.isnan(preds_np).any() or np.isinf(preds_np).any():
                print(f"WARNING: Numerical instability at step {step}!")
                print(f"  NaN count: {np.isnan(preds_np).sum()}")
                print(f"  Inf count: {np.isinf(preds_np).sum()}")
                print(f"  Features range: [{preds_np.min():.3e}, {preds_np.max():.3e}]")
                break

            # Keep the force component from input (not predicted)
            # features = torch.stack([preds[:, 0], preds[:, 1], features_scaled[:, 2]], dim=1)
        else:
            features = gnn(features, data.edge_index, data.bc_mask)
            # Keep the force component from input
            # features = torch.stack([features[:, 0], features[:, 1], features[:, 2]], dim=1)
        
        if gt:
            features_gt = gn.forward(features_gt)
        
        # Store
        t += dt
        t_history[step] = t

        u_history[step] = features[:, 0].detach().cpu().numpy()
        v_history[step] = features[:, 1].detach().cpu().numpy()
        f_history[step] = stringforce(data.coords, t, t_f)
        if gt:
            u_gt[step] = features_gt[:, 0].detach().cpu().numpy()
            v_gt[step] = features_gt[:, 1].detach().cpu().numpy()

        # Update force for next iteration (move to device)
        f_tensor = torch.tensor(f_history[step].flatten(), dtype=torch.float32, device=device)
        features = torch.stack([features[:, 0], features[:, 1], f_tensor], dim=1)
        if gt:
            features_gt = torch.stack([features_gt[:, 0], features_gt[:, 1], f_tensor], dim=1)

    if gt:
        return t_history, u_history, v_history, f_history, u_gt, v_gt
    else:
        return t_history, u_history, v_history, f_history, None, None


def test_model(cfg, model_path, output_dir, scaler_path=None):
    """
    Test the trained model and generate visualizations.
    
    Args:
        cfg: Hydra configuration object
        model_path: Path to the saved model checkpoint
        output_dir: Directory to save test outputs
        scaler_path: Path to the saved scaler (optional)
        
    Returns:
        test_results: Dictionary containing test metrics and output paths
        test_metrics: Dictionary containing numerical metrics
    """
    import logging
    
    log = logging.getLogger(__name__)
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Load scaler if provided
    scaler = None
    if scaler_path is not None and Path(scaler_path).exists():
        scaler = DataScaler.load(scaler_path)
        log.info(f"✓ Scaler loaded from {scaler_path}")
    else:
        log.info("No scaler used (scaling was disabled during training)")
    
    # Simulation parameters
    T = 10.0  # total time
    
    # Setup device
    if cfg.experiment.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Load trained GCN model (uses saved model_config from checkpoint)
    log.info(f"Loading model from {model_path}")
    gcn, ckpt = load_best_model(
        path=model_path,
        device=device
        # model_kwargs automatically loaded from checkpoint's model_config
    )
    
    if cfg.model.get('residual', False):
        log.info("✓ Model in residual mode: Predicts changes (Δu, Δv)")
    
    initial_graph = create_graph(zeros=True)
    nodes, elements = initial_graph.nodes, initial_graph.elements

    # Run simulation
    log.info("Running simulation...")
    start_time = datetime.now()
    t_history, u_history, v_history, f_history, u_gt, v_gt = simulate_wave(
        gcn, initial_graph, T, dt=cfg.dataset.dt, gt=True, scaler=scaler, device=device
    )
    end_time = datetime.now()
    elapsed = end_time - start_time
    log.info(f"Simulation complete: {len(t_history)} time steps in {elapsed}")
    log.info(f"Each step = one GNN forward pass")

    # Save results to .mat file
    matlab_dir = output_dir / "matlab"
    matlab_dir.mkdir(exist_ok=True)
    matlab_file = matlab_dir / "data_gcn.mat"
    
    matlab_times = np.linspace(0, T, num=100)
    indices = [np.argmin(np.abs(t_history - t)) for t in matlab_times]
    pinn_data = u_history[indices]

    savemat(str(matlab_file), {'pinn_data': pinn_data})
    log.info(f"MATLAB data saved to {matlab_file}")

    # Generate plots
    histories = np.array([u_history, v_history, f_history, u_gt, v_gt, np.abs(u_history - u_gt)])
    plot_file = figures_dir / "gcn_string_pred.png"
    
    plot_features_2d(
        nodes,  
        histories,
        output_file=str(plot_file),
        feature_names=['Deformation', 'Velocity', 'Force', 'GT Deformation', 'GT Velocity', 'Deformation Error']
    )

    log.info(f"Plot saved to {plot_file}")
    
    test_results = {
        'num_timesteps': len(t_history),
        'simulation_time_seconds': elapsed.total_seconds(),
        'plot_path': str(plot_file),
        'matlab_path': str(matlab_file)
    }

    test_metrics = {
        'u_mae': float(np.mean(np.abs(u_history - u_gt))),
        'v_mae': float(np.mean(np.abs(v_history - v_gt))),
    }
    
    return test_results, test_metrics


if __name__ == "__main__":

    T = 10.0       # total time

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load trained GCN model (expects best_model.pt in the working directory)
    gcn, ckpt = load_best_model(device=device)
    initial_graph = create_graph(zeros=True)
    nodes, elements = initial_graph.nodes, initial_graph.elements

    # Run simulation
    print("Running simulation...")
    start_time = datetime.now()
    t_history, u_history, v_history, f_history, u_gt, v_gt = simulate_wave(
        gcn, initial_graph, T, gt=True, device=device
    )
    end_time = datetime.now()
    print(f"Simulation complete: {len(t_history)} time steps in {end_time - start_time}")
    print(f"Each step = one GNN forward pass")
    print()

    # Save results to .mat file
    matlab_times = np.linspace(0, T, num=100)
    indices = [np.argmin(np.abs(t_history - t)) for t in matlab_times]
    pinn_data = u_history[indices]
    pinn_data = pinn_data

    # pinn_data = pinn_data.swapaxes(1, 2)  # Transpose to match MATLAB's column-major order
    savemat('/Users/guglielmocappellini/Desktop/research/code/pinns-wave/wave-gnn/1_gcn_string/matlab/data_gcn.mat', {
        'pinn_data': pinn_data
    })

    histories = np.array([u_history, v_history, f_history])

    plot_features_2d(
    nodes,  
    histories,
    output_file='./figures/gcn_string_pred.png'
    )
