import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np
import subprocess
from dataset import create_dataset, WaveGNN1D, build_laplacian_matrix, DeepGCN
from adaptive_weights import create_adaptive_weights_from_config
from datetime import datetime

def rk4_loss(interior_mask, input, output, laplacian, dt=0.01, c=1.0, k=1.0, w1=1.0, w2=1.0):
        """
        Node update function using RK4 time integration.
        
        Args:
            node_features: Current [u, v, f] for each node, shape (N, 3)
            laplacian: Discrete Laplacian, shape (N,)
            
        Returns:
            new_features: Updated [u, v, f] for each node, shape (N, 3)
        """
        # Extract current state
        u = input[:, 0]
        v = input[:, 1]
        force = input[:, 2]
        u_next = output[:, 0]
        v_next = output[:, 1]

        # Define the derivative function
        # du/dt = v
        # dv/dt = c²·Laplacian(u) - k·v + f
        
        # k1: derivatives at t
        du_dt_1 = v
        laplacian_1 = torch.Tensor(laplacian @ u)
        dv_dt_1 = c**2 * laplacian_1 - k * v + force
        
        # k2: derivatives at t + dt/2 using k1
        u_2 = u + 0.5 * dt * du_dt_1
        v_2 = v + 0.5 * dt * dv_dt_1
        du_dt_2 = v_2
        laplacian_2 = torch.Tensor(laplacian @ u_2)
        dv_dt_2 = c**2 * laplacian_2 - k * v_2 + force
        
        # k3: derivatives at t + dt/2 using k2
        u_3 = u + 0.5 * dt * du_dt_2
        v_3 = v + 0.5 * dt * dv_dt_2
        du_dt_3 = v_3
        laplacian_3 = torch.Tensor(laplacian @ u_3)
        dv_dt_3 = c**2 * laplacian_3 - k * v_3 + force
        
        # k4: derivatives at t + dt using k3
        u_4 = u + dt * du_dt_3
        v_4 = v + dt * dv_dt_3
        du_dt_4 = v_4
        laplacian_4 = torch.Tensor(laplacian @ u_4)
        dv_dt_4 = c**2 * laplacian_4 - k * v_4 + force
        
        # Weighted average (RK4 formula)
        u_new = u + (dt / 6.0) * (du_dt_1 + 2*du_dt_2 + 2*du_dt_3 + du_dt_4)
        v_new = v + (dt / 6.0) * (dv_dt_1 + 2*dv_dt_2 + 2*dv_dt_3 + dv_dt_4)
        
        
        loss_1 = ((u_new[interior_mask]-u_next[interior_mask]) ** 2).mean()
        loss_2 = ((v_new[interior_mask]-v_next[interior_mask]) ** 2).mean()
        rk4_loss = w1 * loss_1 + w2 * loss_2
        loss = rk4_loss

        return loss, float(loss_1.detach().item()), float(loss_2.detach().item())

def gn_loss(interior_mask, input, output, laplacian, gn_solver, dt=0.01, c=1.0, k=1.0, w1=1.0, w2=1.0):
    """Compute loss term based on gn solver prediction.
    """
    output_gn = gn_solver.forward(input)
    u_next = output[:, 0]
    v_next = output[:, 1]
    u_gn = output_gn[:, 0]
    v_gn = output_gn[:, 1]

    loss_1 = ((u_gn[interior_mask]-u_next[interior_mask]) ** 2).mean()
    loss_2 = ((v_gn[interior_mask]-v_next[interior_mask]) ** 2).mean()
    pde_loss = w1 * loss_1 + w2 * loss_2
    loss = pde_loss

    return loss, float(loss_1.detach().item()), float(loss_2.detach().item())

def physics_informed_loss(interior_mask, input, output, laplacian, dt=0.01, c=1.0, k=1.0, w1=1.0, w2=1.0):
    """Train step using physics-informed loss:

    Loss = MSE( L u - f ) on interior nodes
    For simplicity we set f=0 (homogeneous PDE) in this demo.
    """
    u = input[:, 0]
    v = input[:, 1]
    u_next = output[:, 0]
    v_next = output[:, 1]

    # PDE residual (L u) : shape (N, 2)
    Lu = torch.Tensor(laplacian @ u)
    f = input[:, 2]

    # PDE loss on interior nodes
    pde_res1 = v[interior_mask]*dt + u[interior_mask] - u_next[interior_mask]
    pde_res2 = v_next[interior_mask] - v[interior_mask] - dt*((c**2)*Lu[interior_mask] - k*v[interior_mask] + f[interior_mask])
    loss_1 = (pde_res1 ** 2).mean()
    loss_2 = (pde_res2 ** 2).mean()
    pde_loss = w1 * loss_1 + w2 * loss_2
    loss = pde_loss

    return loss, float(loss_1.detach().item()), float(loss_2.detach().item())

def train_physics(batch, model, optimizer, device, cfg, rk4=True, adaptive_weights=None):
    """Train on a batched Data object (several graphs concatenated by DataLoader).

    We iterate over graphs inside the batch, build per-graph Laplacian, compute PDE residual
    on interior nodes and enforce BCs as hard constraints by zeroing outputs at bc nodes
    (model.bc_mask should be set to batch.bc_mask before forward).
    
    Args:
        batch: Batched graph data
        model: Neural network model
        optimizer: Optimizer
        device: Device (CPU/GPU)
        cfg: Configuration object
        rk4: Whether to use RK4 loss
        adaptive_weights: AdaptiveLossWeights instance (optional)
    
    Returns:
        Tuple of (total_loss, loss_1_PI, loss_2_PI, loss_1_rk4, loss_2_rk4)
    """
    model.train()
    optimizer.zero_grad()

    batch = batch.to(device)

    # u = model(batch.x, batch.edge_index, batch.bc_mask)  # (N_total, 2)
    device = batch.x.device

    # For each graph in the batch, roll out predictions in time
    pde_loss_sum = torch.tensor(0.0, device=device)
    pde_loss_count = 0

    # Use cached per-graph Laplacians by iterating over the batch as a list of Data objects
    data_list = batch.to_data_list()
    for data in data_list:

        L = data.laplacian
        model_sub = model
        model_sub.bc_mask = data.bc_mask.to(device)
        
        interior_mask = ~data.bc_mask.to(device)
        # interior_mask = torch.ones_like(model_sub.bc_mask, dtype=torch.bool, device=device)
        
        out_sub = model_sub(data.x.to(device), data.edge_index.to(device), data.bc_mask.to(device))

        # Get weights (from adaptive weighting or config)
        if adaptive_weights is not None:
            w1_PI = adaptive_weights.get_weight('PI_loss1')
            w2_PI = adaptive_weights.get_weight('PI_loss2')
            w1_rk4 = adaptive_weights.get_weight('RK4_loss1')
            w2_rk4 = adaptive_weights.get_weight('RK4_loss2')
        else:
            w1_PI = cfg.training.loss.w1_PI
            w2_PI = cfg.training.loss.w2_PI
            w1_rk4 = cfg.training.loss.w1_rk4
            w2_rk4 = cfg.training.loss.w2_rk4

        loss_tensor, loss_1_val, loss_2_val = physics_informed_loss(
            interior_mask,
            data.x.to(device),
            out_sub,
            L,
            w1=w1_PI,
            w2=w2_PI,
        )
        if rk4:
            loss_rk4, loss_1_rk4, loss_2_rk4 = rk4_loss(
                interior_mask,
                data.x.to(device),
                out_sub,
                L,
            w1=w1_rk4,
            w2=w2_rk4,
            )
        else:
            loss_rk4 = torch.tensor(0.0, device=device)

        pde_loss_sum = pde_loss_sum + loss_rk4 + loss_tensor
        pde_loss_count += 1
        # accumulate scalar components for logging
        if 'loss1_sum' not in locals():
            loss1_sum = 0.0
            loss2_sum = 0.0
            loss1_rk4 = 0.0
            loss2_rk4 = 0.0
        loss1_sum += loss_1_val
        loss2_sum += loss_2_val
        loss1_rk4 += loss_1_rk4
        loss2_rk4 += loss_2_rk4

    # After iterating all graphs in the batch, compute mean PDE loss tensor
    if pde_loss_count > 0:
        pde_loss = pde_loss_sum / pde_loss_count
        avg_loss1 = loss1_sum / pde_loss_count
        avg_loss2 = loss2_sum / pde_loss_count
        avg_loss1_rk4 = loss1_rk4 / pde_loss_count
        avg_loss2_rk4 = loss2_rk4 / pde_loss_count

    else:
        pde_loss = torch.tensor(0.0, device=device)
        avg_loss1 = float('nan')
        avg_loss2 = float('nan')
        avg_loss1_rk4 = float('nan')
        avg_loss2_rk4 = float('nan')

    # Loss used for optimization (BCs are hard constraints)
    loss = pde_loss
    loss.backward()
    
    # Apply gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.training.get('grad_clip_norm', 1.0))
    
    optimizer.step()

    return float(loss.detach().item()), avg_loss1, avg_loss2, avg_loss1_rk4, avg_loss2_rk4


@torch.no_grad()
def evaluate_loader(loader, model, device):
    model.eval()
    total_loss = 0.0
    total_loss_1 = 0.0
    total_loss_2 = 0.0
    total_graphs = 0

    for batch in loader:
        data_list = batch.to_data_list()
        for data in data_list:
            data = data.to(device)
            L = data.laplacian

            model.bc_mask = data.bc_mask.to(device)
            preds = model(data.x.to(device), data.edge_index.to(device), data.bc_mask.to(device))

            interior_mask = ~data.bc_mask.to(device)
            # interior_mask = torch.ones_like(model.bc_mask, dtype=torch.bool, device=device)

            loss, loss_1, loss_2 = physics_informed_loss(
                interior_mask,
                data.x.to(device),
                preds,
                L
            )

            total_loss += float(loss.detach().item())
            total_loss_1 += loss_1
            total_loss_2 += loss_2
            total_graphs += 1

    avg_loss = total_loss / total_graphs if total_graphs > 0 else float('nan')
    avg_loss_1 = total_loss_1 / total_graphs if total_graphs > 0 else float('nan')
    avg_loss_2 = total_loss_2 / total_graphs if total_graphs > 0 else float('nan')
    return {"pde_mse": avg_loss, "loss_1": avg_loss_1, "loss_2": avg_loss_2}


def train_model(cfg, train_set, val_set, save_path="best_model.pt"):
    """
    Train a GCN model using the provided configuration and datasets.
    
    Args:
        cfg: Hydra configuration object
        train_set: Training dataset
        val_set: Validation dataset
        save_path: Path to save the best model checkpoint
        
    Returns:
        model: Trained model
        metrics: Dictionary containing training metrics
    """
    import logging
    from pathlib import Path
    log = logging.getLogger(__name__)
    
    # Setup device
    if cfg.experiment.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log.info(f"Using device: {device}")

    # Create data loaders
    train_loader = DataLoader(
        train_set, 
        batch_size=cfg.dataset.batch_size, 
        drop_last=cfg.dataset.drop_last, 
        shuffle=cfg.dataset.shuffle
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=cfg.dataset.batch_size, 
        drop_last=cfg.dataset.drop_last, 
        shuffle=False
    )


    # Build model
    sample = train_set[0]
    model = DeepGCN(
        in_channels=cfg.model.in_channels,
        hidden_channels=cfg.model.hidden_channels,
        out_channels=cfg.model.out_channels,
        conv_types=cfg.model.conv_types,
        final_layer_type=cfg.model.final_layer_type,
        activation=cfg.model.activation,
        dropout=cfg.model.dropout,
        block=cfg.model.block,
        use_bn=cfg.model.use_bn,
        residual=cfg.model.get('residual', False),  # Default to False for backward compatibility
        # Global pooling / encoder-decoder options (all optional for backward compatibility)
        use_global_pooling=cfg.model.get('use_global_pooling', False),
        pooling_position=cfg.model.get('pooling_position', 'end'),
        pooling_type=cfg.model.get('pooling_type', 'mean'),
        encoder_layers=cfg.model.get('encoder_layers', None),
        decoder_channels=cfg.model.get('decoder_channels', None),
        graph_output_dim=cfg.model.get('graph_output_dim', None),
        use_ed_skip=cfg.model.get('use_ed_skip', False),
        ed_skip_type=cfg.model.get('ed_skip_type', 'concat'),
    ).to(device)
    # Safety note: This trainer expects node-level outputs. If pooling at end is enabled,
    # the model will return graph-level outputs which will break the physics losses.
    if getattr(model, 'use_global_pooling', False) and getattr(model, 'pooling_position', 'end') == 'end':
        log.warning("Model is configured with pooling_position='end' (graph-level output). "
                    "This training loop expects node-level outputs. Consider setting "
                    "pooling_position='middle' or use_global_pooling=False.")
    
    log.info(f"Model architecture: {model}")
    if cfg.model.get('residual', False):
        log.info("✓ Residual mode enabled: Model predicts changes (Δu, Δv)")
    else:
        log.info("✓ Absolute mode: Model predicts absolute values (u, v)")
    
    # Scale learning rate based on dataset size to maintain consistent gradient updates
    base_lr = cfg.training.learning_rate
    if cfg.training.get('scale_lr_with_dataset', True):
        # Scale LR proportionally to sqrt(dataset_size / baseline)
        # This helps maintain stability when changing dataset size
        baseline_size = cfg.training.get('lr_baseline_size', 1000)
        dataset_size = len(train_set)
        lr_scale_factor = np.sqrt(dataset_size / baseline_size)
        scaled_lr = base_lr * lr_scale_factor
        log.info(f"Learning rate scaling enabled:")
        log.info(f"  - Base LR: {base_lr:.6f}")
        log.info(f"  - Dataset size: {dataset_size}")
        log.info(f"  - Baseline size: {baseline_size}")
        log.info(f"  - Scale factor: {lr_scale_factor:.4f}")
        log.info(f"  - Scaled LR: {scaled_lr:.6f}")
    else:
        scaled_lr = base_lr
        log.info(f"Learning rate: {scaled_lr:.6f} (scaling disabled)")
    
    # Setup optimizer
    if cfg.training.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=scaled_lr,  # Use scaled learning rate
            weight_decay=cfg.training.weight_decay
        )
    else:
        raise NotImplementedError(f"Optimizer {cfg.training.optimizer} not implemented")

    # Setup adaptive loss weights if enabled
    adaptive_weights = create_adaptive_weights_from_config(cfg)
    if adaptive_weights is not None:
        log.info("=" * 50)
        log.info("Adaptive loss weighting enabled")
        log.info(f"Strategy: {cfg.training.loss.adaptive.strategy}")
        log.info(f"Initial weights: {adaptive_weights.get_weights()}")
        log.info("=" * 50)

    # Setup GN solver if enabled
    gn_solver = None
    if cfg.training.loss.use_gn_solver:
        N = cfg.dataset.num_nodes
        dx = cfg.dataset.domain_length / (N - 1)
        L = build_laplacian_matrix(N, dx)
        gn_solver = WaveGNN1D(L, c=cfg.dataset.wave_speed, k=cfg.dataset.damping, dt=cfg.dataset.dt)
        log.info("GN solver enabled")

    epochs = cfg.training.epochs
    best_pde = float('inf')
    patience_counter = 0

    log.info(f"Starting training for {epochs} epochs...")
    start_time = datetime.now()
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_loss_1 = 0.0
        epoch_loss_2 = 0.0
        epoch_loss_1_rk4 = 0.0
        epoch_loss_2_rk4 = 0.0
        nbatches = 0
        
        for batch in train_loader:
            
            loss, loss_1, loss_2, loss_1_rk4, loss_2_rk4 = train_physics(
                batch, model, optimizer, device, cfg,
                rk4=cfg.training.loss.use_rk4,
                adaptive_weights=adaptive_weights
            )
            
            # Check for numerical instability during training
            if np.isnan(loss) or np.isinf(loss):
                log.error(f"Training diverged at epoch {epoch}, batch {nbatches}! Loss: {loss}")
                log.error("Consider: 1) Reducing learning rate, 2) Enabling gradient clipping, "
                         "3) Checking feature scaling, 4) Reducing model complexity")
                raise ValueError("Training diverged with NaN/Inf loss")
            
            epoch_loss += loss
            epoch_loss_1 += loss_1
            epoch_loss_2 += loss_2
            epoch_loss_1_rk4 += loss_1_rk4
            epoch_loss_2_rk4 += loss_2_rk4
            nbatches += 1

        # Compute average losses for this epoch
        avg_loss = epoch_loss / max(1, nbatches)
        avg_loss_1 = epoch_loss_1 / max(1, nbatches)
        avg_loss_2 = epoch_loss_2 / max(1, nbatches)
        avg_loss_1_rk4 = epoch_loss_1_rk4 / max(1, nbatches)
        avg_loss_2_rk4 = epoch_loss_2_rk4 / max(1, nbatches)

        # Update adaptive weights based on current loss values
        if adaptive_weights is not None:
            loss_dict = {
                'PI_loss1': avg_loss_1,
                'PI_loss2': avg_loss_2,
            }
            if cfg.training.loss.use_rk4:
                loss_dict['RK4_loss1'] = avg_loss_1_rk4
                loss_dict['RK4_loss2'] = avg_loss_2_rk4
            
            adaptive_weights.update(epoch, loss_dict)

        # Evaluate on validation set
        metrics = evaluate_loader(val_loader, model, device)
        avg_loss_2 = epoch_loss_2 / max(1, nbatches)
        avg_loss_1_rk4 = loss_1_rk4 / max(1, nbatches)
        avg_loss_2_rk4 = loss_2_rk4 / max(1, nbatches)

        # Save model when validation PDE MSE improves
        val_pde = metrics.get("pde_mse", float('nan'))
        if not (isinstance(val_pde, float) and (val_pde != val_pde)):  # check not NaN
            if val_pde < best_pde - (cfg.training.early_stopping.min_delta if cfg.training.early_stopping.enabled else 0):
                best_pde = val_pde
                patience_counter = 0
                ckpt = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_pde': val_pde,
                    'config': cfg,
                    'scaling_enabled': cfg.dataset.scaling.enabled,
                    # Save complete model architecture for easy loading
                    'model_config': {
                        'in_channels': cfg.model.in_channels,
                        'hidden_channels': cfg.model.hidden_channels,
                        'out_channels': cfg.model.out_channels,
                        'conv_types': cfg.model.conv_types,
                        'final_layer_type': cfg.model.final_layer_type,
                        'activation': cfg.model.activation,
                        'dropout': cfg.model.dropout,
                        'block': cfg.model.block,
                        'use_bn': cfg.model.use_bn,
                        'gat_heads': cfg.model.gat_heads,
                        'cheb_K': cfg.model.cheb_K,
                        'residual': cfg.model.get('residual', False),
                        'use_global_pooling': cfg.model.get('use_global_pooling', False),
                        'pooling_position': cfg.model.get('pooling_position', 'end'),
                        'pooling_type': cfg.model.get('pooling_type', 'mean'),
                        'encoder_layers': cfg.model.get('encoder_layers', None),
                        'decoder_channels': cfg.model.get('decoder_channels', None),
                        'graph_output_dim': cfg.model.get('graph_output_dim', None),
                    }
                }
                # Save adaptive weights state if enabled
                if adaptive_weights is not None:
                    ckpt['adaptive_weights_state'] = adaptive_weights.state_dict()
                
                torch.save(ckpt, save_path)
                # log.info(f"✓ New best model saved (epoch={epoch}, val PDE MSE={val_pde:.3e})")
            else:
                patience_counter += 1

        # Logging
        if epoch % cfg.training.log_interval == 0 or epoch == 1:
            # Build log message with current weights if adaptive
            log_msg = (
                f"Epoch {epoch:03d} | "
                f"Train {avg_loss:.3e} | PI1 {avg_loss_1:.3e} | PI2 {avg_loss_2:.3e} | "
                f"RK4_1 {avg_loss_1_rk4:.3e} | RK4_2 {avg_loss_2_rk4:.3e} | "
                f"Val {metrics['pde_mse']:.3e} | PI1 {metrics['loss_1']:.3e} | PI2 {metrics['loss_2']:.3e}"
            )
            if adaptive_weights is not None and epoch % (cfg.training.log_interval * 2) == 0:
                weights = adaptive_weights.get_weights()
                log_msg += f"\n        Weights: PI1={weights['PI_loss1']:.2e}, PI2={weights['PI_loss2']:.2e}"
                if 'RK4_loss1' in weights:
                    log_msg += f", RK4_1={weights['RK4_loss1']:.2e}, RK4_2={weights['RK4_loss2']:.2e}"
            
            log.info(log_msg)
        
        # Early stopping
        if cfg.training.early_stopping.enabled and patience_counter >= cfg.training.early_stopping.patience:
            log.info(f"Early stopping triggered at epoch {epoch}")
            break

    log.info(f"Training finished. Best validation PDE MSE: {best_pde:.6e}")
    end_time = datetime.now()
    log.info(f"Total training time: {end_time - start_time}")
    
    return model, {'best_val_pde': best_pde, 'final_epoch': epoch}


@torch.no_grad()
def evaluate_loader_with_scaling(loader, model, device):
    """Evaluate model with input/output scaling."""
    model.eval()
    total_loss = 0.0
    total_loss_1 = 0.0
    total_loss_2 = 0.0
    total_graphs = 0

    for batch in loader:
        # Apply input scaling
        batch_list = batch.to_data_list()
        for data in batch_list:
            data = data.to(device)
            L = data.laplacian

            # Scale input
            model.bc_mask = data.bc_mask.to(device)
            
            # Forward pass with scaled input
            preds = model(data.x.to(device), data.edge_index.to(device), data.bc_mask.to(device))
            
            # Optional: check for numerical issues
            if torch.isnan(preds).any() or torch.isinf(preds).any():
                # Skip this graph to avoid contaminating validation metrics
                continue

            interior_mask = ~data.bc_mask.to(device)

            loss, loss_1, loss_2 = physics_informed_loss(
                interior_mask,
                data.x.to(device),
                preds,
                L
            )

            total_loss += float(loss.detach().item())
            total_loss_1 += loss_1
            total_loss_2 += loss_2
            total_graphs += 1

    avg_loss = total_loss / total_graphs if total_graphs > 0 else float('nan')
    avg_loss_1 = total_loss_1 / total_graphs if total_graphs > 0 else float('nan')
    avg_loss_2 = total_loss_2 / total_graphs if total_graphs > 0 else float('nan')
    return {"pde_mse": avg_loss, "loss_1": avg_loss_1, "loss_2": avg_loss_2}


def main():
    """Legacy main function for backward compatibility."""
    # Create a dataset of many graphs and use graph-level batching
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = create_dataset(num_graphs=100)
    # simple split: train/val
    n_train = int(0.8 * len(dataset))
    train_set = dataset[:n_train]
    val_set = dataset[n_train:]

    train_loader = DataLoader(train_set, batch_size=8, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, drop_last=True, shuffle=False)

    # build model
    sample = dataset[0]
    model = SimpleGCN(in_channels=sample.x.size(1), hidden_channels=64, out_channels=2, layer_types='GCN', final_layer_type='Linear', activation='relu', dropout=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

    N = 100
    dx = 1.0 / (N - 1)
    L = build_laplacian_matrix(N, dx)
    gn_solver = WaveGNN1D(L)

    epochs = 50
    best_pde = float('inf')

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_loss_1 = 0.0
        epoch_loss_2 = 0.0
        nbatches = 0
        for batch in train_loader:
            loss, loss_1, loss_2 = train_physics(batch, model, optimizer, device, gn_solver, rk4=False)
            epoch_loss += loss
            epoch_loss_1 += loss_1
            epoch_loss_2 += loss_2
            nbatches += 1

        metrics = evaluate_loader(val_loader, model, device)

        avg_loss = epoch_loss / max(1, nbatches)
        avg_loss_1 = epoch_loss_1 / max(1, nbatches)
        avg_loss_2 = epoch_loss_2 / max(1, nbatches)

        # Save model when validation PDE MSE improves
        val_pde = metrics.get("pde_mse", float('nan'))
        if not (isinstance(val_pde, float) and (val_pde != val_pde)):  # check not NaN
            if val_pde < best_pde:
                best_pde = val_pde
                ckpt = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_pde': val_pde,
                }
                save_path = "best_model.pt"
                torch.save(ckpt, save_path)
                # print(f"Saved new best model to {save_path} (epoch={epoch}, val PDE MSE={val_pde:.3e})")

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train loss {avg_loss:.3e} | Loss_1 {avg_loss_1:.3e} | Loss_2 {avg_loss_2:.3e} | Val {metrics['pde_mse']:.3e} | Loss_1 {metrics['loss_1']:.3e} | Loss_2 {metrics['loss_2']:.3e}")

    print(f"Best validation PDE MSE seen: {best_pde:.6e}")


if __name__ == "__main__":
    # subprocess.run(["python", "test_dataset.py"])
    main()
    # subprocess.run(["python", "test_gcn.py"])
