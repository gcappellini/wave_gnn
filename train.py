import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import subprocess
from dataset import create_dataset, WaveGNN1D, build_laplacian_matrix, DeepGCN
from scaler import DataScaler
from datetime import datetime

# class SimpleGCN(nn.Module):
#     """
#     Configurable GCN/MLP hybrid.

#     Args:
#         in_channels (int): input feature dim
#         hidden_channels (int or list[int]): hidden dims for each hidden layer
#         out_channels (int): output dim
#         layer_types (list[str] or str): layer types for hidden layers ('Linear' or 'GCN').
#             If a single string is given it will be repeated for all hidden layers.
#         final_layer_type (str): type for final layer ('Linear' or 'GCN')
#         activation (str or callable): 'tanh' or 'relu' or a callable that maps tensor->tensor
#         dropout (float): dropout probability applied after each hidden layer (0 disables)
#     """
#     def __init__(
#         self,
#         in_channels,
#         hidden_channels,
#         out_channels,
#         layer_types="GCN",
#         final_layer_type="Linear",
#         activation="tanh",
#         dropout=0.0,
#     ):
#         super().__init__()

#         # Normalize hidden_channels to a list
#         if isinstance(hidden_channels, int):
#             hidden_sizes = [hidden_channels]
#         else:
#             hidden_sizes = list(hidden_channels)

#         n_hidden = len(hidden_sizes)

#         # Normalize layer_types to list of length n_hidden
#         if isinstance(layer_types, str):
#             layer_types = [layer_types] * n_hidden
#         if len(layer_types) != n_hidden:
#             raise ValueError("len(layer_types) must match number of hidden layers")

#         # Activation function
#         if callable(activation):
#             self.activation = activation
#         elif activation == "tanh":
#             self.activation = torch.tanh
#         elif activation == "relu":
#             self.activation = F.relu
#         else:
#             raise ValueError("activation must be 'tanh', 'relu', or a callable")

#         self.dropout = float(dropout)

#         # Build layers: hidden layers then final layer
#         layers = []
#         prev_dim = in_channels
#         for i, hid in enumerate(hidden_sizes):
#             ltype = layer_types[i].lower()
#             if ltype == "linear":
#                 layers.append(Linear(prev_dim, hid))
#             elif ltype == "gcn" or ltype == "gcnconv":
#                 layers.append(GCNConv(prev_dim, hid))
#             else:
#                 raise ValueError("Unsupported layer type: " + str(layer_types[i]))
#             prev_dim = hid

#         # final layer
#         ft = final_layer_type.lower()
#         if ft == "linear":
#             layers.append(Linear(prev_dim, out_channels))
#         elif ft == "gcn" or ft == "gcnconv":
#             layers.append(GCNConv(prev_dim, out_channels))
#         else:
#             raise ValueError("Unsupported final layer type: " + str(final_layer_type))

#         self.layers = nn.ModuleList(layers)

    # def forward(self, x, edge_index=None, bc_mask=None, ):
    #     """
    #     x: node features (N, F)
    #     edge_index: required for GCNConv layers
    #     bc_mask: boolean mask of boundary nodes to zero-out at the end
    #     """
    #     for i, layer in enumerate(self.layers):
    #         if isinstance(layer, GCNConv):
    #             if edge_index is None:
    #                 raise RuntimeError("edge_index is required for GCNConv layers")
    #             x = layer(x, edge_index)
    #         else:
    #             x = layer(x)

    #         # apply activation+dropout after every layer except the last
    #         if i < len(self.layers) - 1:
    #             x = self.activation(x)
    #             if self.dropout and self.training:
    #                 x = F.dropout(x, p=self.dropout, training=True)

    #     x = x.clone()
    #     x[bc_mask] = 0.0

    #     return x

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

def train_physics(batch, model, optimizer, device, cfg, rk4=True):
    """Train on a batched Data object (several graphs concatenated by DataLoader).

    We iterate over graphs inside the batch, build per-graph Laplacian, compute PDE residual
    on interior nodes and enforce BCs as hard constraints by zeroing outputs at bc nodes
    (model.bc_mask should be set to batch.bc_mask before forward).
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

        loss_tensor, loss_1_val, loss_2_val = physics_informed_loss(
            interior_mask,
            data.x.to(device),
            out_sub,
            L,
            w1=cfg.training.loss.w1_PI,
            w2=cfg.training.loss.w2_PI,
        )
        if rk4:
            loss_rk4, loss_1_rk4, loss_2_rk4 = rk4_loss(
                interior_mask,
                data.x.to(device),
                out_sub,
                L,
            w1=cfg.training.loss.w1_rk4,
            w2=cfg.training.loss.w2_rk4,
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
        scaler: DataScaler object (or None if scaling disabled)
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

    # Setup data scaler if enabled
    scaler = None
    if cfg.dataset.scaling.enabled:
        log.info("=" * 50)
        log.info("Setting up data scaling...")
        scaler = DataScaler(
            method=cfg.dataset.scaling.method,
            per_feature=cfg.dataset.scaling.per_feature,
            epsilon=cfg.dataset.scaling.epsilon
        )
        scaler.fit(train_loader, input_indices=None, output_indices=[0, 1])
        
        # Save scaler alongside model
        scaler_path = Path(save_path).parent / "scaler.pkl"
        scaler.save(scaler_path)
        log.info(f"Scaler saved to {scaler_path}")
        log.info("=" * 50)

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
    ).to(device)
    
    log.info(f"Model architecture: {model}")
    if cfg.model.get('residual', False):
        log.info("✓ Residual mode enabled: Model predicts changes (Δu, Δv)")
    else:
        log.info("✓ Absolute mode: Model predicts absolute values (u, v)")
    
    # Setup optimizer
    if cfg.training.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=cfg.training.learning_rate, 
            weight_decay=cfg.training.weight_decay
        )
    else:
        raise NotImplementedError(f"Optimizer {cfg.training.optimizer} not implemented")

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
        nbatches = 0
        
        for batch in train_loader:
            # Apply input scaling if enabled
            if scaler is not None:
                batch_list = batch.to_data_list()
                for data in batch_list:
                    data.x = scaler.transform_input(data.x)
                # Reconstruct batch
                from torch_geometric.data import Batch
                batch = Batch.from_data_list(batch_list)
            
            loss, loss_1, loss_2, loss_1_rk4, loss_2_rk4 = train_physics(
                batch, model, optimizer, device, cfg,
                rk4=cfg.training.loss.use_rk4
            )
            epoch_loss += loss
            epoch_loss_1 += loss_1
            epoch_loss_2 += loss_2
            nbatches += 1

        # Evaluate on validation set (with scaling if enabled)
        if scaler is not None:
            metrics = evaluate_loader_with_scaling(val_loader, model, device, scaler)
        else:
            metrics = evaluate_loader(val_loader, model, device)

        avg_loss = epoch_loss / max(1, nbatches)
        avg_loss_1 = epoch_loss_1 / max(1, nbatches)
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
                }
                torch.save(ckpt, save_path)
                # log.info(f"✓ New best model saved (epoch={epoch}, val PDE MSE={val_pde:.3e})")
            else:
                patience_counter += 1

        # Logging
        if epoch % cfg.training.log_interval == 0 or epoch == 1:
            log.info(
                f"Epoch {epoch:03d} | "
                f"Train {avg_loss:.3e} | PI1 {avg_loss_1:.3e} | PI2 {avg_loss_2:.3e} | RK4_1 {avg_loss_1_rk4:.3e} | RK4_2 {avg_loss_2_rk4:.3e} | "
                f"Val {metrics['pde_mse']:.3e} | PI1 {metrics['loss_1']:.3e} | PI2 {metrics['loss_2']:.3e}"
            )
        
        # Early stopping
        if cfg.training.early_stopping.enabled and patience_counter >= cfg.training.early_stopping.patience:
            log.info(f"Early stopping triggered at epoch {epoch}")
            break

    log.info(f"Training finished. Best validation PDE MSE: {best_pde:.6e}")
    end_time = datetime.now()
    log.info(f"Total training time: {end_time - start_time}")
    
    return model, {'best_val_pde': best_pde, 'final_epoch': epoch}, scaler


@torch.no_grad()
def evaluate_loader_with_scaling(loader, model, device, scaler):
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
            x_scaled = scaler.transform_input(data.x.to(device))
            model.bc_mask = data.bc_mask.to(device)
            
            # Forward pass with scaled input
            preds_scaled = model(x_scaled, data.edge_index.to(device), data.bc_mask.to(device))
            
            # Inverse scale predictions back to original space for loss computation
            preds = scaler.inverse_transform_output(preds_scaled)

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
