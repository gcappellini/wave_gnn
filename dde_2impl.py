"""Backend supported: tensorflow.compat.v1, paddle
Implementation of the wave propagation example in paper https://arxiv.org/abs/2012.10047.
References:
    https://github.com/PredictiveIntelligenceLab/MultiscalePINNs.
"""
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os
import paddle
from datetime import datetime

# # Set Paddle as the backend
# dde.config.set_default_float("float32")
# dde.backend.set_default_backend("paddle")

# Create directory if it doesn't exist
output_fold = 'pi-operator/free_evol'
os.makedirs(output_fold, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

A = 2
C = 10
dde.config.set_random_seed(35)
np.random.seed(35)

def pde(x, y):
    dy_tt = dde.grad.hessian(y, x, i=1, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_tt - C**2 * dy_xx

def output_transform(x, y):
    x_coord = x[:, 0:1]
    t_coord = x[:, 1:2]
    transform = x_coord * (1 - x_coord)
    return transform * y

def func(x):
    # Handle both numpy arrays and Paddle tensors
    if isinstance(x, paddle.Tensor):
        x_coord, t_coord = paddle.split(x, 2, axis=1)
        return paddle.sin(np.pi * x_coord) * paddle.cos(C * np.pi * t_coord) + \
               paddle.sin(A * np.pi * x_coord) * paddle.cos(A * C * np.pi * t_coord)
    else:
        x_coord, t_coord = np.split(x, 2, axis=1)
        return np.sin(np.pi * x_coord) * np.cos(C * np.pi * t_coord) + \
               np.sin(A * np.pi * x_coord) * np.cos(A * C * np.pi * t_coord)


def compute_ntk_weights(model, data):
    """
    Compute adaptive weights based on NTK theory using PaddlePaddle backend.
    Returns weights for [pde_loss, ic1_loss, ic2_loss] and traces
    
    FIXED: Keep tensors in computational graph instead of converting to numpy
    """
    
    # Get training points
    train_x = data.train_x_all
    num_domain = data.num_domain
    num_initial = data.num_initial
    
    pde_points = train_x[:num_domain]
    ic_points = train_x[num_domain:num_domain + num_initial]
    
    # Convert to paddle tensors
    pde_points_pd = paddle.to_tensor(pde_points, dtype='float32', stop_gradient=False)
    ic_points_pd = paddle.to_tensor(ic_points, dtype='float32', stop_gradient=False)
    
    # Get trainable parameters
    params = [p for p in model.net.parameters() if not p.stop_gradient]
    
    # Helper function to compute trace for PDE residual
    def compute_trace_pde(points):
        trace = 0.0
        
        for i in range(points.shape[0]):
            point_tensor = points[i:i+1]
            point_tensor.stop_gradient = False
            
            # Forward pass through network (keeps computational graph)
            y = model.net(point_tensor)
            
            # Compute PDE residual (keeps computational graph)
            residual = pde(point_tensor, y)
            residual_scalar = residual.sum()
            
            # Compute gradients w.r.t. parameters
            grads = paddle.grad(
                outputs=residual_scalar,
                inputs=params,
                create_graph=False,
                retain_graph=False,
                allow_unused=True
            )
            
            # Sum squared gradients
            for g in grads:
                if g is not None:
                    trace += paddle.sum(g ** 2).item()
        
        return trace
    
    # Helper function for IC1 (y - func(x))
    def compute_trace_ic1(points):
        trace = 0.0
        
        for i in range(points.shape[0]):
            point_tensor = points[i:i+1]
            point_tensor.stop_gradient = False
            
            # Network output
            y = model.net(point_tensor)
            
            # IC1: y - func(x)
            target = func(point_tensor)
            ic_residual = (y - target).sum()
            
            # Compute gradients
            grads = paddle.grad(
                outputs=ic_residual,
                inputs=params,
                create_graph=False,
                retain_graph=False,
                allow_unused=True
            )
            
            for g in grads:
                if g is not None:
                    trace += paddle.sum(g ** 2).item()
        
        return trace
    
    # Helper function for IC2 (du/dt at t=0)
    def compute_trace_ic2(points):
        trace = 0.0
        
        for i in range(points.shape[0]):
            point_tensor = points[i:i+1]
            point_tensor.stop_gradient = False
            
            # Network output
            y = model.net(point_tensor)
            
            # IC2: du/dt (using dde.grad which works with paddle)
            y_t = dde.grad.jacobian(y, point_tensor, i=0, j=1)
            y_t_scalar = y_t.sum()
            
            # Compute gradients
            grads = paddle.grad(
                outputs=y_t_scalar,
                inputs=params,
                create_graph=False,
                retain_graph=False,
                allow_unused=True
            )
            
            for g in grads:
                if g is not None:
                    trace += paddle.sum(g ** 2).item()
        
        return trace
    
    # Compute traces for each loss component
    print("  Computing K_pde...")
    K_pde = compute_trace_pde(pde_points_pd)
    
    print("  Computing K_ic1...")
    K_ic1 = compute_trace_ic1(ic_points_pd)
    
    print("  Computing K_ic2...")
    K_ic2 = compute_trace_ic2(ic_points_pd)
    
    # Total trace
    total_trace = K_pde + K_ic1 + K_ic2
    
    # Compute adaptive weights using Algorithm 1 formula
    lambda_pde = total_trace / (K_pde + 1e-10)
    lambda_ic1 = total_trace / (K_ic1 + 1e-10)
    lambda_ic2 = total_trace / (K_ic2 + 1e-10)
    
    print(f"  Traces: K_pde={K_pde:.2e}, K_ic1={K_ic1:.2e}, K_ic2={K_ic2:.2e}")
    print(f"  Weights: λ_pde={lambda_pde:.2f}, λ_ic1={lambda_ic1:.2f}, λ_ic2={lambda_ic2:.2f}")
    
    weights = np.array([lambda_pde, lambda_ic1, lambda_ic2])
    traces = np.array([K_pde, K_ic1, K_ic2])
    
    return weights, traces


class AdaptiveWeightCallback(dde.callbacks.Callback):
    def __init__(self, model, data, update_every=1000):
        super().__init__()
        self.model = model
        self.data = data
        self.update_every = update_every
        
        # Initialize logging lists
        self.epochs_log = []
        self.K_pde_log = []
        self.K_ic1_log = []
        self.K_ic2_log = []
        self.lambda_pde_log = []
        self.lambda_ic1_log = []
        self.lambda_ic2_log = []
        
    def on_epoch_end(self):
        if self.model.train_state.epoch % self.update_every == 0:
            weights, traces = compute_ntk_weights(self.model, self.data)
            
            # Log epoch, traces, and weights
            self.epochs_log.append(self.model.train_state.epoch)
            self.K_pde_log.append(traces[0])
            self.K_ic1_log.append(traces[1])
            self.K_ic2_log.append(traces[2])
            self.lambda_pde_log.append(weights[0])
            self.lambda_ic1_log.append(weights[1])
            self.lambda_ic2_log.append(weights[2])
            
            print(f"\nEpoch {self.model.train_state.epoch}: Updated weights = {weights}")
            
            # Update loss weights
            self.model.compile(
                "adam",
                lr=self.model.opt._learning_rate,
                metrics=["l2 relative error"],
                loss_weights=weights,
            )


geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

ic_1 = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)
ic_2 = dde.icbc.OperatorBC(
    geomtime,
    lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=1),
    lambda x, _: dde.utils.isclose(x[1], 0),
)

data = dde.data.TimePDE(
    geomtime,
    pde,
    [ic_1, ic_2],
    num_domain=360,
    num_boundary=0,
    num_initial=360,
    solution=func,
    num_test=10000,
)

layer_size = [2] + [200] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
optimizer = "adam"
learning_rate = 0.001
iterations = 5000

net = dde.nn.STMsFFN(
    layer_size, activation, initializer, sigmas_x=[1], sigmas_t=[1, 10]
)
net.apply_feature_transform(lambda x: (x - 0.5) * 2 * np.sqrt(3))
net.apply_output_transform(output_transform)

model = dde.Model(data, net)


# Create custom optimizer with gradient clipping
custom_optimizer = paddle.optimizer.Adam(
    learning_rate=learning_rate,
    parameters=net.parameters(),
    # grad_clip=grad_clip
)

model.compile(
    custom_optimizer,
    metrics=["l2 relative error"],
    loss_weights=[1.0, 1.0, 1.0],
)

# Compute initial NTK-based weights
print("Computing initial NTK weights...")
initial_weights, initial_traces = compute_ntk_weights(model, data)
print(f"Initial adaptive weights: {initial_weights}")
print(f"Initial traces: {initial_traces}")

# Recompile with adaptive weights and gradient clipping
custom_optimizer = paddle.optimizer.Adam(
    learning_rate=learning_rate,
    parameters=net.parameters(),
    # grad_clip=grad_clip
)

model.compile(
    custom_optimizer,
    metrics=["l2 relative error"],
    loss_weights=[1.0, 1.0, 1.0], #initial_weights,
    decay=("inverse time", 2000, 0.9),
)

# pde_residual_resampler = dde.callbacks.PDEPointResampler(period=1)
adaptive_weight_callback = AdaptiveWeightCallback(model, data, update_every=100)

losshistory, train_state = model.train(
    iterations=iterations,
    callbacks=[adaptive_weight_callback],
    display_every=500,
    model_save_path=f"{output_fold}/model_{timestamp}.ckpt"
)

dde.utils.save_best_state(train_state, fname_train=f"train_{timestamp}", fname_test=f"test_{timestamp}")

# Plot all loss components separately
loss_train = np.array(losshistory.loss_train)
loss_test = np.array(losshistory.loss_test)
steps = np.array(losshistory.steps)

fig = plt.figure(figsize=(10, 6))
plt.semilogy(steps, loss_train[:, 0], label="PDE residual")
plt.semilogy(steps, loss_train[:, 1], label="IC1")
plt.semilogy(steps, loss_train[:, 2], label="IC2")
plt.semilogy(steps, np.sum(loss_train, axis=1), label="Total loss", linestyle='--', linewidth=2)
plt.semilogy(steps, np.sum(loss_test, axis=1), label="Total loss (test)", linestyle='--', linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("Loss Components Evolution")
plt.savefig(f'{output_fold}/loss_components_{timestamp}.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot NTK traces (K_pde, K_ic1, K_ic2) evolution
if len(adaptive_weight_callback.epochs_log) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs_array = np.array(adaptive_weight_callback.epochs_log)
    
    # K_pde
    axes[0].plot(epochs_array, adaptive_weight_callback.K_pde_log, 'o-', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('K_pde (NTK Trace)')
    axes[0].set_title('PDE Residual NTK Trace')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # K_ic1
    axes[1].plot(epochs_array, adaptive_weight_callback.K_ic1_log, 'o-', linewidth=2, markersize=6, color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('K_ic1 (NTK Trace)')
    axes[1].set_title('IC1 (u) NTK Trace')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    # K_ic2
    axes[2].plot(epochs_array, adaptive_weight_callback.K_ic2_log, 'o-', linewidth=2, markersize=6, color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('K_ic2 (NTK Trace)')
    axes[2].set_title('IC2 (u_t) NTK Trace')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{output_fold}/ntk_traces_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot adaptive weights evolution (all in one plot)
    fig = plt.figure(figsize=(10, 6))
    plt.plot(epochs_array, adaptive_weight_callback.lambda_pde_log, 'o-', label='λ_pde', linewidth=2, markersize=6)
    plt.plot(epochs_array, adaptive_weight_callback.lambda_ic1_log, 'o-', label='λ_ic1', linewidth=2, markersize=6)
    plt.plot(epochs_array, adaptive_weight_callback.lambda_ic2_log, 'o-', label='λ_ic2', linewidth=2, markersize=6)
    plt.xlabel('Epoch')
    plt.ylabel('Adaptive Weight Value')
    plt.title('Adaptive Loss Weights Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_fold}/adaptive_weights_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nNTK traces and weights plots saved to {output_fold}/")

x = np.linspace(0, 1, 100)[:, None]
t = np.linspace(0, 1, 100)[:, None]
X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

u_pred = model.predict(X_star).reshape(100, 100) 
u_true = func(X_star).reshape(100, 100) 

# Create a 3-subplot figure: predicted, ground truth, absolute error
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Predicted solution 
im0 = axes[0].imshow(u_pred, extent=[0, 1, t[0], t[-1]], aspect='auto', origin='lower')
axes[0].set_xlabel('x')
axes[0].set_ylabel('t')
axes[0].set_title('Predicted')
plt.colorbar(im0, ax=axes[0])

# Ground truth solution
im1 = axes[1].imshow(u_true, extent=[0, 1, t[0], t[-1]], aspect='auto', origin='lower')
axes[1].set_xlabel('x')
axes[1].set_ylabel('t')
axes[1].set_title('Ground Truth')
plt.colorbar(im1, ax=axes[1])

# Absolute error
abs_error = np.abs(u_pred - u_true)
rel_error = np.abs(u_pred - u_true) / (np.abs(u_true) + 1e-10)
im2 = axes[2].imshow(abs_error, extent=[0, 1, t[0], t[-1]], aspect='auto', origin='lower')
axes[2].set_xlabel('x')
axes[2].set_ylabel('t')
axes[2].set_title(f'Absolute Error (L2: {np.linalg.norm(abs_error)/np.linalg.norm(u_true):.4f})')
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig(f'{output_fold}/comparison_{timestamp}.png', dpi=150)
plt.close()