"""Backend supported: tensorflow.compat.v1, paddle
Implementation of the wave propagation example in paper https://arxiv.org/abs/2012.10047.
References:
    https://github.com/PredictiveIntelligenceLab/MultiscalePINNs.
"""
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Set Paddle as the backend
dde.config.set_default_float("float32")
dde.backend.set_default_backend("paddle")

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
    x, t = np.split(x, 2, axis=1)
    return np.sin(np.pi * x) * np.cos(C * np.pi * t) + np.sin(A * np.pi * x) * np.cos(
        A * C * np.pi * t
    )

# Compute NTK-based adaptive weights
def compute_ntk_weights(model, data):
    """
    Compute adaptive weights using NTK trace estimates per loss component (Paddle).
    For PDE: trace based on residual r = u_tt - C^2 u_xx using model.predict(operator=pde)
    For ICs: trace based on direct outputs u and u_t
    Returns weights for [pde_loss, ic1_loss, ic2_loss].
    """
    import paddle

    # Get training points (numpy) and split
    train_x = data.train_x_all
    num_domain = data.num_domain
    num_initial = data.num_initial

    pde_points_np = train_x[:num_domain]
    ic_points_np = train_x[num_domain:num_domain + num_initial]

    # Convert to Paddle tensors (float32, requires_grad)
    pde_points = paddle.to_tensor(pde_points_np, dtype='float32', stop_gradient=False)
    ic_points = paddle.to_tensor(ic_points_np, dtype='float32', stop_gradient=False)

    def ntk_trace_pde_residual(x_tensor):
        """Compute NTK trace for PDE residual using DeepXDE's operator interface"""
        params = [p for p in model.net.parameters()]
        total = paddle.to_tensor(0.0, dtype='float32')
        n = x_tensor.shape[0]
        
        for i in range(n):
            xi = x_tensor[i:i+1]
            xi.stop_gradient = False
            
            # Use DeepXDE's predict with operator to get residual
            # Note: predict expects numpy, but we need gradients, so we compute manually
            yi = model.net(xi)
            
            # Compute residual using the pde function
            residual = pde(xi, yi)
            r_scalar = residual.sum()
            
            # Compute gradients of residual w.r.t. parameters
            grads = paddle.grad(outputs=[r_scalar], inputs=params, create_graph=False, retain_graph=False, allow_unused=True)
            sq_terms = [paddle.sum(g * g) for g in grads if g is not None]
            if sq_terms:
                total = total + paddle.add_n(sq_terms)
                
        return float(total.numpy().item())

    def ntk_trace_for_output(x_tensor):
        """Compute NTK trace for direct network output (for ICs)"""
        params = [p for p in model.net.parameters()]
        total = paddle.to_tensor(0.0, dtype='float32')
        n = x_tensor.shape[0]
        for i in range(n):
            xi = x_tensor[i:i+1]
            yi = model.net(xi)
            y_scalar = yi.sum()
            grads = paddle.grad(outputs=[y_scalar], inputs=params, create_graph=False, retain_graph=False, allow_unused=True)
            sq_terms = [paddle.sum(g * g) for g in grads if g is not None]
            if sq_terms:
                total = total + paddle.add_n(sq_terms)
        return float(total.numpy().item())

    # NTK trace estimates
    K_pde = ntk_trace_pde_residual(pde_points)
    K_ic1 = ntk_trace_for_output(ic_points)
    K_ic2 = K_ic1  # Both ICs use same points

    print(f"  Raw NTK traces: K_pde={K_pde:.2e}, K_ic1={K_ic1:.2e}, K_ic2={K_ic2:.2e}")

    total_trace = K_pde + K_ic1 + K_ic2 + 1e-12

    # Adaptive weights (inverse proportional to each trace)
    lambda_pde = total_trace / (K_pde + 1e-12)
    lambda_ic1 = total_trace / (K_ic1 + 1e-12)
    lambda_ic2 = total_trace / (K_ic2 + 1e-12)

    # Normalize weights to prevent extreme values
    weights = np.array([lambda_pde, lambda_ic1, lambda_ic2], dtype=np.float32)
    weights = weights / np.mean(weights)  # Mean = 1
    
    # Optional: Cap maximum weight ratio to prevent one loss dominating
    max_weight = np.max(weights)
    if max_weight > 100:  # If any weight > 100x the mean
        print(f"  Warning: Large weight ratio detected ({max_weight:.1f}), clipping to 100x")
        weights = np.clip(weights, None, 100)
        weights = weights / np.mean(weights)  # Renormalize after clipping

    return weights

class AdaptiveWeightCallback(dde.callbacks.Callback):
    def __init__(self, model, data, update_every=1000):
        super().__init__()
        self.model = model
        self.data = data
        self.update_every = update_every
        
    def on_epoch_end(self):
        if self.model.train_state.epoch % self.update_every == 0:
            weights = compute_ntk_weights(self.model, self.data)
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

layer_size = [2] + [100] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.STMsFFN(
    layer_size, activation, initializer, sigmas_x=[1], sigmas_t=[1, 10]
)
# net.apply_feature_transform(lambda x: (x - 0.5) * 2 * np.sqrt(3))
net.apply_output_transform(output_transform)

model = dde.Model(data, net)
# Initialize with equal weights
    # (Removed TF-specific Jacobian; Paddle-based NTK trace computed directly above.)
model.compile(
    "adam",
    lr=0.001,
    metrics=["l2 relative error"],
    loss_weights=[1.0, 1.0, 1.0],
    decay=("inverse time", 2000, 0.9),
)

# Compute initial NTK-based weights
print("Computing initial NTK weights...")
initial_weights = compute_ntk_weights(model, data)
print(f"Initial adaptive weights: {initial_weights}")

# Recompile with adaptive weights
model.compile(
    "adam",
    lr=0.001,
    metrics=["l2 relative error"],
    loss_weights=initial_weights,
    decay=("inverse time", 2000, 0.9),
)

# pde_residual_resampler = dde.callbacks.PDEPointResampler(period=1)
adaptive_weight_callback = AdaptiveWeightCallback(model, data, update_every=1000)

losshistory, train_state = model.train(
    iterations=40000, 
    callbacks=[adaptive_weight_callback], 
    display_every=500, 
    model_save_path=f"{output_fold}/model_{timestamp}.ckpt"
)

dde.saveplot(losshistory, train_state, issave=True, isplot=True, output_dir=output_fold, 
             loss_fname=f"loss_{timestamp}", train_fname=f"train_{timestamp}", test_fname=f"test_{timestamp}")

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