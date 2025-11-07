"""Backend supported: tensorflow.compat.v1, paddle

Implementation of the wave propagation example in paper https://arxiv.org/abs/2012.10047.
References:
    https://github.com/PredictiveIntelligenceLab/MultiscalePINNs.
"""
import os
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# Create directory if it doesn't exist
output_fold = 'pi-operator/free_evol'
os.makedirs(output_fold, exist_ok=True)

# Physical parameters
c = 1.0         # wave speed
k = 1.0         # damping coefficient
A = 0

# Scaling parameters
L = 1.0         # spatial domain length
T_max = 1.0     # time domain length
u_max = 0.04    # maximum displacement (output scale)
f_max = 3.0     # maximum forcing (input scale)

# Derived non-dimensional parameters for scaled PDE
c_star = c * T_max / L      # scaled wave speed
f_star = f_max * T_max**2 / u_max    # scaled forcing coefficient
k_star = k * T_max                    # scaled damping coefficient

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def get_initial_loss(model):
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(0)
    return losshistory.loss_train[0]

def output_transform(x, y):
    x_coord = x[:, 0:1]
    t_coord = x[:, 1:2]
    transform = x_coord * (1 - x_coord) * t_coord**2
    return transform * y

def pde(x, y):
    # dy_t = dde.grad.jacobian(y, x, j=1) 
    dy_tt = dde.grad.hessian(y, x, i=1, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_tt - c_star**2 * dy_xx


def ic_func(x):
    x_coord = x[:, 0:1]
    return np.sin(np.pi * x_coord) #+ np.sin(A * np.pi * x) 

def sol(x):
    x_coord = x[:, 0:1]
    t_coord = x[:, 1:2]
    return 0.5*(ic_func(x_coord-c_star*t_coord)+ic_func(x_coord+c_star*t_coord))#*np.exp(-k_star*t)

geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# bc = dde.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
ic_1 = dde.icbc.IC(geomtime, ic_func, lambda _, on_initial: on_initial)
# do not use dde.NeumannBC here, since `normal_derivative` does not work with temporal coordinate.
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
    num_boundary=360,
    num_initial=360,
    solution=sol,
    num_test=10000,
)

layer_size = [2] + [100] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.STMsFFN(
    layer_size, activation, initializer, sigmas_x=[1], sigmas_t=[1, 10]
)
# net = dde.nn.FNN(layer_size, activation, initializer) 

net.apply_output_transform(output_transform)

model = dde.Model(data, net)
# initial_losses = get_initial_loss(model)
# loss_weights = 5 / initial_losses
model.compile(
    "adam",
    lr=0.001,
    metrics=["l2 relative error"],
    # loss_weights=loss_weights,
    decay=("inverse time", 2000, 0.9),
)
pde_residual_resampler = dde.callbacks.PDEPointResampler(period=1)
losshistory, train_state = model.train(
    iterations=5000, callbacks=[pde_residual_resampler], display_every=500, model_save_path=f"{output_fold}/model_{timestamp}.ckpt"
)

dde.saveplot(losshistory, train_state, issave=True, isplot=True, output_dir=output_fold, 
             loss_fname=f"loss_{timestamp}", train_fname=f"train_{timestamp}", test_fname=f"test_{timestamp}")

dde.utils.plot_loss_history(losshistory, fname=f'{output_fold}/loss_history_{timestamp}.png')

x = np.linspace(0, 1, 100)[:, None]
t = np.linspace(0, 1, 100)[:, None]
X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_pred = model.predict(X_star).reshape(100, 100) * u_max
u_true = sol(X_star).reshape(100, 100) * u_max

# Create a 3-subplot figure: predicted, ground truth, absolute error
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Predicted solution (physical units)
im0 = axes[0].imshow(u_pred, extent=[0, 1, t[0], t[-1]], aspect='auto', origin='lower')
axes[0].set_xlabel('x')
axes[0].set_ylabel('t')
axes[0].set_title('Predicted (Physical Units)')
plt.colorbar(im0, ax=axes[0])

# Ground truth solution (physical units)
im1 = axes[1].imshow(u_true, extent=[0, 1, t[0], t[-1]], aspect='auto', origin='lower')
axes[1].set_xlabel('x')
axes[1].set_ylabel('t')
axes[1].set_title('Ground Truth (Physical Units)')
plt.colorbar(im1, ax=axes[1])

# Absolute error (physical units)
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