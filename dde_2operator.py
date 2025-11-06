"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle
original code: https://github.com/lululxvi/deepxde/blob/master/examples/operator/diff_rec_aligned_pideeponet.py
"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator
import os
import torch

if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    # PyTorch will use GPU automatically if available
    device = torch.device("cuda:0")
else:
    print("No GPU available, using CPU")
    device = torch.device("cpu")


# Create directory if it doesn't exist
os.makedirs('pi-operator', exist_ok=True)

c = 1.0
k = 1.0

dde.config.set_random_seed(35)
np.random.seed(35)

class SpatioTemporalGRF:
    """Function space for v(x,t) - spatiotemporal forcing"""
    def __init__(self, length_scale_x=0.6, length_scale_t=0.2, f_scale=3.0):
        self.grf_x = dde.data.GRF(length_scale=length_scale_x)  # Spatial variation
        self.grf_t = dde.data.GRF(length_scale=length_scale_t)  # Temporal variation
        self.f_scale = f_scale
    
    def random(self, n):
        # Return features for both spatial and temporal components
        return (self.grf_x.random(n), self.grf_t.random(n))
    
    def eval_batch(self, features, sensors):
        """
        features: tuple of (spatial_features, temporal_features)
        sensors: array of shape (n_points, 2) with columns [x, t]
        """
        x_features, t_features = features
        x_coords = sensors[:, 0:1]  # spatial coordinates
        t_coords = sensors[:, 1:2]  # temporal coordinates
        
        # Evaluate spatial and temporal components separately
        v_x = self.grf_x.eval_batch(x_features, x_coords)  # shape: (n_batch, n_points)
        v_t = self.grf_t.eval_batch(t_features, t_coords)  # shape: (n_batch, n_points)
        
        # Combine: v(x,t) = v_x(x) * v_t(t) (multiplicative) or v_x(x) + v_t(t) (additive)
        # Using multiplicative for smooth spatiotemporal variation
        return self.f_scale * v_x * v_t

class ScaledGRF:
    """GRF wrapper with controllable f_scale and offset"""
    def __init__(self, length_scale, f_scale=3.0, offset=0.0):
        self.grf = dde.data.GRF(length_scale=length_scale)
        self.f_scale = f_scale
        self.offset = offset
    
    def random(self, n):
        return self.grf.random(n)
    
    def eval_batch(self, features, sensors):
        values = self.grf.eval_batch(features, sensors)
        return self.f_scale * values + self.offset
    
# PDE
def pde(x, y, v):
    dy_t = dde.grad.jacobian(y, x, j=1)
    dy_tt = dde.grad.hessian(y, x, j=1)
    dy_xx = dde.grad.hessian(y, x, j=0)
    return dy_tt - c**2 * dy_xx - v + k * dy_t 

def output_transform(x, y):
    """
    Transform network output to satisfy:
    - BC: u(0,t) = u(1,t) = 0 (spatial boundaries)
    - IC: u(x,0) = 0 (initial condition)
    
    For DeepONet:
    x is a tuple: (branch_input, trunk_input)
    x[0]: branch input (function evaluations) - shape (n_functions, 50)
    x[1]: trunk input (x, t coordinates) - shape (n_points, 2)
    y: raw network output - shape (n_functions, n_points)
    """
    # Extract trunk input (spatiotemporal coordinates)
    trunk_input = x[1]  # shape: (n_points, 2)
    x_coord = trunk_input[:, 0:1]  # spatial coordinate, shape (n_points, 1)
    t_coord = trunk_input[:, 1:2]  # time coordinate, shape (n_points, 1)
    
    # Compute transform: shape (n_points, 1)
    transform = x_coord * (1 - x_coord) * t_coord
    
    # Transpose to shape (1, n_points) and broadcast multiply with y (n_functions, n_points)
    # This applies the same spatial/temporal constraint to all functions
    return y * transform.T

geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# BC and IC are enforced as hard constraints via output_transform
# So we don't need to include them in the PDE (optional - you can keep them for extra enforcement)
# bc = dde.icbc.DirichletBC(geomtime, lambda _: 0, lambda _, on_boundary: on_boundary)
# ic = dde.icbc.IC(geomtime, lambda _: 0, lambda _, on_initial: on_initial)

pde = dde.data.TimePDE(
    geomtime,
    pde,
    [],  # No soft constraints needed - enforced via transform
    num_domain=200,
    num_boundary=40,
    num_initial=20,
    num_test=500,
)

# Function space - using spatiotemporal forcing v(x,t)
func_space = SpatioTemporalGRF(length_scale_x=0.6, length_scale_t=0.3, f_scale=3.0)

# Data
# Sensor points now need to cover both x and t
n_sensors_x = 20
n_sensors_t = 20
x_sensors = np.linspace(0, 1, n_sensors_x)
t_sensors = np.linspace(0, 1, n_sensors_t)
xv_sensors, tv_sensors = np.meshgrid(x_sensors, t_sensors)
eval_pts = np.vstack((xv_sensors.ravel(), tv_sensors.ravel())).T  # shape: (400, 2)

data = dde.data.PDEOperatorCartesianProd(
    pde, func_space, eval_pts, 1000, function_variables=[0, 1], num_test=100, batch_size=50
)

# Net - branch network now takes 400 inputs (20x20 spatiotemporal sensors)
net = dde.nn.DeepONetCartesianProd(
    [400, 128, 128, 128],  # Branch: 400 = 20x20 sensors for v(x,t)
    [2, 128, 128, 128],    # Trunk: still (x,t) coordinates
    "tanh",
    "Glorot normal",
)

# Apply output transform to enforce BC/IC as hard constraints
net.apply_output_transform(output_transform)

model = dde.Model(data, net)
model.compile("adam", lr=0.0005)
start_time = datetime.now()
losshistory, train_state = model.train(iterations=10000)
end_time = datetime.now()
print(f"Training time: {end_time - start_time}")
dde.utils.plot_loss_history(losshistory)

func_feats = func_space.random(1)

# For visualization, create spatiotemporal grid
xs = np.linspace(0, 1, num=100)[:, None]
ts = np.linspace(0, 1, num=100)[:, None]
xv_full, tv_full = np.meshgrid(xs.ravel(), ts.ravel())
xt_full = np.vstack((xv_full.ravel(), tv_full.ravel())).T

gt_data = np.load('./ground_truth.npz')
x = np.linspace(0, 1, num=100)[:, None]
t = gt_data['t_history'][:100]
u_true = gt_data['u_gt'][:100]
f_gt = gt_data['f_gt'][:100]  # Assuming f_gt is also spatiotemporal forcing

# Convert f_gt to v_branch format
# f_gt has shape (n_time, n_space), e.g., (100, 100)
# We need to interpolate it to the sensor grid (n_sensors_t, n_sensors_x)

# Original grid for f_gt
x_gt = np.linspace(0, 1, f_gt.shape[1])  # spatial points in f_gt
t_gt = t.ravel()  # temporal points in f_gt

# Create interpolator
f_interp = RegularGridInterpolator((t_gt, x_gt), f_gt, method='cubic', bounds_error=False, fill_value=0)

# Sensor grid points
x_sensors_arr = np.linspace(0, 1, n_sensors_x)
t_sensors_arr = np.linspace(0, 1, n_sensors_t)
xv_sensors, tv_sensors = np.meshgrid(x_sensors_arr, t_sensors_arr)

# Evaluate f_gt at sensor points: need (t, x) order for interpolator
sensor_points = np.vstack((tv_sensors.ravel(), xv_sensors.ravel())).T  # shape: (n_sensors, 2)
f_at_sensors = f_interp(sensor_points)  # shape: (n_sensors,)

# Reshape to v_branch format: (1, n_sensors)
v_branch = f_at_sensors.reshape(1, -1)
print(f"v_branch shape: {v_branch.shape}, created from f_gt shape: {f_gt.shape}")

# Prepare x_trunk: query points for solution
xv, tv = np.meshgrid(x, t)
x_trunk = np.vstack((np.ravel(xv), np.ravel(tv))).T

u_pred = model.predict((v_branch, x_trunk))
u_pred = u_pred.reshape((len(t), 100))

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
im2 = axes[2].imshow(abs_error, extent=[0, 1, t[0], t[-1]], aspect='auto', origin='lower')
axes[2].set_xlabel('x')
axes[2].set_ylabel('t')
axes[2].set_title('Absolute Error')
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig('pi-operator/comparison.png', dpi=150)
plt.close()


# Save the predicted solution
np.savez('pi-operator/prediction.npz', u_pred=u_pred, x=x, t=t)

# Save the model outputs
model.save('pi-operator/model_final')

# Save the loss history
dde.utils.external.save_loss_history(losshistory, 'pi-operator/loss.dat')
