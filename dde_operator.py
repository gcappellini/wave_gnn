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

# Physical parameters
c = 1.0         # wave speed
k = 1.0         # damping coefficient

# Scaling parameters
L = 1.0         # spatial domain length
T_max = 1.0     # time domain length
u_max = 0.04    # maximum displacement (output scale)
f_max = 3.0     # maximum forcing (input scale)

# Derived non-dimensional parameters for scaled PDE
c_star = c**2 * T_max**2 / L**2      # scaled wave speed squared
f_star = f_max * T_max**2 / u_max    # scaled forcing coefficient
k_star = k * T_max                    # scaled damping coefficient

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
    
# PDE - using scaled variables
def pde(x, y, v):
    """
    Scaled PDE: all variables are non-dimensionalized to [0,1] range
    
    x: scaled coordinates [x_scaled, t_scaled] where both ∈ [0, 1]
    y: scaled displacement u_scaled = u / u_max
    v: scaled forcing f_scaled = f / f_max
    
    Original PDE: u_tt = c² u_xx + f - k u_t
    Scaled PDE: u_scaled_tt = c_star * u_scaled_xx + f_star * f_scaled - k_star * u_scaled_t
    """
    dy_t = dde.grad.jacobian(y, x, j=1)       # ∂u_scaled/∂t_scaled
    dy_tt = dde.grad.hessian(y, x, j=1)       # ∂²u_scaled/∂t_scaled²
    dy_xx = dde.grad.hessian(y, x, j=0)       # ∂²u_scaled/∂x_scaled²
    
    return dy_tt - c_star * dy_xx - f_star * v + k_star * dy_t 

def output_transform(x, y):
    """
    Transform network output to satisfy:
    - BC: u(0,t) = u(1,t) = 0 (spatial boundaries)
    - IC: u(x,0) = 0 (initial displacement)
    - IC: du/dt(x,0) = 0 (initial velocity)
    
    For DeepONet:
    x is a tuple: (branch_input, trunk_input)
    x[0]: branch input (function evaluations) - shape (n_functions, n_sensors)
    x[1]: trunk input (x, t coordinates) - shape (n_points, 2)
    y: raw network output - shape (n_functions, n_points)
    
    To enforce both u(x,0)=0 and du/dt(x,0)=0, we use:
    u(x,t) = x*(1-x) * t^2 * y_network
    
    This gives:
    - u(x,0) = 0 (displacement IC)
    - du/dt = x*(1-x) * 2t * y_network, so du/dt(x,0) = 0 (velocity IC)
    - u(0,t) = u(1,t) = 0 (spatial BC)
    """
    # Extract trunk input (spatiotemporal coordinates)
    trunk_input = x[1]  # shape: (n_points, 2)
    x_coord = trunk_input[:, 0:1]  # spatial coordinate, shape (n_points, 1)
    t_coord = trunk_input[:, 1:2]  # time coordinate, shape (n_points, 1)
    
    # Compute transform: t^2 instead of t to enforce velocity IC
    transform = x_coord * (1 - x_coord) * t_coord**2
    
    # Transpose to shape (1, n_points) and broadcast multiply with y (n_functions, n_points)
    return y * transform.T

geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# BC and IC constraints
# Spatial boundary conditions: u(0,t) = u(1,t) = 0
bc = dde.icbc.DirichletBC(geomtime, lambda _: 0, lambda _, on_boundary: on_boundary)

# Initial condition on displacement: u(x,0) = 0
ic_u = dde.icbc.IC(geomtime, lambda _: 0, lambda _, on_initial: on_initial)

# Initial condition on velocity: du/dt(x,0) = 0
def velocity_ic(x, y, v):
    """Compute du/dt and enforce it to be 0 at t=0"""
    return dde.grad.jacobian(y, x, j=1)  # This should be 0 at initial time

ic_v = dde.icbc.OperatorBC(
    geomtime,
    velocity_ic,
    lambda _, on_initial: on_initial
)

pde = dde.data.TimePDE(
    geomtime,
    pde,
    # [bc, ic_u, ic_v], 
    [], 
    num_domain=100,
    num_boundary=20,
    num_initial=10,
    num_test=250,
)

# Function space - using spatiotemporal forcing v(x,t)
func_space = SpatioTemporalGRF(length_scale_x=0.6, length_scale_t=0.3, f_scale=3.0)

# Data
# Sensor points now need to cover both x and t
n_sensors_x = 25
n_sensors_t = 25
branch_in = n_sensors_t * n_sensors_x
x_sensors = np.linspace(0, 1, n_sensors_x)
t_sensors = np.linspace(0, 1, n_sensors_t)
xv_sensors, tv_sensors = np.meshgrid(x_sensors, t_sensors)
eval_pts = np.vstack((xv_sensors.ravel(), tv_sensors.ravel())).T  # shape: (25, 2)

data = dde.data.PDEOperatorCartesianProd(
    pde, func_space, eval_pts, 1000, function_variables=[0, 1], num_test=100, batch_size=50
)

# Net - branch network now takes 25 inputs (5x5 spatiotemporal sensors)
net = dde.nn.DeepONetCartesianProd(
    [branch_in, 128, 128, 128],  # Branch: 25 = 5x5 sensors for v(x,t)
    [2, 128, 128, 128],    # Trunk: still (x,t) coordinates
    "tanh",
    "Glorot normal",
)

# Apply output transform to enforce BC/IC as hard constraints
net.apply_output_transform(output_transform)

model = dde.Model(data, net)
model.compile("adam", lr=0.005)
losshistory, train_state = model.train(iterations=5000)
dde.utils.plot_loss_history(losshistory, fname=f'pi-operator/loss_history_{timestamp}.png')

func_feats = func_space.random(1)

# For visualization, create spatiotemporal grid
xs = np.linspace(0, 1, num=100)[:, None]
ts = np.linspace(0, 1, num=100)[:, None]
xv_full, tv_full = np.meshgrid(xs.ravel(), ts.ravel())
xt_full = np.vstack((xv_full.ravel(), tv_full.ravel())).T

gt_data = np.load('./ground_truth.npz')
x = np.linspace(0, 1, num=100)[:, None]
t = gt_data['t_history'][80:180]
u_true = gt_data['u_gt'][80:180]  # Physical values
f_gt = gt_data['f_gt'][80:180]    # Physical values

# Scale ground truth to [0, 1] range
u_true_scaled = u_true / u_max
f_gt_scaled = f_gt / f_max

# Convert f_gt_scaled to v_branch format
# f_gt_scaled has shape (n_time, n_space), e.g., (100, 100)
# We need to interpolate it to the sensor grid (n_sensors_t, n_sensors_x)

# Original grid for f_gt_scaled
x_gt = np.linspace(0, 1, f_gt_scaled.shape[1])  # spatial points in f_gt (0 to 1)
t_gt = t.ravel()  # temporal points in f_gt (actual time values)

# Create interpolator using actual coordinates
f_interp = RegularGridInterpolator(
    (t_gt, x_gt), 
    f_gt_scaled,  # Use scaled forcing
    method='cubic', 
    bounds_error=False, 
    fill_value=0
)

# Sensor grid points in normalized [0, 1] space
# Map from [0, 1] to actual time range
t_min, t_max = t_gt.min(), t_gt.max()
x_sensors_normalized = np.linspace(0, 1, n_sensors_x)  # Already in [0, 1]
t_sensors_normalized = np.linspace(0, 1, n_sensors_t)  # Normalized time

# Map normalized time sensors to actual time values
t_sensors_actual = t_min + t_sensors_normalized * (t_max - t_min)

# Create meshgrid with actual coordinates
xv_sensors_actual, tv_sensors_actual = np.meshgrid(x_sensors_normalized, t_sensors_actual)

# Evaluate f_gt at sensor points: (t, x) order for interpolator
sensor_points = np.vstack((tv_sensors_actual.ravel(), xv_sensors_actual.ravel())).T
f_at_sensors = f_interp(sensor_points)  # shape: (n_sensors,)

# Reshape to v_branch format: (1, n_sensors)
v_branch = f_at_sensors.reshape(1, -1)

# Prepare x_trunk: query points for solution
xv, tv = np.meshgrid(x, t)
x_trunk = np.vstack((np.ravel(xv), np.ravel(tv))).T

# Predict scaled output
u_pred_scaled = model.predict((v_branch, x_trunk))
u_pred_scaled = u_pred_scaled.reshape((len(t), 100))

# Scale back to physical units
u_pred = u_pred_scaled * u_max

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
plt.savefig(f'pi-operator/comparison_{timestamp}.png', dpi=150)
plt.close()

# Plot f_at_sensors and f_gt side by side
fig_f, axes_f = plt.subplots(1, 3, figsize=(15, 4))

# Left subplot: forcing function at sensor locations (scaled)
f_sensors_2d = f_at_sensors.reshape((n_sensors_t, n_sensors_x))
im_f0 = axes_f[0].imshow(f_sensors_2d, extent=[0, 1, t[0], t[-1]], aspect='auto', origin='lower')
axes_f[0].set_xlabel('x (sensor positions)')
axes_f[0].set_ylabel('t (sensor positions)')
axes_f[0].set_title('Forcing at Sensors (Scaled)')
plt.colorbar(im_f0, ax=axes_f[0])

# Right subplot: entire f_gt (physical units)
im_f1 = axes_f[1].imshow(f_gt_scaled, extent=[0, 1, t[0], t[-1]], aspect='auto', origin='lower')
axes_f[1].set_xlabel('x')
axes_f[1].set_ylabel('t')
axes_f[1].set_title('Ground Truth Forcing (Scaled)')
plt.colorbar(im_f1, ax=axes_f[1])

# Right subplot: entire f_gt (physical units)
im_f2 = axes_f[2].imshow(f_gt, extent=[0, 1, t[0], t[-1]], aspect='auto', origin='lower')
axes_f[2].set_xlabel('x')
axes_f[2].set_ylabel('t')
axes_f[2].set_title('Ground Truth Forcing (Physical Units)')
plt.colorbar(im_f2, ax=axes_f[2])

plt.tight_layout()
plt.savefig(f'pi-operator/f_sensors_{timestamp}.png', dpi=150)
plt.close()

# Save the predicted solution
np.savez(f'pi-operator/prediction_{timestamp}.npz', u_pred=u_pred, x=x, t=t)

# Save the model outputs
model.save(f'pi-operator/model_final_{timestamp}')

# Save the loss history
dde.utils.external.save_loss_history(losshistory, f'pi-operator/loss_{timestamp}.dat')
