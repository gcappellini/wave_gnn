"""Backend supported: tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np
import os
from datetime import datetime
from plot import plot_loss_components, plot_comparison
import matplotlib.pyplot as plt

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join('logs_pideeponet', timestamp)
os.makedirs(log_dir, exist_ok=True)

seed = 2
np.random.seed(seed)
dde.config.set_random_seed(seed)

c = 1.0
k = 1.0

# PDE
def pde(x, y, v):

    grad_y = dde.zcs.LazyGrad(x, y)
    dy_t = grad_y.compute((0, 1))
    dy_tt = grad_y.compute((0, 2))
    dy_xx = grad_y.compute((2, 0))
    return dy_tt - c * dy_xx + k * dy_t - 10 * v

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

geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# bc = dde.icbc.DirichletBC(geomtime, lambda _: 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, lambda _: 0, lambda _, on_initial: on_initial)

ic_2 = dde.icbc.OperatorBC(
    geomtime,
    lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=1),
    lambda _, on_initial: on_initial
    # lambda x, _: dde.utils.isclose(x[1], 0),
)

pde = dde.data.TimePDE(
    geomtime,
    pde,
    [ic, ic_2],
    num_domain=200,
    num_boundary=40,
    num_initial=20,
    num_test=500,
)

# Function space
func_space = dde.data.GRF(length_scale=0.2)

# Data
eval_pts = np.linspace(0, 1, num=50)[:, None]
data = dde.zcs.PDEOperatorCartesianProd(
    pde, func_space, eval_pts, 1000, function_variables=[0], num_test=100, batch_size=50
)

# Net
net = dde.nn.DeepONetCartesianProd(
    [50, 128, 128, 128],
    [2, 128, 128, 128],
    "tanh",
    "Glorot normal",
)
net.apply_output_transform(output_transform)
model = dde.zcs.Model(data, net)
model.compile("adam", lr=0.0005)
start = datetime.now()
losshistory, train_state = model.train(iterations=20000, model_save_path=f"{log_dir}/model.ckpt")
end = datetime.now()
tr_t = end - start
plot_loss_components(losshistory, log_dir)

func_feats = func_space.random(1)
# xs = np.linspace(0, 1, num=100)[:, None]
# v = func_space.eval_batch(func_feats, xs)[0]
# x, t, u_true = solve_ADR(
#     0,
#     1,
#     0,
#     1,
#     lambda x: 0.01 * np.ones_like(x),
#     lambda x: np.zeros_like(x),
#     lambda u: 0.01 * u**2,
#     lambda u: 0.02 * u,
#     lambda x, t: np.tile(v[:, None], (1, len(t))),
#     lambda x: np.zeros_like(x),
#     100,
#     100,
# )
# u_true = u_true.T
# plt.figure()
# plt.imshow(u_true)
# plt.colorbar()

x= np.linspace(0, 1, num=100)
t= np.linspace(0, 1, num=100)

v_branch = func_space.eval_batch(func_feats, np.linspace(0, 1, num=50)[:, None])


plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, 1, num=50), v_branch[0], 'b-', linewidth=2)
plt.xlabel('x')
plt.ylabel('v(x)')
plt.title('Force v(x)')
plt.grid(True)
plt.savefig(os.path.join(log_dir, 'v_branch.png'), dpi=300, bbox_inches='tight')
plt.close()

xv, tv = np.meshgrid(x, t)
x_trunk = np.vstack((np.ravel(xv), np.ravel(tv))).T
u_pred = model.predict((v_branch, x_trunk))
u_pred = u_pred.reshape((100, 100))

plot_comparison(0*u_pred, u_pred, 0, log_dir)

# Save inputs and predictions to CSV
np.savetxt(os.path.join(log_dir, 'v_branch.csv'), v_branch, delimiter=',')
np.savetxt(os.path.join(log_dir, 'x_trunk.csv'), x_trunk, delimiter=',')
np.savetxt(os.path.join(log_dir, 'u_pred.csv'), u_pred, delimiter=',')

with open(os.path.join(log_dir, 'training_time.txt'), 'w') as f:
    f.write(f"Training time: {tr_t}\n")