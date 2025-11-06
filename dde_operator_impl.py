"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle
original code: https://deepxde.readthedocs.io/en/latest/demos/operator/poisson.1d.pideeponet.html
"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np

C = 1.0

# Poisson equation: -u_xx = f
def equation(x, y, f):
    dy_tt = dde.grad.hessian(y, x, i=1, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_tt-C**2 * dy_xx - f


# Domain is interval [0, 1]
geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


# Zero Dirichlet BC
def u_boundary(_):
    return 0


def boundary(_, on_boundary):
    return on_boundary


bc = dde.icbc.DirichletBC(geomtime, u_boundary, boundary)

# Define PDE
pde = dde.data.TimePDE(geomtime, equation, bc, num_domain=100, num_boundary=2)

# Function space for f(x) are polynomials
degree = 3
space = dde.data.PowerSeries(N=degree + 1)

# Choose evaluation points
num_eval_points = 10
evaluation_points = geomtime.uniform_points(num_eval_points, boundary=True)

# Define PDE operator
pde_op = dde.data.PDEOperatorCartesianProd(
    pde,
    space,
    evaluation_points,
    num_function=100,
)

# Setup DeepONet
dim_x = 1
p = 32
net = dde.nn.DeepONetCartesianProd(
    [num_eval_points, 32, p],
    [dim_x, 32, p],
    activation="tanh",
    kernel_initializer="Glorot normal",
)

# Define and train model
model = dde.Model(pde_op, net)
dde.optimizers.set_LBFGS_options(maxiter=1000)
model.compile("L-BFGS")
model.train()

# Plot realisations of f(x)
n = 3
features = space.random(n)
fx = space.eval_batch(features, evaluation_points)

x = geom.uniform_points(100, boundary=True)
y = model.predict((fx, x))

# Setup figure
fig = plt.figure(figsize=(7, 8))
plt.subplot(2, 1, 1)
plt.title("Poisson equation: Source term f(x) and solution u(x)")
plt.ylabel("f(x)")
z = np.zeros_like(x)
plt.plot(x, z, "k-", alpha=0.1)

# Plot source term f(x)
for i in range(n):
    plt.plot(evaluation_points, fx[i], "--")

# Plot solution u(x)
plt.subplot(2, 1, 2)
plt.ylabel("u(x)")
plt.plot(x, z, "k-", alpha=0.1)
for i in range(n):
    plt.plot(x, y[i], "-")
plt.xlabel("x")

plt.show()