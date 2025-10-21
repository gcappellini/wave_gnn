import torch
import numpy
from dataset import create_graph
from test_gcn import stringforce
from scipy.io import savemat
import numpy as np
from plot import plot_features_2d

def rk4_update(input, laplacian, dt=0.01, c=1.0, k=1.0, w1=1.0, w2=1.0):
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
        
        return u_new, v_new


starting_graph = create_graph(zeros=True)
dt = 0.01
tf = 10

time_steps = int(tf/dt)
u_history = numpy.zeros((time_steps, starting_graph.x.shape[0]))
v_history = numpy.zeros((time_steps, starting_graph.x.shape[0]))
f_history = numpy.zeros((time_steps, starting_graph.x.shape[0]))
t_history = np.linspace(0, tf, time_steps)
u_history[0] = starting_graph.x[:, 0].cpu().numpy()

for i in range(time_steps):
    u_new, v_new = rk4_update(starting_graph.x, starting_graph.laplacian)
    starting_graph.x[:, 0] = u_new
    starting_graph.x[:, 1] = v_new
    starting_graph.x[:, 2] = torch.Tensor(stringforce(starting_graph.coords, i*dt, tf))
    u_history[i] = starting_graph.x[:, 0].cpu().numpy()
    v_history[i] = starting_graph.x[:, 1].cpu().numpy()
    f_history[i] = starting_graph.x[:, 2].cpu().numpy()


# Save results to .mat file
matlab_times = np.linspace(0, tf, num=100)
indices = [np.argmin(np.abs(t_history - t)) for t in matlab_times]
pinn_data = u_history[indices]

# pinn_data = pinn_data.swapaxes(1, 2)  # Transpose to match MATLAB's column-major order
savemat('/Users/guglielmocappellini/Desktop/research/code/pinns-wave/wave-gnn/1_gcn_string/matlab/data_rk4.mat', {
    'pinn_data': pinn_data
})

histories = np.array([u_history, v_history, f_history])

plot_features_2d(
starting_graph.nodes,  
histories,
output_file='./figures/gcn_string_rk4.png'
)


