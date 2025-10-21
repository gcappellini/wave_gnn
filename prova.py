from train import rk4_loss
import torch
import numpy as np
import os
from dataset import build_laplacian_matrix, WaveGNN1D, create_graph


initial_graph = create_graph(282)
nodes, elements = initial_graph.nodes, initial_graph.elements

gn_solver = WaveGNN1D(initial_graph.laplacian)
interior_mask = ~initial_graph.bc_mask

features = gn_solver.forward(initial_graph.x)

err, err_1, err_2 = rk4_loss(
    interior_mask,
    initial_graph.x,
    features,
    initial_graph.laplacian)

print(err, err_1, err_2)