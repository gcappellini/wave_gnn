import numpy as np
import matplotlib.pyplot as plt

gt_data = np.load('./ground_truth.npz')
u_gt = gt_data['u_gt'][:100]
t_history = gt_data['t_history'][:100]
print(u_gt.shape, t_history.shape)
# v = np.load('./v.npz')
# print(v['v'].shape)
# v_random = v['v']
# xs = v['xs']
