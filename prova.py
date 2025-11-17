import pandas as pd
from model_2d import PINNDeepONet_Wave2D



gt_data = pd.read_csv('data/gt_wave2D_nosource.csv', header=None).values
v_gt = gt_data[gt_data[:, 3] == 0, 5]
print(f"Ground truth v range: [{v_gt.min():.6f}, {v_gt.max():.6f}]")
print(f"Expected range for b=0.5: ~[-0.5, 0.5]")

a_test = 2.0
b_test = 0.0
n_sensors_ic = 20      # Creates 20x20 grid (400 sensors)
n_sensors_src = 20     # Creates 20x20 grid (400 sensors)
branch_hidden = 200
trunk_hidden = 200
p = 200


model = PINNDeepONet_Wave2D(
    n_sensors_ic=n_sensors_ic,
    n_sensors_src=n_sensors_src,
    branch_hidden=branch_hidden,
    trunk_hidden=trunk_hidden,
    p=p
)

u0_test = model.generate_ic_displacement(a_test)
v0_test = model.generate_ic_velocity(b_test)
print(f"IC velocity range: [{v0_test.min():.6f}, {v0_test.max():.6f}]")
print(f"Expected: b*sin(π*x)*sin(π*y) with b={b_test} → range ~[-{b_test}, {b_test}]")