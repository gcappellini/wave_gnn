import numpy as np

# Load the saved sensors data
sensors = np.load('./data/rollout_sensors.npz')
# These should be (n_intervals, n_sensors)
u0_sensors = sensors['u0_sensors']
v0_sensors = sensors['v0_sensors']

# We'll fit a sine series to each interval and collect the coefficients
# Use the same fit_sine_series as before (only sine terms)
def fit_sine_series(y, n_terms):
    x = np.linspace(0, 1, len(y))
    A = [np.sin(np.pi * (k+1) * x) for k in range(n_terms)]
    A = np.vstack(A).T
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return coeffs

n_terms = 10  # Change if you want more/fewer terms

u0_coeffs = []
v0_coeffs = []
for i in range(u0_sensors.shape[0]):
    u0_coeffs.append(fit_sine_series(u0_sensors[i], n_terms))
    v0_coeffs.append(fit_sine_series(v0_sensors[i], n_terms))

u0_coeffs = np.array(u0_coeffs)  # shape: (n_intervals, n_terms)
v0_coeffs = np.array(v0_coeffs)

# # Compute min/max for each coefficient
# print('u0_sensors coefficients range:')
# for k in range(n_terms):
#     print(f'  a{k+1}: min={u0_coeffs[:,k].min():.3f}, max={u0_coeffs[:,k].max():.3f}')

# print('v0_sensors coefficients range:')
# for k in range(n_terms):
#     print(f'  b{k+1}: min={v0_coeffs[:,k].min():.3f}, max={v0_coeffs[:,k].max():.3f}')

# Fit target with 1-term sine series
target = 0.5 * np.sin(np.pi * np.linspace(0, 1, u0_sensors.shape[1]))
n_terms_target = 1
coeff_target = fit_sine_series(target, n_terms_target)
print(f'Fitted coefficient for target (a1): {coeff_target[0]:.3f}')
