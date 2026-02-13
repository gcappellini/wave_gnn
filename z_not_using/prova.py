
import numpy as np
import matplotlib.pyplot as plt

# Load the saved sensors data
sensors = np.load('./data/rollout_sensors.npz')
u0_sensors = sensors['u0_sensors']  # shape: (n_intervals, n_sensors)
v0_sensors = sensors['v0_sensors']  # shape: (n_intervals, n_sensors)

def fit_sine_series(y, n_terms):
    x = np.linspace(0, 1, len(y))
    A = [np.sin(np.pi * n * x) for n in range(1, n_terms + 1)]
    A = np.vstack(A).T
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    y_fit = A @ coeffs
    return y_fit, coeffs

def plot_sine_fits(sensor_data, title, n_terms_list=[1, 2, 3, 5, 10]):
    n_intervals, n_sensors = sensor_data.shape
    x = np.linspace(0, 1, n_sensors)
    fig, axes = plt.subplots(len(n_terms_list), 1, figsize=(12, 2*len(n_terms_list)), sharex=True)
    for idx, n_terms in enumerate(n_terms_list):
        ax = axes[idx]
        ax.set_title(f'{title} - Sine fit with {n_terms} terms')
        for i in range(n_intervals):
            y = sensor_data[i]
            y_fit, _ = fit_sine_series(y, n_terms)
            ax.plot(x, y, color='gray', alpha=0.2)
            ax.plot(x, y_fit, color='C1', alpha=0.5)
        ax.set_ylabel('Sensor value')
    axes[-1].set_xlabel('Normalized sensor position')
    plt.tight_layout()
    plt.show()

# Plot and fit u0_sensors
# plot_sine_fits(u0_sensors, 'u0_sensors', n_terms_list=[1, 2, 3, 5, 10])
# # Plot and fit v0_sensors
# plot_sine_fits(v0_sensors, 'v0_sensors', n_terms_list=[1, 2, 3, 5, 10])

target = 0.5*np.sin(np.pi * np.linspace(0, 1, u0_sensors.shape[1]))
print(target)

# plot_sine_fits(0.5*np.sin(np.pi * np.linspace(0, 1, u0_sensors.shape[1])), 'u0_sensors', n_terms_list=[1])