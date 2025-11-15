# PINN-DeepONet for Damped Wave Equation with Gaussian Source

## Overview

This implementation trains a Physics-Informed DeepONet to solve the damped wave equation with varying Gaussian source amplitudes.

## Mathematical Formulation

**PDE:**
```
u_tt + k*u_t = c² * u_xx + f(x)
```

**Boundary Conditions (Hard Constraints):**
```
u(0, t) = u(1, t) = 0
```

**Initial Conditions:**
```
u(x, 0) = a * sin(π*x)    (displacement)
u_t(x, 0) = b * sin(π*x)  (velocity)
```

**Gaussian Source:**
```
f(x) = A * exp(-((x - 0.5) / 0.3)²)
```

## Architecture

### Two-Branch DeepONet with 2D IC Output

1. **Branch_IC**: Encodes both initial displacement and velocity
   - Input: `[u0_sensors, v0_sensors]` (size: 2*n_sensors_ic)
   - Output: 2*p features, split into:
     - `b_u`: p features for displacement IC
     - `b_v`: p features for velocity IC

2. **Branch_Source**: Encodes forcing function f(x)
   - Input: source sensors (size: n_sensors_src)
   - Output: p features

3. **Trunk**: Encodes spatiotemporal coordinates (x, t)
   - Input: [x, t]
   - Output: p features

4. **Output**: 
   ```
   u_net = b_u · T + b_v · T + b_src · T + bias
   u = x * (1 - x) * u_net  (hard BC enforcement)
   ```

## Training Strategy

### Case 1: Free Wave (no_source)
- Source type: 'zero'
- Source amplitude: 0.0
- Trains on IC variations only

### Case 2: Forced Wave (with_source)
- Source type: 'gaussian'
- **Training**: Amplitude varied randomly in [0, 5.0]
- **Testing**: Fixed amplitude (e.g., 3.0)
- Goal: Learn response to different forcing strengths

## Physics-Informed Loss

```
L_total = w_pde * L_pde + w_ic_u * L_ic_u + w_ic_v * L_ic_v
```

Where:
- **L_pde**: PDE residual `||u_tt + k*u_t - c²*u_xx - f(x)||²`
- **L_ic_u**: Displacement IC `||u(x,0) - a*sin(π*x)||²`
- **L_ic_v**: Velocity IC `||u_t(x,0) - b*sin(π*x)||²`
- No BC loss needed (hard constraints)

## Files

### Python
- `pinn_deeponet_wave_2branch.py`: Main training script
  - Set `TRAINING_CASE = 'with_source'` for Gaussian source training
  - Set `TRAINING_CASE = 'no_source'` for free wave

### MATLAB Ground Truth
- `wave_equation_matlab_withsource.m`: Generates validation data with Gaussian source
  - Uses pdepe solver with system formulation
  - Outputs: `gt_wave1D_withsource.csv` [t, x, u, f]
  - Visualization: `matlab_wave_withsource.png`

## Usage

### 1. Generate MATLAB Ground Truth
```matlab
% Edit parameters in wave_equation_matlab_withsource.m:
source_amp = 3.0;    % Test amplitude
a_ic = 1.5;          % Displacement IC
b_ic = 0.0;          % Zero initial velocity
T_max = 2.0;

% Run in MATLAB:
wave_equation_matlab_withsource
% Produces: gt_wave1D_withsource.csv
```

### 2. Train PINN-DeepONet
```bash
python pinn_deeponet_wave_2branch.py
```

The script will:
- Train on varying source amplitudes [0, 5.0]
- Test at amplitude 3.0
- Compare against MATLAB solution
- Save model: `pinn_deeponet_wave_withsource.pth`
- Save plots:
  - `pinn_wave_solution_with_source.png` (6 subplots with MATLAB comparison)
  - `pinn_wave_training_with_source.png` (loss history)

## Key Features

1. **Amplitude Randomization**: During training, source amplitude is sampled from [0, source_amplitude] each epoch for generalization

2. **Hard BC Enforcement**: Transformation `u = x*(1-x)*u_net` ensures exact boundary satisfaction without soft penalties

3. **Validation**: Side-by-side comparison with MATLAB pdepe solution
   - Absolute error heatmap
   - Error evolution over time
   - Relative L2 error metrics

4. **Flexible Training**: Switch between free wave and forced wave cases easily

## Parameters

### Model
- n_sensors_ic: 20
- n_sensors_src: 20
- branch_hidden: 50
- trunk_hidden: 50
- p: 50 (embedding dimension)
- wave_speed: c = 1.0
- damping_coeff: k = 1.0

### Training
- n_epochs: 5000
- n_colloc: 200
- lr: 1e-3 with ReduceLROnPlateau
- a_range: (1.0, 2.0) - displacement IC amplitude
- b_range: (0.0, 0.0) - zero initial velocity
- T_max: 2.0

### Gaussian Source
- Center: 0.5
- Width: 0.3
- Training amplitude range: [0, 5.0]
- Test amplitude: 3.0

## Expected Output

### Console Output
```
======================================================================
PHYSICS-INFORMED DEEPONET FOR DAMPED WAVE EQUATION
======================================================================
PDE: u_tt + k*u_t = c² * u_xx + f(x)
...
Training case: with_source
...
Epoch [5000/5000] | Loss: 1.23e-04 | PDE: 5.67e-05 | IC_u: 4.32e-05 | IC_v: 2.31e-05
✓ Model saved: pinn_deeponet_wave_withsource.pth
✓ Loaded ground truth: gt_wave1D_withsource.csv
...
```

### Plots
1. **Solution comparison** (6 subplots):
   - Row 1: PINN prediction, MATLAB solution, absolute error
   - Row 2: Center slice, boundary slice, error evolution

2. **Training history** (4 subplots):
   - Total loss, PDE loss, IC displacement loss, IC velocity loss

## Notes

- Gaussian source is space-dependent only: f(x), not f(x,t)
- Damping term k*u_t provides energy dissipation
- Training samples different amplitudes for robust learning
- Testing at unseen amplitude evaluates generalization
