# Physics-Informed DeepONet for 1D Wave Equation

This implementation trains a Physics-Informed DeepONet (PI-DeepONet) to learn the solution operator for the 1D wave equation.

## Problem Setup

**PDE**: 
```
u_tt - c² u_xx = 0,  (x,t) ∈ [0,1] × [0,1]
```

**Boundary Conditions**:
```
u(0,t) = u(1,t) = 0  (Dirichlet)
```

**Initial Conditions**:
```
u(x,0) = u0(x) = sin(πx) + sin(aπx)  (displacement)
u_t(x,0) = 0  (velocity)
```

where `a` is a parameter that varies between different functions.

## Architecture

**PI-DeepONet** consists of two networks:

1. **Branch Network**: 
   - Input: Initial condition u0(x) sampled at `m` sensor locations
   - Architecture: [m] → [100, 100, 100] → [basis_size]
   - Activation: ReLU
   - Learns features from initial conditions

2. **Trunk Network**:
   - Input: Space-time coordinates (x,t)
   - Architecture: [2] → [100, 100, 100] → [basis_size]
   - Activation: tanh
   - Learns basis functions in space-time

3. **Output**: 
   - Dot product of branch and trunk outputs
   - Output transform: `u_net * x * (1-x)` to hard-enforce boundary conditions

## Training

The network is trained on:
- **100 different initial conditions** (varying parameter `a`)
- **800 points per function** (600 PDE + 100 BC + 100 IC)
- **Total: 80,000 training points**

Loss function:
```
L = L_data + λ_pde * L_pde
```

where:
- `L_data`: MSE between predictions and true solutions (d'Alembert)
- `L_pde`: MSE of PDE residual (u_tt - c² u_xx)

## Usage

```bash
python wave1D_pideeponet.py
```

### Key Parameters

```python
# In wave1D_pideeponet.py:
c = 1.0              # Wave speed
num_functions = 100  # Number of training functions
m = 100              # Number of branch sensors
num_domain = 600     # PDE points per function
num_boundary = 100   # BC points per function  
num_initial = 100    # IC points per function
iters = 40000        # Training iterations
basis_size = 100     # Latent dimension
```

## Output

The script produces:

1. **Training logs**: Loss history and metrics
2. **Visualizations**: 
   - True solutions
   - Predicted solutions
   - Absolute errors
   - For both training and unseen test functions
3. **Saved model**: Checkpoint for inference
4. **Parameters**: JSON file with all settings and results

Results are saved in `logs_pideeponet/<timestamp>/`

## Validation

The trained operator is tested on:
1. **Training functions**: Check for overfitting
2. **Unseen functions**: Test generalization with new values of `a`

Metrics:
- L2 relative error
- Pointwise absolute error
- Mean and std across test set

## Example Results

After training, you should see:
```
Testing training function 0 (a = 2.34)...
  L2 relative error: 1.23e-03

Testing unseen function (a = 1.50)...
  L2 relative error: 2.45e-03
```

## Notes

- **d'Alembert solution** is used as ground truth
- **Periodic extension** applied for wave propagation outside [0,1]
- **Hard boundary conditions** via output transform
- **Initial velocity = 0** (only displacement varies)

## Comparison with Standard PINN

| Method | Training | Inference | Generalization |
|--------|----------|-----------|----------------|
| **PINN** | Train per function | Instant | None (one function only) |
| **PI-DeepONet** | Train once on many | Instant | New functions without retraining |

PI-DeepONet learns the **operator** mapping initial conditions to solutions, enabling zero-shot inference on new initial conditions.
