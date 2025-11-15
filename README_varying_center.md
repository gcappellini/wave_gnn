# Training with Varying Gaussian Source Center Locations

## Overview

Modified the PINN-DeepONet to train on **varying Gaussian source center locations** instead of varying amplitudes. This allows the model to learn the wave response to forcing applied at different spatial positions.

## Key Changes

### 1. Updated Method Signatures

**`generate_source()`**:
```python
def generate_source(self, source_type='gaussian', amplitude=0.5, center=0.5):
    # Now accepts 'center' parameter for Gaussian location
```

**`source_function()`**:
```python
def source_function(self, x, source_type='gaussian', amplitude=0.5, center=0.5):
    # Now accepts 'center' parameter for evaluation at arbitrary points
```

**`train_pinn()`**:
```python
def train_pinn(self, ..., source_center=0.5, center_range=None, ...):
    # New parameters:
    # - source_center: fixed center (used when center_range=None)
    # - center_range: tuple (min, max) for random sampling during training
```

**`plot_solution()`**:
```python
def plot_solution(model, ..., source_center=0.5, ...):
    # Now accepts 'source_center' for visualization
```

### 2. Training Loop Modification

Instead of sampling random amplitude:
```python
# OLD: Random amplitude sampling
amp_sample = source_amplitude * torch.rand(1).item()
```

Now samples random center location:
```python
# NEW: Random center sampling
if center_range is not None and source_type == 'gaussian':
    center_sample = center_range[0] + (center_range[1] - center_range[0]) * torch.rand(1).item()
else:
    center_sample = source_center
```

### 3. Training Configuration

**Training parameters**:
```python
source_type='gaussian',
source_amplitude=3.0,        # Fixed amplitude
source_center=0.5,           # Default (not used when center_range provided)
center_range=(0.1, 0.9),     # Randomly sample center from this range
```

**Test parameters**:
```python
source_test_amp = 3.0        # Fixed amplitude
source_test_center = 0.5     # Test at center of domain (middle of training range)
```

## Mathematical Formulation

**Gaussian Source**:
```
f(x; A, x_c, σ) = A * exp(-((x - x_c) / σ)²)
```

Where:
- A = 3.0 (fixed amplitude)
- x_c ∈ [0.1, 0.9] (varying during training)
- σ = 0.3 (fixed width)

**Training Strategy**:
- Each epoch samples a random center location x_c from [0.1, 0.9]
- Model learns to respond to forcing at different positions
- Tests generalization at x_c = 0.5 (center of domain)

## Usage

### 1. Train the Model

```bash
python pinn_deeponet_wave_2branch.py
```

Set `TRAINING_CASE = 'with_source'` (line ~578)

The script will:
- Train on Gaussian sources centered randomly in [0.1, 0.9]
- Use fixed amplitude A=3.0
- Test at center location x_c=0.5
- Save model: `pinn_deeponet_wave_withsource.pth`

### 2. Generate MATLAB Ground Truth

Create/modify the MATLAB script to match the test case:
```matlab
source_amp = 3.0;
source_center = 0.5;  % Test center
source_width = 0.3;
```

Run:
```matlab
wave_equation_matlab_withsource
```

This produces: `gt_wave1D_withsource.csv`

### 3. Expected Console Output

```
======================================================================
PHYSICS-INFORMED DEEPONET FOR DAMPED WAVE EQUATION
======================================================================
...
Training case: with_source
...
======================================================================
TRAINING PHYSICS-INFORMED DEEPONET FOR WAVE EQUATION
======================================================================
...
Source type: gaussian, amplitude: 3.0, center range: (0.1, 0.9)
...
```

## Generalization Testing

The model is trained on sources centered in [0.1, 0.9] and tested at x_c=0.5:

- **Training**: Source can be near left boundary (x=0.1), center (x=0.5), or right boundary (x=0.9)
- **Testing**: Evaluate at center location (x=0.5)
- **Goal**: Model should generalize to different source positions

## Comparison with Previous Approach

| Aspect | Previous (Varying Amplitude) | Current (Varying Center) |
|--------|------------------------------|-------------------------|
| **Amplitude** | Random ∈ [0, 5.0] | Fixed = 3.0 |
| **Center** | Fixed = 0.5 | Random ∈ [0.1, 0.9] |
| **Width** | Fixed = 0.3 | Fixed = 0.3 |
| **Learning** | Response to different forcing strengths | Response to different forcing locations |
| **Test case** | A=3.0 at x=0.5 | A=3.0 at x=0.5 |

## Advantages of Varying Center

1. **Spatial Generalization**: Model learns that forcing at different positions creates different wave patterns
2. **Boundary Effects**: Learns interaction with boundaries (sources near x=0 and x=1 behave differently)
3. **Physical Intuition**: More realistic - external forcing often varies in location, not just strength
4. **Richer Training**: Sources near boundaries vs center produce qualitatively different solutions

## Implementation Notes

- Width σ=0.3 kept constant (varying width could be future work)
- Center range [0.1, 0.9] avoids extreme boundary effects
- Hard BC still enforced via u = x(1-x)*u_net transformation
- Same damping (k=1.0) and wave speed (c=1.0) as before

## Future Extensions

1. **Vary both amplitude and center**: Sample from (A, x_c) joint distribution
2. **Variable width**: Add σ to the training parameters
3. **Multiple sources**: Train on superposition of Gaussians at different locations
4. **Time-varying center**: f(x,t) with moving source location
