# Data Scaling Implementation Guide

## Overview

Data scaling has been implemented to normalize input and output features, which improves training stability and convergence for the GCN model. The implementation supports both standard normalization (z-score) and min-max scaling.

## Features

### Supported Scaling Methods

1. **Standard Normalization (z-score)**
   - Formula: `(x - mean) / std`
   - Good for data with outliers
   - Preserves the shape of the original distribution
   
2. **Min-Max Normalization**
   - Formula: `(x - min) / (max - min)`
   - Scales data to [0, 1] range
   - Sensitive to outliers

### Scaling Options

- **Per-feature scaling**: Each feature (u, v, f) is scaled independently
- **Global scaling**: All features scaled with same statistics
- **Configurable epsilon**: Prevents division by zero

## Configuration

### Enable/Disable Scaling

In `configs/dataset/default.yaml`:

```yaml
scaling:
  enabled: true           # Enable or disable scaling
  method: "standard"      # "standard" or "minmax"
  per_feature: true       # Scale each feature independently
  epsilon: 1.0e-8        # Small value to prevent division by zero
```

### Examples

#### Standard Normalization (Recommended)
```yaml
scaling:
  enabled: true
  method: "standard"
  per_feature: true
  epsilon: 1.0e-8
```

#### Min-Max Normalization
```yaml
scaling:
  enabled: true
  method: "minmax"
  per_feature: true
  epsilon: 1.0e-8
```

#### Disable Scaling
```yaml
scaling:
  enabled: false
```

## Implementation Details

### Training Pipeline

1. **Fit Scaler** (train_model)
   ```python
   if cfg.dataset.scaling.enabled:
       scaler = DataScaler(...)
       scaler.fit(train_loader)
       scaler.save("scaler.pkl")
   ```

2. **Scale Inputs** (during training)
   ```python
   if scaler is not None:
       data.x = scaler.transform_input(data.x)
   ```

3. **Model Forward Pass**
   ```python
   output_scaled = model(input_scaled, edge_index, bc_mask)
   ```

4. **Inverse Scale for Loss** (during validation)
   ```python
   output = scaler.inverse_transform_output(output_scaled)
   loss = criterion(output, target)  # Loss in original space
   ```

### Testing Pipeline

1. **Load Scaler**
   ```python
   scaler = DataScaler.load("scaler.pkl")
   ```

2. **Scale Inputs During Simulation**
   ```python
   features_scaled = scaler.transform_input(features)
   preds_scaled = model(features_scaled, ...)
   preds = scaler.inverse_transform_output(preds_scaled)
   ```

## Files Modified

### New Files
- `scaler.py`: DataScaler class implementation

### Modified Files
- `configs/dataset/default.yaml`: Added scaling configuration
- `train.py`: 
  - Import DataScaler
  - Fit scaler on training data
  - Scale inputs during training
  - Save scaler alongside model
  - Added `evaluate_loader_with_scaling()`
- `test_gcn.py`:
  - Load scaler during testing
  - Apply scaling during simulation
  - Updated `simulate_wave()` to accept scaler
- `main.py`:
  - Handle scaler return value from training
  - Pass scaler path to testing

## Usage Examples

### Training with Scaling

```bash
# Use default scaling (standard normalization)
python main.py

# Use min-max scaling
python main.py dataset.scaling.method=minmax

# Disable per-feature scaling
python main.py dataset.scaling.per_feature=false

# Disable scaling entirely
python main.py dataset.scaling.enabled=false
```

### Experiment Comparisons

```bash
# Compare scaling methods
python main.py -m dataset.scaling.method=standard,minmax

# Compare with/without scaling
python main.py -m dataset.scaling.enabled=true,false

# Compare per-feature vs global scaling
python main.py -m dataset.scaling.per_feature=true,false
```

## Benefits

1. **Improved Convergence**: Normalized features lead to more stable gradients
2. **Better Generalization**: Prevents features with larger magnitudes from dominating
3. **Consistent Scaling**: Same normalization applied during training and testing
4. **Reproducibility**: Scaler parameters saved with model checkpoint

## Best Practices

1. **Always fit scaler on training data only**
   - Never include validation/test data in scaling statistics
   - Current implementation correctly fits on `train_loader`

2. **Save scaler with model**
   - Scaler saved automatically in `checkpoints/scaler.pkl`
   - Loaded automatically during testing if available

3. **Compute loss in original space** (for physics-informed losses)
   - Model outputs are inverse-scaled before computing PDE residuals
   - Ensures physical consistency of losses

4. **Choose appropriate method**
   - Standard normalization: Better for data with outliers (recommended)
   - Min-max normalization: When you need bounded output range

## Troubleshooting

### Issue: Training unstable with scaling enabled

**Solution**: Try adjusting learning rate
```bash
python main.py training.learning_rate=0.001
```

### Issue: Scaler not found during testing

**Check**:
1. Scaling was enabled during training
2. Scaler file exists in `checkpoints/scaler.pkl`
3. Correct path passed to `test_model()`

### Issue: Poor performance with scaling

**Try**:
1. Disable per-feature scaling: `dataset.scaling.per_feature=false`
2. Switch method: `dataset.scaling.method=minmax`
3. Disable scaling: `dataset.scaling.enabled=false`

## Technical Notes

### Scaler State

The DataScaler stores:
- Input statistics: mean, std (or min, max)
- Output statistics: mean, std (or min, max)
- Configuration: method, per_feature, epsilon
- Fitted flag: whether scaler has been fitted

### Thread Safety

- Scaler is fitted once before training
- Read-only during training/testing
- Thread-safe for inference

### Memory Considerations

- Scaler stores statistics per feature (if per_feature=True)
- For 3 input features: 3 means + 3 stds = 6 values
- Minimal memory footprint (~100 bytes)

## Future Improvements

Potential enhancements:
1. Robust scaling (using median/IQR)
2. Log scaling for highly skewed features
3. Custom scaling per feature type (u, v, f separately)
4. Online/adaptive scaling during training
5. Feature-wise scaling configuration

## References

- Standard scaling: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- Min-max scaling: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
- Feature scaling best practices: https://sebastianraschka.com/Articles/2014_about_feature_scaling.html
