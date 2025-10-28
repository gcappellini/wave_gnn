"""
Adaptive Loss Weighting for Physics-Informed Neural Networks

This module implements various strategies for adaptively weighting multiple loss terms
during training to balance their contributions and improve convergence.

Strategies:
1. Equal initialization: Normalize weights so all losses contribute equally at epoch 1
2. Moving average: Track exponential moving average of loss magnitudes
3. GradNorm: Balance gradient magnitudes across tasks (future work)
4. Softmax adaptation: Adjust weights based on relative loss values
"""

import torch
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

log = logging.getLogger(__name__)


class AdaptiveLossWeights:
    """
    Manages adaptive weighting of multiple loss terms.
    
    Supports various strategies:
    - 'equal_init': Equalize all loss terms at epoch 1, then keep fixed
    - 'equal_init_ema': Equalize at epoch 1, then use exponential moving average
    - 'ema': Always use exponential moving average (no initial equalization)
    - 'fixed': Use fixed weights from config (no adaptation)
    
    Args:
        loss_names: List of loss term names (e.g., ['PI_loss1', 'PI_loss2', 'RK4_loss1', 'RK4_loss2'])
        initial_weights: Dictionary mapping loss names to initial weights
        strategy: Adaptation strategy ('equal_init', 'equal_init_ema', 'ema', 'fixed')
        ema_alpha: Exponential moving average coefficient (0.9 means 90% old, 10% new)
        update_frequency: How often to update weights (in epochs)
        eps: Small constant to avoid division by zero
        min_weight: Minimum allowed weight value
        max_weight: Maximum allowed weight value
    """
    
    def __init__(
        self,
        loss_names: List[str],
        initial_weights: Dict[str, float],
        strategy: str = 'equal_init',
        ema_alpha: float = 0.9,
        update_frequency: int = 1,
        eps: float = 1e-8,
        min_weight: float = 1e-4,
        max_weight: float = 1e4,
    ):
        self.loss_names = loss_names
        self.strategy = strategy
        self.ema_alpha = ema_alpha
        self.update_frequency = update_frequency
        self.eps = eps
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Initialize weights from config
        self.weights = {name: initial_weights.get(name, 1.0) for name in loss_names}
        self.initial_weights = self.weights.copy()
        
        # Tracking variables
        self.loss_ema = {name: None for name in loss_names}
        self.initial_losses = None
        self.epoch = 0
        self.initialized = False
        
        log.info(f"Adaptive loss weights initialized with strategy: {strategy}")
        log.info(f"Initial weights: {self.weights}")
    
    def update(self, epoch: int, loss_values: Dict[str, float]) -> Dict[str, float]:
        """
        Update weights based on current loss values and strategy.
        
        Args:
            epoch: Current epoch number (1-indexed)
            loss_values: Dictionary mapping loss names to their current values
            
        Returns:
            Updated weights dictionary
        """
        self.epoch = epoch
        
        # Filter out invalid losses
        valid_losses = {k: v for k, v in loss_values.items() 
                       if k in self.loss_names and not (np.isnan(v) or np.isinf(v))}
        
        if not valid_losses:
            log.warning(f"No valid loss values at epoch {epoch}, keeping current weights")
            return self.weights
        
        # Strategy: fixed weights (no adaptation)
        if self.strategy == 'fixed':
            return self.weights
        
        # Strategy: equal initialization at epoch 1
        if self.strategy == 'equal_init' and epoch == 1:
            self._equalize_losses(valid_losses)
            self.initialized = True
            return self.weights
        
        # Strategy: equal init + EMA
        if self.strategy == 'equal_init_ema':
            if epoch == 1:
                self._equalize_losses(valid_losses)
                self.initialized = True
            elif epoch > 1 and epoch % self.update_frequency == 0:
                self._update_ema_weights(valid_losses)
            return self.weights
        
        # Strategy: pure EMA (no equal init)
        if self.strategy == 'ema':
            if epoch % self.update_frequency == 0:
                self._update_ema_weights(valid_losses)
            return self.weights
        
        return self.weights
    
    def _equalize_losses(self, loss_values: Dict[str, float]):
        """
        Compute weights to equalize all loss terms.
        
        The idea: if we have losses L1, L2, L3, we want:
            w1 * L1 ≈ w2 * L2 ≈ w3 * L3
        
        Solution: Set wi = 1 / Li (with normalization)
        This makes wi * Li ≈ constant for all i
        """
        if not loss_values:
            return
        
        # Store initial losses for reference
        self.initial_losses = loss_values.copy()
        
        # Compute inverse losses (with epsilon for stability)
        inverse_losses = {name: 1.0 / (val + self.eps) for name, val in loss_values.items()}
        
        # Normalize so that mean weight equals mean of initial weights
        mean_inverse = np.mean(list(inverse_losses.values()))
        mean_initial = np.mean([self.initial_weights[name] for name in loss_values.keys()])
        normalization = mean_initial / (mean_inverse + self.eps)
        
        # Update weights
        for name in loss_values.keys():
            new_weight = inverse_losses[name] * normalization
            # Clamp to reasonable range
            new_weight = np.clip(new_weight, self.min_weight, self.max_weight)
            self.weights[name] = new_weight
        weighted_loss = {name: self.weights[name] * loss_values[name] for name in loss_values.keys()}
        total_weighted_loss = sum(weighted_loss.values())
        self.weights = {name: self.weights[name] / (total_weighted_loss + self.eps)
                        for name in loss_values.keys()}
        
        log.info(f"Equalized weights at epoch 1:")
        for name in self.loss_names:
            if name in loss_values:
                weighted_loss = self.weights[name] * loss_values[name]
                log.info(f"  {name}: loss={loss_values[name]:.3e}, "
                        f"weight={self.weights[name]:.3e}, "
                        f"weighted={weighted_loss:.3e}")
    
    def _update_ema_weights(self, loss_values: Dict[str, float]):
        """
        Update weights using exponential moving average of loss magnitudes.
        
        The weights are adjusted to keep weighted losses approximately constant.
        """
        # Update EMA for each loss
        for name, value in loss_values.items():
            if self.loss_ema[name] is None:
                self.loss_ema[name] = value
            else:
                self.loss_ema[name] = self.ema_alpha * self.loss_ema[name] + (1 - self.ema_alpha) * value
        
        # Compute new weights based on EMA
        valid_emas = {k: v for k, v in self.loss_ema.items() if v is not None}
        if not valid_emas:
            return
        
        # Inverse of EMA gives weights to equalize contributions
        inverse_emas = {name: 1.0 / (ema + self.eps) for name, ema in valid_emas.items()}
        
        # Normalize to keep mean weight constant
        mean_inverse = np.mean(list(inverse_emas.values()))
        mean_current = np.mean([self.weights[name] for name in valid_emas.keys()])
        normalization = mean_current / (mean_inverse + self.eps)
        
        # Update weights with smooth transition
        alpha_smooth = 0.9  # Smooth weight updates to avoid instability
        for name in valid_emas.keys():
            new_weight = inverse_emas[name] * normalization
            new_weight = np.clip(new_weight, self.min_weight, self.max_weight)
            # Smooth update
            self.weights[name] = alpha_smooth * self.weights[name] + (1 - alpha_smooth) * new_weight
        
        if self.epoch % (self.update_frequency * 5) == 0:  # Log every 5 updates
            log.info(f"Updated weights at epoch {self.epoch} (EMA strategy):")
            for name in self.loss_names:
                if name in valid_emas:
                    log.info(f"  {name}: ema={self.loss_ema[name]:.3e}, weight={self.weights[name]:.3e}")
    
    def get_weights(self) -> Dict[str, float]:
        """Return current weights as dictionary."""
        return self.weights.copy()
    
    def get_weight(self, loss_name: str) -> float:
        """Get weight for a specific loss term."""
        return self.weights.get(loss_name, 1.0)
    
    def reset(self):
        """Reset to initial weights."""
        self.weights = self.initial_weights.copy()
        self.loss_ema = {name: None for name in self.loss_names}
        self.initial_losses = None
        self.epoch = 0
        self.initialized = False
        log.info("Adaptive weights reset to initial values")
    
    def state_dict(self) -> dict:
        """Return state for checkpointing."""
        return {
            'weights': self.weights,
            'loss_ema': self.loss_ema,
            'initial_losses': self.initial_losses,
            'epoch': self.epoch,
            'initialized': self.initialized,
        }
    
    def load_state_dict(self, state: dict):
        """Load state from checkpoint."""
        self.weights = state['weights']
        self.loss_ema = state['loss_ema']
        self.initial_losses = state['initial_losses']
        self.epoch = state['epoch']
        self.initialized = state['initialized']
        log.info(f"Loaded adaptive weights state from epoch {self.epoch}")


def create_adaptive_weights_from_config(cfg) -> Optional[AdaptiveLossWeights]:
    """
    Create AdaptiveLossWeights instance from Hydra config.
    
    Args:
        cfg: Hydra config with training.loss section
        
    Returns:
        AdaptiveLossWeights instance if adaptive weighting is enabled, None otherwise
    """
    if not cfg.training.loss.get('adaptive', {}).get('enabled', False):
        return None
    
    # Define loss names
    loss_names = ['PI_loss1', 'PI_loss2']
    if cfg.training.loss.get('use_rk4', False):
        loss_names.extend(['RK4_loss1', 'RK4_loss2'])
    if cfg.training.loss.get('use_energy', False):
        loss_names.append('Energy_loss')
    
    # Initial weights from config
    initial_weights = {
        'PI_loss1': cfg.training.loss.w1_PI,
        'PI_loss2': cfg.training.loss.w2_PI,
    }
    if cfg.training.loss.get('use_rk4', False):
        initial_weights['RK4_loss1'] = cfg.training.loss.w1_rk4
        initial_weights['RK4_loss2'] = cfg.training.loss.w2_rk4
    if cfg.training.loss.get('use_energy', False):
        initial_weights['Energy_loss'] = cfg.training.loss.w_energy
    
    # Adaptive parameters
    adaptive_cfg = cfg.training.loss.adaptive
    strategy = adaptive_cfg.get('strategy', 'equal_init')
    ema_alpha = adaptive_cfg.get('ema_alpha', 0.9)
    update_frequency = adaptive_cfg.get('update_frequency', 1)
    
    return AdaptiveLossWeights(
        loss_names=loss_names,
        initial_weights=initial_weights,
        strategy=strategy,
        ema_alpha=ema_alpha,
        update_frequency=update_frequency,
    )
