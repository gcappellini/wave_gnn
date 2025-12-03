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

from paddle import grad
import torch
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import os

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
        initial_weights: Dict[str, float],
        strategy: str = 'equal_init',
        ema_alpha: float = 0.9,
        update_frequency: int = 1,
        eps: float = 1e-8,
        min_weight: float = 1e-4,
        max_weight: float = 1e4,
        ntk_args=None,
        ntk_update_freq: int = 100,
    ):
        self.loss_names = list(initial_weights.keys())
        self.strategy = strategy
        self.ema_alpha = ema_alpha
        self.update_frequency = update_frequency
        self.eps = eps
        self.min_weight = min_weight
        self.max_weight = max_weight
        # NTK-specific
        self.ntk_args = ntk_args
        self.ntk_update_freq = ntk_update_freq

        # Initialize weights from config
        self.weights = {name: initial_weights.get(name, 1.0) for name in self.loss_names}
        self.initial_weights = self.weights.copy()

        # Tracking variables
        self.loss_ema = {name: None for name in self.loss_names}
        self.initial_losses = None
        self.epoch = 0
        self.initialized = False

        self.ntk_traces_log = []  # List of (trace_pde, trace_ic_u, trace_ic_v)
        self.weights_log = [] 

        log.info(f"Adaptive loss weights initialized with strategy: {strategy}")
        log.info(f"Initial weights: {self.weights}")
    
    def update(self, epoch: int, context: dict = None) -> Dict[str, float]:
        """
        Update weights based on current loss values and strategy.
        
        Args:
            epoch: Current epoch number (1-indexed)
            loss_values: Dictionary mapping loss names to their current values
            ntk_args: Optional dict of arguments for NTK computation (overrides self.ntk_args)
        Returns:
            Updated weights dictionary
        """
        self.epoch = epoch
        loss_values = context.get('loss_values', {})
        
        # Filter out invalid losses
        valid_losses = {k: v for k, v in loss_values.items() 
                       if k in self.loss_names and not (np.isnan(v) or np.isinf(v))}
        
        if not valid_losses:
            log.warning(f"No valid loss values at epoch {epoch}, keeping current weights")
            self.weights_log.append(self.weights.copy())
            return self.weights
        
        # Strategy: fixed weights (no adaptation)
        if self.strategy == 'fixed':
            self.weights_log.append(self.weights.copy())
            return self.weights
        
        # Strategy: equal initialization at epoch 1
        if self.strategy == 'equal_init' and epoch == 1:
            self._equalize_losses(valid_losses)
            self.initialized = True
            self.weights_log.append(self.weights.copy())
            return self.weights
        
        # Strategy: equal init + EMA
        if self.strategy == 'equal_init_ema':
            if epoch == 1:
                self._equalize_losses(valid_losses)
                self.initialized = True
            elif epoch > 1 and epoch % self.update_frequency == 0:
                self._update_ema_weights(valid_losses)
            self.weights_log.append(self.weights.copy())
            return self.weights
        
        # Strategy: pure EMA (no equal init)
        if self.strategy == 'ema':
            if epoch % self.update_frequency == 0:
                self._update_ema_weights(valid_losses)
            self.weights_log.append(self.weights.copy())
            return self.weights

        # Strategy: NTK-based weights
        if self.strategy == 'ntk':
            if epoch % self.ntk_update_freq == 0:
                K_pde, K_ic_u, K_ic_v = self._compute_ntk_matrices(context)
                weights_arr, _ = self._compute_ntk_weights_from_matrices(K_pde, K_ic_u, K_ic_v)
                for i, name in enumerate(self.loss_names):
                    self.weights[name] = float(weights_arr[i])
            self.weights_log.append(self.weights.copy())
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


    def _compute_ntk_weights_from_matrices(self, K_pde, K_ic_u, K_ic_v):
        """
        Compute adaptive weights from full NTK matrices.
        Uses the trace of each matrix for weight computation.
        
        Args:
            K_pde: NTK Gram matrix for PDE residual (n_pde x n_pde)
            K_ic_u: NTK Gram matrix for ic_u (n_ic x n_ic)
            K_ic_v: NTK Gram matrix for ic_v (n_ic x n_ic)
        
        Returns:
            weights: [lambda_pde, lambda_ic_u, lambda_ic_v]
            traces: [trace_pde, trace_ic_u, trace_ic_v]
        """
        # Compute traces
        trace_pde = np.trace(K_pde)
        trace_ic_u = np.trace(K_ic_u)
        trace_ic_v = np.trace(K_ic_v)
        
        # Total trace
        total_trace = trace_pde + trace_ic_u + trace_ic_v
        
        # Compute adaptive weights using Algorithm 1 formula
        lambda_pde = total_trace / (trace_pde + 1e-10)
        lambda_ic_u = total_trace / (trace_ic_u + 1e-10)
        lambda_ic_v = total_trace / (trace_ic_v + 1e-10)
        
        weights = np.array([lambda_pde, lambda_ic_u, lambda_ic_v])
        traces = np.array([trace_pde, trace_ic_u, trace_ic_v])

        self.ntk_traces_log.append((trace_pde, trace_ic_u, trace_ic_v))
        
        return weights, traces


    def _compute_ntk_matrices(self, context: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute full NTK Gram matrices for the current batch and log their eigenvalues for advanced analysis.
        Supports both 1D (xt_colloc, xt_ic) and 2D (xyt_colloc, xyt_ic) naming conventions.
        
        Args:
            model: the PINN model (self)
            xt_colloc/xyt_colloc: torch.Tensor, (n_colloc, 2) or (n_colloc, 3) collocation points
            xt_ic/xyt_ic: torch.Tensor, (n_ic, 2) or (n_ic, 3) initial condition points
            u0_sensors, v0_sensors, src_sensors: torch.Tensor, sensors for IC and source
            src_colloc: torch.Tensor, source values at collocation points
            a_coeffs, b_coeffs: Tensor, IC coefficient arrays
        Returns:
            K_pde, K_ic_u, K_ic_v: NTK Gram matrices
        """
        model = context['model']
        # Handle both 1D (xt_*) and 2D (xyt_*) naming conventions
        xt_colloc = context['xyt_colloc'] if 'xyt_colloc' in context else context['xt_colloc']
        xt_ic = context['xyt_ic'] if 'xyt_ic' in context else context['xt_ic']
        u0_sensors = context['u0_sensors']
        v0_sensors = context['v0_sensors']
        src_sensors = context['src_sensors']
        src_colloc = context['src_colloc']
        a_coeffs = context['a_coeffs']
        b_coeffs = context['b_coeffs']

        # Get trainable parameters
        params = [p for p in model.parameters() if p.requires_grad]
        num_params = sum(p.numel() for p in params)
        n_colloc = xt_colloc.shape[0]
        n_ic = xt_ic.shape[0]
        print(f"  [NTK] Computing NTK matrices: {n_colloc} PDE pts, {n_ic} IC pts, {num_params} params")

        # --- PDE Jacobian ---
        J_pde = np.zeros((n_colloc, num_params))
        for i in range(n_colloc):
            xt = xt_colloc[i:i+1].clone().detach().requires_grad_(True)
            # Use current sensors and source for this batch
            residual = model.compute_pde_residual(u0_sensors, v0_sensors, src_sensors, xt, src_colloc[i:i+1])
            residual_scalar = residual.sum()
            grads = torch.autograd.grad(
                outputs=residual_scalar,
                inputs=params,
                create_graph=False,
                retain_graph=False,
                allow_unused=True
            )
            grad_vec = []
            for g in grads:
                if g is not None:
                    grad_vec.append(g.detach().cpu().numpy().flatten())
                else:
                    grad_vec.append(np.zeros(0))
            J_pde[i, :] = np.concatenate(grad_vec)

        # --- IC_u Jacobian ---
        J_ic_u = np.zeros((n_ic, num_params))
        # For 2D models, pass both x and y coordinates
        u_ic_true = model.generate_ic_sine_series(a_coeffs, xt_ic[:, 0], xt_ic[:, 1]) if xt_ic.shape[1] >= 2 else model.generate_ic_sine_series(a_coeffs, xt_ic[:, 0])
        for i in range(n_ic):
            xt = xt_ic[i:i+1].clone().detach().requires_grad_(True)
            u_pred = model.forward(u0_sensors, v0_sensors, src_sensors, xt)
            loss = (u_pred - u_ic_true[i]).sum()
            grads = torch.autograd.grad(
                outputs=loss,
                inputs=params,
                create_graph=False,
                retain_graph=False,
                allow_unused=True
            )
            grad_vec = []
            for g in grads:
                if g is not None:
                    grad_vec.append(g.detach().cpu().numpy().flatten())
                else:
                    grad_vec.append(np.zeros(0))
            J_ic_u[i, :] = np.concatenate(grad_vec)

        # --- IC_v Jacobian ---
        J_ic_v = np.zeros((n_ic, num_params))
        # For 2D models, pass both x and y coordinates
        v_ic_true = model.generate_ic_sine_series(b_coeffs, xt_ic[:, 0], xt_ic[:, 1]) if xt_ic.shape[1] >= 2 else model.generate_ic_sine_series(b_coeffs, xt_ic[:, 0])
        for i in range(n_ic):
            xt = xt_ic[i:i+1].clone().detach().requires_grad_(True)
            u_pred = model.forward(u0_sensors, v0_sensors, src_sensors, xt)
            grad_xt = torch.autograd.grad(u_pred, xt, torch.ones_like(u_pred), create_graph=True, retain_graph=True)[0]
            u_t_pred = grad_xt[:, 1]  # shape [1]
            loss = (u_t_pred - v_ic_true[i]).sum()
            grads = torch.autograd.grad(
                outputs=loss,
                inputs=params,
                create_graph=False,
                retain_graph=False,
                allow_unused=True
            )
            grad_vec = []
            for g in grads:
                if g is not None:
                    grad_vec.append(g.detach().cpu().numpy().flatten())
                else:
                    grad_vec.append(np.zeros(0))
            J_ic_v[i, :] = np.concatenate(grad_vec)

        # Gram matrices
        K_pde = J_pde @ J_pde.T
        K_ic_u = J_ic_u @ J_ic_u.T
        K_ic_v = J_ic_v @ J_ic_v.T

        # --- Eigenvalue logging for advanced NTK analysis ---
        # Compute and store eigenvalues (sorted descending) for each matrix
        eigvals_pde = np.sort(np.linalg.eigvalsh(K_pde))[::-1]
        eigvals_ic_u = np.sort(np.linalg.eigvalsh(K_ic_u))[::-1]
        eigvals_ic_v = np.sort(np.linalg.eigvalsh(K_ic_v))[::-1]

        # Initialize log if not present
        if not hasattr(self, 'ntk_eigvals_log') or self.ntk_eigvals_log is None:
            self.ntk_eigvals_log = []
        self.ntk_eigvals_log.append({
            "K_pde": eigvals_pde,
            "K_ic_u": eigvals_ic_u,
            "K_ic_v": eigvals_ic_v
        })

        return K_pde, K_ic_u, K_ic_v
    
    def plot_ntk_traces(self, output_dir):
        # Only plot eigenvalue spectra; require eigenvalue log to be present
        if not hasattr(self, 'ntk_eigvals_log') or not self.ntk_eigvals_log:
            raise RuntimeError("No NTK eigenvalue log found. Please ensure eigenvalues are logged during training.")
        n_steps = len(self.ntk_eigvals_log)
        idxs = [0]
        if n_steps > 1:
            idxs.append(n_steps // 2)
        if n_steps > 2:
            idxs.append(n_steps - 1)
        labels = [f'step={i}' for i in idxs]
        plt.figure(figsize=(18, 5))
        for j, key in enumerate(["K_pde", "K_ic_u", "K_ic_v"]):
            plt.subplot(1, 3, j+1)
            for idx, label in zip(idxs, labels):
                eigvals = self.ntk_eigvals_log[idx][key]
                plt.plot(range(1, len(eigvals)+1), eigvals, label=label)
            plt.xlabel('Eigenvalue index')
            plt.ylabel('Eigenvalue')
            plt.xscale('log')
            plt.yscale('log')
            plt.title(key)
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'NTK_eigenvalues.png'))
        plt.close()

    def plot_weights_evolution(self, output_dir):
        weights_arr = {name: [] for name in self.loss_names}
        for wdict in self.weights_log:
            for name in self.loss_names:
                weights_arr[name].append(wdict[name])
        plt.figure(figsize=(8, 5))
        for name in self.loss_names:
            plt.plot(weights_arr[name], label=name)
        plt.xlabel('Update step')
        plt.ylabel('Weight')
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Weights_evolution.png'))
        plt.close()
    


