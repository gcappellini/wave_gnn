"""
Unified training script with scheduled loss weighting.
Replaces phase1, phase1b, phase2, etc. with a single configurable training loop.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import os
import pickle
import copy
from datetime import datetime
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def sample_coeffs(n, range_tuple, p=3.0, as_2d=False, device='cpu'):
    """Sample IC coefficients from Beta distribution."""
    a, b = range_tuple
    # Beta(p, p) is symmetric around 0.5, scaled to [a, b]
    coeffs = torch.from_numpy(np.random.beta(p, p, size=n)).float().to(device)
    coeffs = a + (b - a) * coeffs
    if as_2d:
        # Reshape to 2D for 2-coefficient IC (2 sin basis functions)
        coeffs = coeffs.reshape(-1, 2)
    return coeffs


class UnifiedTrainer:
    """Unified trainer with scheduled loss weighting."""
    
    def __init__(self, model, cfg, device='cpu'):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.training_history = self._init_history()
        
    def _init_history(self):
        """Initialize tracking dictionary."""
        return {
            'epoch': [],
            'stage': [],
            'loss_total': [],
            'loss_ic_u': [],
            'loss_ic_v': [],
            'loss_pde': [],
            'w_ic_u': [],
            'w_ic_v': [],
            'w_pde': [],
            'lr': [],
            'val_epochs': [],
            'val_loss_ic_u': [],
            'val_loss_ic_v': [],
            'val_loss_pde': [],
            'val_metric': []
        }
    
    def train(self, test_cases, output_fold):
        """
        Unified training loop with loss weight scheduling.
        
        Args:
            test_cases: list of test case dictionaries
            output_fold: output directory for checkpoints
        """
        log.info("=" * 70)
        log.info("UNIFIED TRAINING WITH SCHEDULED LOSS WEIGHTING")
        log.info("=" * 70)
        
        loss_schedule = self.cfg.training.loss_schedule
        if not loss_schedule.enabled:
            log.error("Loss scheduling not enabled in config!")
            return None
        
        stages = loss_schedule.stages
        total_epochs = sum(stage.epochs for stage in stages)
        log.info(f"Total epochs: {total_epochs} across {len(stages)} stages\n")
        
        # Extract config
        a_range = tuple(self.cfg.data.a_range)
        b_range = tuple(self.cfg.data.b_range)
        n_ic = self.cfg.data.get('n_ic', 64)
        n_ic_u = self.cfg.data.get('n_ic_u', 16)
        n_ic_v = self.cfg.data.get('n_ic_v', 16)
        n_batches = self.cfg.training.get('n_batches', 4)
        batch_size = self.cfg.training.get('batch_size', 8)
        val_interval = self.cfg.training.get('val_interval', 20)
        max_grad_norm = self.cfg.training.get('max_grad_norm', 1.0)
        
        # Setup IC grid (shared across all stages)
        n_ic_grid = int(np.sqrt(n_ic))
        x_ic_1d = torch.linspace(self.model.domain[0], self.model.domain[1], n_ic_grid, device=self.device)
        y_ic_1d = torch.linspace(self.model.domain[0], self.model.domain[1], n_ic_grid, device=self.device)
        X_ic, Y_ic = torch.meshgrid(x_ic_1d, y_ic_1d, indexing='ij')
        x_ic_flat = X_ic.flatten()
        y_ic_flat = Y_ic.flatten()
        
        best_model_state = None
        best_metric = float('inf')
        global_epoch = 0
        
        # ===== STAGE LOOP =====
        for stage_idx, stage in enumerate(stages):
            log.info(f"\n{'='*70}")
            log.info(f"STAGE {stage_idx + 1}/{len(stages)}: {stage.name}")
            log.info(f"Epochs: {stage.epochs} | LR: {stage.lr:.2e}")
            log.info(f"Weights -> IC_u: {stage.weights.w_ic_u:.2f}, IC_v: {stage.weights.w_ic_v:.2f}, PDE: {stage.weights.w_pde:.2f}")
            log.info(f"{'='*70}\n")
            
            # Setup optimizer for this stage
            optimizer = torch.optim.Adam(self.model.parameters(), lr=stage.lr)
            # ReduceLROnPlateau: reduce LR only when metric stops improving
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.cfg.training.get('lr_reduce_factor', 0.5),
                patience=self.cfg.training.get('lr_patience', 5),
                threshold=self.cfg.training.get('lr_threshold', 1e-4),
                cooldown=self.cfg.training.get('lr_cooldown', 0),
                min_lr=self.cfg.training.get('min_lr', 1e-6),
                verbose=False
            )
            
            # ===== EPOCH LOOP =====
            for epoch_in_stage in range(stage.epochs):
                self.model.train()
                loss_ic_u_accum = 0.0
                loss_ic_v_accum = 0.0
                loss_pde_accum = 0.0
                
                batch_size = self.cfg.training.get('batch_size', 8)
                
                # ===== BATCH LOOP =====
                for batch_idx in range(n_batches):
                    # Sample IC coefficients
                    a_coeffs_list = []
                    b_coeffs_list = []
                    
                    for _ in range(batch_size):
                        # Sample scalar or 1D coefficients, NOT 2D
                        a_coeff = torch.randn(self.cfg.data.get('n_ic_u', 2), device=self.device)
                        b_coeff = torch.randn(self.cfg.data.get('n_ic_v', 2), device=self.device)
                        a_coeffs_list.append(a_coeff)
                        b_coeffs_list.append(b_coeff)
                    
                    a_batch = torch.stack(a_coeffs_list)  # (batch_size, n_ic_u)
                    b_batch = torch.stack(b_coeffs_list)  # (batch_size, n_ic_v)
                    
                    # Clamp to valid ranges
                    a_batch = torch.clamp(a_batch, self.cfg.data.a_range[0], self.cfg.data.a_range[1])
                    b_batch = torch.clamp(b_batch, self.cfg.data.b_range[0], self.cfg.data.b_range[1])
                    
                    # Generate IC fields (sensors)
                    u0_sensors = self.model.generate_ic_sine_series(a_batch)
                    v0_sensors = self.model.generate_ic_sine_series(b_batch)
                    src_sensors = self.model.generate_source(
                        self.cfg.data.source_type,
                        self.cfg.data.source_amplitude,
                        self.cfg.data.center_x,
                        self.cfg.data.center_y
                    )
                    
                    # Setup IC evaluation points (t=0)
                    t_ic = torch.zeros_like(x_ic_flat)
                    xyt_ic = torch.stack([x_ic_flat, y_ic_flat, t_ic], dim=-1)
                    xyt_ic.requires_grad_(True)
                    
                    # Extract weights for this stage
                    w_ic_u = stage.weights.w_ic_u
                    w_ic_v = stage.weights.w_ic_v
                    w_pde = stage.weights.w_pde
                    
                    # IC losses - only compute if weight > 0
                    loss_ic_u = torch.tensor(0.0, device=self.device)
                    loss_ic_v = torch.tensor(0.0, device=self.device)
                    
                    if w_ic_u > 0 or w_ic_v > 0:
                        # Forward pass at IC
                        u0_pred = self.model.forward(u0_sensors, v0_sensors, src_sensors, xyt_ic)
                        
                        # Compute true IC values
                        u0_true = self.model.generate_ic_sine_series(a_batch, x_ic_flat, y_ic_flat)
                        
                        # Displacement loss (if needed)
                        if w_ic_u > 0:
                            loss_ic_u = torch.mean((u0_pred - u0_true) ** 2)
                        
                        # Velocity loss (if needed)
                        if w_ic_v > 0:
                            v0_pred = self.model.get_velocity(u0_sensors, v0_sensors, src_sensors, xyt_ic)
                            v0_true = self.model.generate_ic_sine_series(b_batch, x_ic_flat, y_ic_flat)
                            loss_ic_v = torch.mean((v0_pred - v0_true) ** 2)
                    
                    # PDE loss (if stage requires it)
                    loss_pde = torch.tensor(0.0, device=self.device)
                    if w_pde > 0:
                        # Sample collocation points for PDE
                        n_colloc = self.cfg.data.get('n_colloc', 256)
                        x_colloc = self.model.domain[0] + (self.model.domain[1] - self.model.domain[0]) * torch.rand(n_colloc, device=self.device)
                        y_colloc = self.model.domain[0] + (self.model.domain[1] - self.model.domain[0]) * torch.rand(n_colloc, device=self.device)
                        t_colloc = torch.rand(n_colloc, device=self.device) * self.cfg.data.T_max
                        xyt_colloc = torch.stack([x_colloc, y_colloc, t_colloc], dim=-1)
                        
                        # Compute source values AT COLLOCATION POINTS (not at sensor grid)
                        if self.cfg.data.source_type == 'gaussian':
                            # Gaussian source centered at (center_x, center_y)
                            cx, cy = self.cfg.data.center_x, self.cfg.data.center_y
                            sigma = 0.1
                            src_values_colloc = (self.cfg.data.source_amplitude * 
                                               torch.exp(-((x_colloc - cx)**2 + (y_colloc - cy)**2) / (2 * sigma**2)))
                        else:
                            # Zero source
                            src_values_colloc = torch.zeros(n_colloc, device=self.device)
                        
                        # Compute PDE residual (using existing method)
                        loss_pde = self.model.compute_pde_residual(
                            u0_sensors, v0_sensors, src_sensors,
                            xyt_colloc, src_values_colloc
                        ).mean()
                    
                    # Weighted loss
                    loss_total = w_ic_u * loss_ic_u + w_ic_v * loss_ic_v + w_pde * loss_pde
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss_total.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    
                    # Accumulate
                    loss_ic_u_accum += loss_ic_u.item() / n_batches
                    loss_ic_v_accum += loss_ic_v.item() / n_batches
                    loss_pde_accum += loss_pde.item() / n_batches
                
                # Accumulated total loss for logging
                loss_total_accum = w_ic_u * loss_ic_u_accum + w_ic_v * loss_ic_v_accum + w_pde * loss_pde_accum
                
                # Log to history
                self.training_history['epoch'].append(global_epoch)
                self.training_history['stage'].append(stage.name)
                self.training_history['loss_total'].append(loss_total_accum)
                self.training_history['loss_ic_u'].append(loss_ic_u_accum)
                self.training_history['loss_ic_v'].append(loss_ic_v_accum)
                self.training_history['loss_pde'].append(loss_pde_accum)
                self.training_history['w_ic_u'].append(w_ic_u)
                self.training_history['w_ic_v'].append(w_ic_v)
                self.training_history['w_pde'].append(w_pde)
                self.training_history['lr'].append(current_lr)
                
                # Validation
                if (global_epoch + 1) % val_interval == 0:
                    val_ic_u, val_ic_v, val_pde, val_metric = self._validate(test_cases, stage)
                    self.training_history['val_epochs'].append(global_epoch)
                    self.training_history['val_loss_ic_u'].append(val_ic_u)
                    self.training_history['val_loss_ic_v'].append(val_ic_v)
                    self.training_history['val_loss_pde'].append(val_pde)
                    self.training_history['val_metric'].append(val_metric)
                    # Step LR scheduler based on validation metric (plateau)
                    lr_scheduler.step(val_metric)
                    
                    if val_metric < best_metric:
                        best_metric = val_metric
                        best_model_state = copy.deepcopy(self.model.state_dict())
                    
                    log.info(f"Stage {stage_idx + 1} | Ep {epoch_in_stage + 1:3d}/{stage.epochs} | "
                            f"Loss: {loss_total_accum:.4e} | Val IC_u: {val_ic_u:.4e} | "
                            f"Val IC_v: {val_ic_v:.4e} | Val PDE: {val_pde:.4e}")
                elif (epoch_in_stage + 1) % 100 == 0:
                    log.info(f"Stage {stage_idx + 1} | Ep {epoch_in_stage + 1:3d}/{stage.epochs} | "
                            f"Loss: {loss_total_accum:.4e} | IC_u: {loss_ic_u_accum:.4e} | "
                            f"IC_v: {loss_ic_v_accum:.4e} | PDE: {loss_pde_accum:.4e}")
                
                # Record current LR after potential scheduler update
                current_lr = optimizer.param_groups[0]['lr']
                self.training_history['epoch'].append(global_epoch)
                self.training_history['stage'].append(stage.name)
                self.training_history['loss_total'].append(loss_total_accum)
                self.training_history['loss_ic_u'].append(loss_ic_u_accum)
                self.training_history['loss_ic_v'].append(loss_ic_v_accum)
                self.training_history['loss_pde'].append(loss_pde_accum)
                self.training_history['w_ic_u'].append(w_ic_u)
                self.training_history['w_ic_v'].append(w_ic_v)
                self.training_history['w_pde'].append(w_pde)
                self.training_history['lr'].append(current_lr)
                
                global_epoch += 1
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            log.info(f"\n✓ Restored best model (Val Metric: {best_metric:.6e})")
        
        # Save history with schedule
        history_file = os.path.join(output_fold, 'training_history_unified.pkl')
        os.makedirs(output_fold, exist_ok=True)
        with open(history_file, 'wb') as f:
            pickle.dump({
                'history': self.training_history,
                'schedule': [dict(s) for s in stages],
                'best_metric': best_metric
            }, f)
        log.info(f"Training history saved to {history_file}\n")
        
        return self.training_history
    
    def _validate(self, test_cases, stage):
        """Validate model on test set, skipping zero-weight terms."""
        self.model.eval()
        val_ic_u = 0.0
        val_ic_v = 0.0
        val_pde = 0.0
        
        # Get weights for current stage
        w_ic_u = stage.weights.w_ic_u
        w_ic_v = stage.weights.w_ic_v
        w_pde = stage.weights.w_pde
        
        with torch.no_grad():
            for test_case in test_cases:
                # Generate test case
                u0_sensors = self.model.generate_ic_sine_series(test_case['a_coeffs'])
                v0_sensors = self.model.generate_ic_sine_series(test_case['b_coeffs'])
                src_sensors = self.model.generate_source(
                    self.cfg.data.source_type,
                    self.cfg.data.source_amplitude,
                    test_case['center_x'],
                    test_case['center_y']
                )
                
                # IC evaluation - only compute if weights > 0
                if w_ic_u > 0 or w_ic_v > 0:
                    n_ic_grid = int(np.sqrt(self.cfg.data.n_ic))
                    x_ic = torch.linspace(self.model.domain[0], self.model.domain[1], n_ic_grid, device=self.device)
                    y_ic = torch.linspace(self.model.domain[0], self.model.domain[1], n_ic_grid, device=self.device)
                    X_ic, Y_ic = torch.meshgrid(x_ic, y_ic, indexing='ij')
                    xyt_ic = torch.stack([X_ic.flatten(), Y_ic.flatten(), torch.zeros_like(X_ic.flatten())], dim=-1)
                    
                    # True IC values
                    u0_true = self.model.generate_ic_sine_series(test_case['a_coeffs'], X_ic.flatten(), Y_ic.flatten())
                    v0_true = self.model.generate_ic_sine_series(test_case['b_coeffs'], X_ic.flatten(), Y_ic.flatten())
                    
                    # Displacement IC
                    if w_ic_u > 0:
                        u0_pred = self.model.forward(u0_sensors, v0_sensors, src_sensors, xyt_ic)
                        val_ic_u += torch.mean((u0_pred - u0_true) ** 2).item()
                    
                    # Velocity IC
                    if w_ic_v > 0:
                        v0_pred = self.model.get_velocity(u0_sensors, v0_sensors, src_sensors, xyt_ic)
                        val_ic_v += torch.mean((v0_pred - v0_true) ** 2).item()
                
                # PDE evaluation - requires gradients
                if w_pde > 0:
                    with torch.enable_grad():
                        x_colloc = self.model.domain[0] + (self.model.domain[1] - self.model.domain[0]) * torch.rand(256, device=self.device)
                        y_colloc = self.model.domain[0] + (self.model.domain[1] - self.model.domain[0]) * torch.rand(256, device=self.device)
                        t_colloc = torch.rand(256, device=self.device) * self.cfg.data.T_max
                        xyt_colloc = torch.stack([x_colloc, y_colloc, t_colloc], dim=-1)
                        xyt_colloc.requires_grad_(True)  # Enable gradients for PDE residual computation
                        
                        # Compute source values AT COLLOCATION POINTS
                        if self.cfg.data.source_type == 'gaussian':
                            cx, cy = test_case['center_x'], test_case['center_y']
                            sigma = 0.1
                            src_values_colloc = (self.cfg.data.source_amplitude * 
                                               torch.exp(-((x_colloc - cx)**2 + (y_colloc - cy)**2) / (2 * sigma**2)))
                        else:
                            src_values_colloc = torch.zeros(256, device=self.device)
                        
                        val_pde += self.model.compute_pde_residual(u0_sensors, v0_sensors, src_sensors, xyt_colloc, src_values_colloc).mean().item()
        
        n_test = len(test_cases)
        val_ic_u /= n_test
        val_ic_v /= n_test
        val_pde /= n_test
        val_metric = val_ic_u + val_ic_v + val_pde  # Combined metric
        
        return val_ic_u, val_ic_v, val_pde, val_metric
