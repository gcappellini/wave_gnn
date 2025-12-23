"""
Simplified main script using unified training approach.
This replaces the complex multi-case logic with a single, clean training pipeline.
"""

import torch
import os
import logging
import pandas as pd
from datetime import datetime
from hydra import main as hydra_main
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from model_2d import PINNDeepONet_Wave2D
from model_2d_nosource import PINNDeepONet_Wave2D_NoSource
from train_unified import UnifiedTrainer
from plot_2d import (plot_solution_2d_comparison, plot_ic_reconstruction, 
                      plot_ic_v_reconstruction, plot_loss_history)
import torch.cuda

# Get script directory for absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

log = logging.getLogger(__name__)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def _build_checkpoint_path(date, time, model_file='model.pth', load_fold_type='outputs'):
    """Helper function to construct checkpoint path"""
    if '/' in time:
        load_fold = 'multirun'
    else:
        load_fold = load_fold_type
    return f'{load_fold}/{date}/{time}/{model_file}'


@hydra_main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Unified training pipeline with scheduled loss weighting.
    """
    # Get output directory from Hydra
    hydra_cfg = HydraConfig.get()
    output_fold = hydra_cfg.runtime.output_dir
    
    log.info("=" * 70)
    log.info("UNIFIED TRAINING PIPELINE")
    log.info("=" * 70)
    log.info(f"\n{OmegaConf.to_yaml(cfg)}\n")
    log.info(f"Output directory: {output_fold}\n")
    
    # ============================================================
    # Load Ground Truth Data
    # ============================================================
    TRAINING_CASE = "no_source"  if cfg.data.source_type == 'zero' else "with_source"
    gt_data = None
    gt_file = os.path.join(SCRIPT_DIR, f'data/gt_wave2D_{TRAINING_CASE}.csv')
    # Load if available (optional for validation)
    if os.path.exists(gt_file):
        gt_data = pd.read_csv(gt_file, header=None).values
        log.info(f"✓ Loaded ground truth data from {gt_file}")
    else:
        log.warning(f"Ground truth file not found: {gt_file}")
    
    # ============================================================
    # Create Model
    # ============================================================
    log.info("Creating model...")
    
    # Use simplified model without source for no_source case
    if cfg.data.source_type == 'zero':
        model = PINNDeepONet_Wave2D_NoSource(cfg)
        log.info("✓ Using simplified model (NO SOURCE)")
    else:
        model = PINNDeepONet_Wave2D(cfg)
        log.info("✓ Using full model (WITH SOURCE)")
    
    model = model.to(DEVICE)
    log.info(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters\n")
    
    # ============================================================
    # Load Pretrained Trunk (optional)
    # ============================================================
    if cfg.run.get('load_pretrained_trunk', False) and not cfg.run.get('load_model', False):
        trunk_path = cfg.run.get('pretrained_trunk_path', os.path.join(SCRIPT_DIR, 'data/pretrained_trunk_p64.pth'))
        if not os.path.isabs(trunk_path):
            trunk_path = os.path.join(SCRIPT_DIR, trunk_path)
        if not hasattr(model, 'trunk'):
            log.error("Model does not expose a 'trunk' attribute; cannot load pretrained trunk.")
            return
        if os.path.exists(trunk_path):
            try:
                trunk_ckpt = torch.load(trunk_path, map_location=DEVICE)
                # Derive a usable state_dict for the trunk
                trunk_state = None
                if isinstance(trunk_ckpt, dict):
                    if 'trunk_state_dict' in trunk_ckpt:
                        trunk_state = trunk_ckpt['trunk_state_dict']
                    elif 'state_dict' in trunk_ckpt:
                        # Try to extract only trunk.* keys if present; else use as-is
                        sd = trunk_ckpt['state_dict']
                        filtered = {k.replace('trunk.', ''): v for k, v in sd.items() if k.startswith('trunk.')}
                        trunk_state = filtered if len(filtered) > 0 else sd
                    else:
                        # Assume the dict is directly a state_dict
                        trunk_state = trunk_ckpt
                else:
                    # Assume the checkpoint is directly a state_dict
                    trunk_state = trunk_ckpt

                missing, unexpected = model.trunk.load_state_dict(trunk_state, strict=False)
                log.info(f"✓ Trunk loaded from {trunk_path} | missing: {len(missing)}, unexpected: {len(unexpected)}")

                if cfg.run.get('freeze_trunk', True):
                    for p in model.trunk.parameters():
                        p.requires_grad = False
                    log.info("Trunk loaded from SVD pre-training and FROZEN.")
                else:
                    log.info("Trunk loaded from SVD pre-training (not frozen).")
            except Exception as e:
                log.error(f"Failed to load pretrained trunk from {trunk_path}: {e}")
                return
        else:
            log.error(f"Pretrained trunk file not found: {trunk_path}")
            return
    
    # ============================================================
    # Load Model (if specified)
    # ============================================================
    skip_training = False
    
    if cfg.run.get('load_model', False):
        load_from = cfg.run.get('load_from', None)
        if load_from:
            model_path = _build_checkpoint_path(load_from[0], load_from[1], 
                                               model_file='model_final.pth')
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(DEVICE)
                log.info(f"✓ Loaded model from {model_path}\n")
                
                # Check if we should skip training
                resume_training = cfg.run.get('resume_training', False)
                if not resume_training:
                    log.info("resume_training=False → Skipping training, going directly to validation\n")
                    skip_training = True
                else:
                    log.info("resume_training=True → Continuing training from checkpoint\n")
            else:
                log.error(f"Model checkpoint not found: {model_path}")
                return
        else:
            log.error("load_model=True but load_from path not specified")
            return
    
    # ============================================================
    # Generate Test Cases
    # ============================================================
    log.info("Generating test cases...")
    test_cases = []
    n_test = cfg.training.get('n_test_cases', 10)
    n_ic_u = cfg.data.get('n_ic_u', 2)
    n_ic_v = cfg.data.get('n_ic_v', 2)
    ic_mode = cfg.data.get('ic_coeff_mode', '1d')  # '1d' (diagonal) or '2d' (full KxL like MATLAB)
    
    for i in range(n_test):
        if ic_mode == '2d':
            # Full 2D coefficients with decay 1/(k^2 + l^2), consistent with MATLAB make_coeffs
            k_idx = torch.arange(1, n_ic_u + 1, device=DEVICE).view(-1, 1)
            l_idx = torch.arange(1, n_ic_v + 1, device=DEVICE).view(1, -1)
            decay = 1.0 / (k_idx.float()**2 + l_idx.float()**2)
            a_coeff = (-1 + 2 * torch.rand(n_ic_u, n_ic_v, device=DEVICE)) * decay
            b_coeff = (-1 + 2 * torch.rand(n_ic_u, n_ic_v, device=DEVICE)) * decay
        else:
            # Diagonal/symmetric 1D mode (sum_k coeff[k] sin(kπx) sin(kπy))
            a_coeff = torch.randn(n_ic_u, device=DEVICE) * 0.2
            b_coeff = torch.randn(n_ic_v, device=DEVICE) * 0.1

        test_cases.append({
            'a_coeffs': a_coeff,
            'b_coeffs': b_coeff,
            'center_x': 0.5,
            'center_y': 0.5
        })
    log.info(f"✓ Generated {n_test} test cases\n")
    
    # ============================================================
    # Train with Unified Approach (or Skip if Load Model)
    # ============================================================
    if skip_training:
        log.info("=" * 70)
        log.info("SKIPPING TRAINING (loaded model with resume_training=False)")
        log.info("=" * 70 + "\n")
        history = {}
        training_time = None
    else:
        log.info("Starting unified training...")
        start_time = datetime.now()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        trainer = UnifiedTrainer(model, cfg, device=DEVICE)
        history = trainer.train(test_cases, output_fold)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = datetime.now()
        training_time = end_time - start_time
        log.info(f"Training completed in {training_time}\n")
        
        # ============================================================
        # Save Final Model
        # ============================================================
        model_path = os.path.join(output_fold, 'model_final.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
            'training_time': str(training_time)
        }, model_path)
        log.info(f"✓ Final model saved to {model_path}\n")
    
    # ============================================================
    # Validation Plots (3-part validation)
    # ============================================================
    log.info("Generating validation plots...")
    
    # Create a test case for validation plots
    a_test = torch.tensor(0.5, device=DEVICE)
    b_test = torch.tensor(0.0, device=DEVICE)
    test_case_val = {
        'a_coeffs': a_test.unsqueeze(0) if a_test.dim() == 0 else a_test,
        'b_coeffs': b_test.unsqueeze(0) if b_test.dim() == 0 else b_test
    }
    
    # 1. IC_U Reconstruction (Displacement)
    try:
        log.info("\n[1/4] Validating Initial Condition (Displacement)...")
        ic_u_path = os.path.join(output_fold, 'ic_u_reconstruction.png')
        plot_ic_reconstruction(model, test_case_val, n_grid=100, save_path=ic_u_path, gt_data=gt_data)
        log.info(f"✓ IC_U validation plot saved\n")
    except Exception as e:
        log.warning(f"Could not generate IC_U validation plot: {e}\n")
    
    # 2. IC_V Reconstruction (Velocity)
    try:
        log.info("[2/4] Validating Initial Condition (Velocity)...")
        ic_v_path = os.path.join(output_fold, 'ic_v_reconstruction.png')
        plot_ic_v_reconstruction(model, test_case_val, n_grid=100, save_path=ic_v_path, gt_data=gt_data)
        log.info(f"✓ IC_V validation plot saved\n")
    except Exception as e:
        log.warning(f"Could not generate IC_V validation plot: {e}\n")
    
    # 3. Solution Comparison (Space-Time Evolution)
    try:
        log.info("[3/4] Validating Space-Time Solution...")
        result = plot_solution_2d_comparison(
            model, a_test=0.5, b_test=0.0, source_type='zero',
            source_amplitude=7.5, center_x=0.5, center_y=0.5, T_max=2.0,
            gt_data=gt_data
        )
        # Handle both (fig, metrics) and fig returns
        if isinstance(result, tuple):
            fig_solution = result[0]
        else:
            fig_solution = result
        
        plot_path = os.path.join(output_fold, 'solution_comparison.png')
        fig_solution.savefig(plot_path, dpi=150, bbox_inches='tight')
        log.info(f"✓ Solution comparison plot saved\n")
    except Exception as e:
        log.warning(f"Could not generate solution comparison plot: {e}\n")
    
    # 4. Loss History
    try:
        log.info("[4/4] Plotting Training Loss History...")
        if history:
            loss_path = os.path.join(output_fold, 'loss_history.png')
            plot_loss_history(history, save_path=loss_path)
            log.info(f"✓ Loss history plot saved\n")
    except Exception as e:
        log.warning(f"Could not generate loss history plot: {e}\n")
    
    log.info("=" * 70)
    log.info("✓ TRAINING COMPLETE")
    log.info("=" * 70)
        


if __name__ == "__main__":
    main()
