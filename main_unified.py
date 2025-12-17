"""
Simplified main script using unified training approach.
This replaces the complex multi-case logic with a single, clean training pipeline.
"""

import torch
import os
import logging
from datetime import datetime
from hydra import main as hydra_main
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from model_2d import PINNDeepONet_Wave2D
from train_unified import UnifiedTrainer
from plot_2d import plot_solution_2d_comparison
import torch.cuda

log = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


@hydra_main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Unified training pipeline with scheduled loss weighting.
    """
    log.info("=" * 70)
    log.info("UNIFIED TRAINING PIPELINE")
    log.info("=" * 70)
    log.info(f"\n{OmegaConf.to_yaml(cfg)}\n")
    
    # Get output directory from Hydra
    hydra_cfg = HydraConfig.get()
    output_fold = hydra_cfg.runtime.output_dir
    log.info(f"Output directory: {output_fold}\n")
    
    # ============================================================
    # Load Ground Truth Data
    # ============================================================
    gt_data = None
    # Load if available (optional for validation)
    
    # ============================================================
    # Create Model
    # ============================================================
    log.info("Creating model...")
    model = PINNDeepONet_Wave2D(
        n_sensors_ic=cfg.model.n_sensors_ic,
        n_sensors_src=cfg.model.n_sensors_src,
        branch_width=cfg.model.branch_width,
        trunk_width=cfg.model.trunk_width,
        branch_depth=cfg.model.branch_depth,
        trunk_depth=cfg.model.trunk_depth,
        p=cfg.model.p,
        wave_speed=cfg.model.wave_speed,
        damping_coeff=cfg.model.damping_coeff,
        use_fft_trunk=cfg.model.use_fft_trunk,
        fft_trunk_args=cfg.model.fft_trunk_args
    )
    model = model.to(DEVICE)
    log.info(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters\n")
    
    # ============================================================
    # Load Pretrained Model if Specified
    # ============================================================
    if cfg.run.get('load_pretrain', False):
        load_pretrain_from = cfg.run.get('load_pretrain_from', None)
        if load_pretrain_from:
            pretrain_path = f"outputs/{load_pretrain_from[0]}/{load_pretrain_from[1]}/model_pretrain.pth"
            if os.path.exists(pretrain_path):
                checkpoint = torch.load(pretrain_path, map_location=DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(DEVICE)
                log.info(f"✓ Loaded pretrained Phase 1 model from {pretrain_path}")
                log.info(f"  Test Metric: {checkpoint.get('test_metric', 'N/A')}\n")
            else:
                log.error(f"Pretrain checkpoint not found: {pretrain_path}")
                return
    
    # ============================================================
    # Generate Test Cases
    # ============================================================
    log.info("Generating test cases...")
    test_cases = []
    n_test = cfg.training.get('n_test_cases', 10)
    for i in range(n_test):
        test_cases.append({
            'a_coeffs': torch.randn(1, 2, device=DEVICE) * 0.2,
            'b_coeffs': torch.randn(1, 2, device=DEVICE) * 0.1,
            'center_x': 0.5,
            'center_y': 0.5
        })
    log.info(f"✓ Generated {n_test} test cases\n")
    
    # ============================================================
    # Train with Unified Approach
    # ============================================================
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
    # Validation Plots
    # ============================================================
    log.info("Generating validation plots...")
    try:
        fig_solution, metrics = plot_solution_2d_comparison(
            model, a_test=0.5, b_test=0.0, source_type='zero',
            source_amplitude=7.5, center_x=0.5, center_y=0.5, T_max=2.0,
            gt_data=gt_data
        )
        plot_path = os.path.join(output_fold, 'solution_comparison.png')
        fig_solution.savefig(plot_path, dpi=150, bbox_inches='tight')
        log.info(f"✓ Solution comparison plot saved\n")
    except Exception as e:
        log.warning(f"Could not generate solution plots: {e}\n")
    
    log.info("=" * 70)
    log.info("✓ TRAINING COMPLETE")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
