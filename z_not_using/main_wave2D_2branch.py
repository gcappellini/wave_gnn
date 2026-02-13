import torch
import pandas as pd
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

# Setup logging
log = logging.getLogger(__name__)

# Get script directory for absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# GPU SETUP AND OPTIMIZATION
# ============================================================
# Detect device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

@hydra_main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # ============================================================
    # LOGGING
    # ============================================================
    log.info("=" * 70)
    log.info("CONFIGURATION")
    log.info("=" * 70)
    log.info(f"\n{OmegaConf.to_yaml(cfg)}\n")
    
    # ============================================================
    # EXTRACT CONFIGURATION FROM HYDRA
    # ============================================================
    
    # Get output folder from Hydra (automatically created in outputs/day/hour/)
    hydra_cfg = HydraConfig.get()
    output_fold = hydra_cfg.runtime.output_dir
    log.info(f"Output directory: {output_fold}")
    
    # Extract configuration values
    load_from = cfg.run.load_from
    load_model = cfg.run.load_model
    resume_training = cfg.run.resume_training
    
    # Training parameters
    val_interval = cfg.training.val_interval
    
    # Model parameters
    
    log.info(f"✓ Configuration loaded from Hydra (saved to {output_fold}/.hydra/)\n")
    
    # ============================================================
    # TEST CASE DEFINITION
    # ============================================================
    
    TRAINING_CASE = 'no_source'  # Options: 'no_source', 'with_source'
    
    # Test case parameters
    a_test = torch.tensor([[0.5, 0.3], [0.2, 0.1]]) if TRAINING_CASE == 'no_source' else torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    b_test = torch.tensor([[0.2, 0.1], [0.05, 0.025]]) if TRAINING_CASE == 'no_source' else torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    source_amplitude = 0.0 if TRAINING_CASE == 'no_source' else 15.0
    center_x_test = 0.35
    center_y_test = 0.65
    T_max = 1.0
    self_feeding = False
    
    # ============================================================
    # INITIALIZE MODEL
    # ============================================================
    
    model = PINNDeepONet_Wave2D(cfg)
    model = model.to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        log.info(f"GPU Memory: {torch.cuda.get_device_properties(DEVICE).total_memory / 1e9:.2f} GB")
        log.info(f"GPU: {torch.cuda.get_device_name(DEVICE)}")
    log.info(f"Model initialized with {total_params:,} parameters")
    log.info(f"\nModel architecture:\n{model}")
    
    # ============================================================
    # LOAD GROUND TRUTH (if available)
    # ============================================================
    gt_file = os.path.join(SCRIPT_DIR, f'data/gt_wave2D_{TRAINING_CASE}.csv')
    gt_rollout_file = os.path.join(SCRIPT_DIR, f'data/gt_wave2D_{TRAINING_CASE}_rollout.csv')
    
    gt_data = None
    gt_rollout_data = None
    
    if os.path.exists(gt_file):
        log.info(f"Loading ground truth from {gt_file}")
        gt_data = pd.read_csv(gt_file, header=None).values
        log.info(f"Ground truth shape: {gt_data.shape}")
        log.info(f"Columns: [x, y, t, f, u, v]")
    else:
        log.warning(f"Ground truth file {gt_file} not found")
        log.warning(f"Training will proceed without validation")
    
    if os.path.exists(gt_rollout_file):
        log.info(f"Loading rollout ground truth from {gt_rollout_file}")
        gt_rollout_data = pd.read_csv(gt_rollout_file, header=None).values
        log.info(f"Rollout ground truth shape: {gt_rollout_data.shape}")
    
    # ============================================================
    # TRAINING OR LOADING
    # ============================================================
    
    def _build_checkpoint_path(date, time, model_file='model.pth', load_fold_type='outputs'):
        """Helper function to construct checkpoint path"""
        if '/' in time:
            load_fold = 'multirun'
        else:
            load_fold = load_fold_type
        return f'{load_fold}/{date}/{time}/{model_file}'
    
    # Extract configuration values
    load_model = cfg.run.load_model
    load_pretrain = cfg.run.get('load_pretrain', False)
    resume_training = cfg.run.get('resume_training', False)
    continue_phase1_lbfgs = cfg.run.get('continue_phase1_lbfgs', False)
    load_from = cfg.run.get('load_from', None)
    load_pretrain_from = cfg.run.get('load_pretrain_from', None)
    resume_from = cfg.run.get('resume_from', None)
    
    # ============================================================
    # Case 1a: Load Pretrained Phase 1 and Continue to Phase 2 (if train_adam=True)
    # ============================================================
    if load_pretrain and not continue_phase1_lbfgs and not load_model and not resume_training:
        if load_pretrain_from is None:
            log.error("load_pretrain_from must be specified when load_pretrain=True")
            return
        
        checkpoint_path = _build_checkpoint_path(load_pretrain_from[0], load_pretrain_from[1], 
                                                  model_file='model_pretrain.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(DEVICE)  # Ensure model is on correct device after loading
            log.info(f"✓ Pretrained Phase 1 model loaded from {checkpoint_path}")
            log.info(f"  Test Metric: {checkpoint.get('test_metric', 'N/A')}")
            
            # Plot IC reconstruction for loaded pretrained model
            from plot_2d import plot_ic_reconstruction
            test_a = torch.tensor([[0.5, 0.3], [0.2, 0.1]]) if TRAINING_CASE == 'no_source' else torch.tensor([[0.0, 0.0], [0.0, 0.0]])
            test_b = torch.tensor([[0.2, 0.1], [0.05, 0.025]]) if TRAINING_CASE == 'no_source' else torch.tensor([[0.0, 0.0], [0.0, 0.0]])
            test_case = {
                'a_coeffs': test_a,
                'b_coeffs': test_b,
                'center_x': 0.35,
                'center_y': 0.65
            }
            log.info("Generating IC reconstruction diagnostic plot for loaded pretrained model...")
            plot_ic_reconstruction(model, test_case, save_path=os.path.join(output_fold, 'ic_pretrain_diagnostic.png'), gt_data=gt_data)
            log.info(f"IC reconstruction plot saved to {output_fold}/ic_pretrain_diagnostic.png")
            
            # Continue to Phase 2 training if enabled
            if not cfg.training.get('train_adam', False):
                log.info("Phase 2 (train_adam) is disabled. Skipping to validation only.")
                history = None
                training_time = None
            # else: Fall through to Phase 2 training logic below
        else:
            log.error(f"Pretrain checkpoint file {checkpoint_path} not found!")
            return
    
    # ============================================================
    # Case 1b: Load Pretrained Phase 1 and Continue with LBFGS Only
    # ============================================================
    elif load_pretrain and continue_phase1_lbfgs and not load_model and not resume_training:
        if load_pretrain_from is None:
            log.error("load_pretrain_from must be specified when load_pretrain=True and continue_phase1_lbfgs=True")
            return
        
        checkpoint_path = _build_checkpoint_path(load_pretrain_from[0], load_pretrain_from[1], 
                                                  model_file='model_pretrain.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(DEVICE)  # Ensure model is on correct device after loading
            log.info(f"✓ Pretrained Phase 1 model loaded from {checkpoint_path}")
            log.info(f"  Continuing with Phase 1 LBFGS fine-tuning...")
            
            # Configure for Phase 1 LBFGS continuation
            cfg.training.pretrain_ic = False  # Don't re-run Adam phase
            cfg.training.train_adam = False   # Skip Phase 2
            cfg.training.train_lbfgs = False  # Skip Phase 3
            cfg.training.use_lbfgs_finetune_ph1 = True  # Enable LBFGS in Phase 1
            
            # Run only LBFGS fine-tuning for Phase 1
            start_time = datetime.now()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Call a special method for Phase 1 LBFGS continuation
            history, pretrain_history, best_model_info = model.train_phase1_lbfgs_continuation(
                cfg,
                output_fold=output_fold,
                device=DEVICE,
                gt_data=gt_data
            )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = datetime.now()
            training_time = end_time - start_time
            
            # Save model
            model_filename = os.path.join(output_fold, 'model_pretrain_continued.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'history': history,
            }, model_filename)
            log.info(f"Continued Phase 1 model saved to {model_filename}")
        else:
            log.error(f"Pretrain checkpoint file {checkpoint_path} not found!")
            return
    
    # ============================================================
    # Case 2: Load Full Model (Phase 1 + Phase 2)
    # ============================================================
    elif load_model:
        if load_from is None:
            log.error("load_from must be specified when load_model=True")
            return
        
        checkpoint_path = _build_checkpoint_path(load_from[0], load_from[1], model_file='model.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(DEVICE)  # Ensure model is on correct device after loading
            log.info(f"✓ Full model loaded from {checkpoint_path}")
            history = None
            training_time = None
        else:
            log.error(f"Checkpoint file {checkpoint_path} not found!")
            log.error(f"Set load_model=False to train a new model.")
            return
    
    # ============================================================
    # Case 3: Resume Training from Checkpoint
    # ============================================================
    elif resume_training:
        if resume_from is None:
            log.error("resume_from must be specified when resume_training=True")
            return
        
        checkpoint_path = _build_checkpoint_path(resume_from[0], resume_from[1], model_file='model.pth')
        if not os.path.exists(checkpoint_path):
            log.error(f"Checkpoint not found: {checkpoint_path}")
            exit(1)
        
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(DEVICE)  # Ensure model is on correct device after loading
        log.info(f"✓ Model loaded for resume training: {checkpoint_path}")
        
        # Continue with training below
    
    # ============================================================
    # Phase 2: Train PDE Loss (Adam) - Triggered by train_adam flag
    # ============================================================
    if cfg.training.get('train_adam', False):
        if TRAINING_CASE == 'with_source':
            source_type = 'gaussian'
        else:
            source_type = 'zero'
        
        log.info("=" * 70)
        log.info("PHASE 2: Training PDE Loss with Adam Optimizer")
        log.info("=" * 70)
        
        start_time = datetime.now()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        history, pretrain_history, best_model_info = model.train_pinn(
            cfg,
            output_fold=output_fold,
            device=DEVICE,
            gt_data=gt_data
        )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = datetime.now()
        training_time = end_time - start_time
        
        model_filename = os.path.join(output_fold, 'model.pth')
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
            'pretrain_history': pretrain_history,
            'best_model_info': best_model_info
        }, model_filename)
        
        log.info(f"Model saved to {model_filename}")
        log.info(f"Best Model Info:")
        log.info(f"  Epoch: {best_model_info['best_epoch']}")
        log.info(f"  L2 Metric: {best_model_info['best_test_metric']:.6e}")
        log.info(f"  Loss Components:")
        log.info(f"    - PDE: {best_model_info['best_losses']['pde']:.6e}")
        log.info(f"    - IC_u: {best_model_info['best_losses']['ic_u']:.6e}")
        log.info(f"    - IC_v: {best_model_info['best_losses']['ic_v']:.6e}")
        
        # ============================================================
        # PLOT TRAINING HISTORY
        # ============================================================
        
        # Plot with pretraining history if available
        from plot_2d import plot_training_with_pretraining
        fig_history = plot_training_with_pretraining(history, pretrain_history, val_interval=val_interval)
        training_plot_filename = os.path.join(output_fold, f'pinn_wave_training_{TRAINING_CASE}.png')
        fig_history.savefig(training_plot_filename, dpi=150, bbox_inches='tight')
        log.info(f"Training history (with pretraining) saved to {training_plot_filename}")
    
    # ============================================================
    # Case 4: Train from Scratch (optionally with Phase 1 pretraining)
    # ============================================================
    elif not load_model and not load_pretrain:
        if TRAINING_CASE == 'with_source':
            source_type = 'gaussian'
        else:
            source_type = 'zero'
        
        log.info("=" * 70)
        log.info("Training from Scratch (Case 4)")
        log.info("=" * 70)
        
        start_time = datetime.now()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        history, pretrain_history, best_model_info = model.train_pinn(
            cfg,
            output_fold=output_fold,
            device=DEVICE,
            gt_data=gt_data
        )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = datetime.now()
        training_time = end_time - start_time
        
        model_filename = os.path.join(output_fold, 'model.pth')
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
            'pretrain_history': pretrain_history,
            'best_model_info': best_model_info
        }, model_filename)
        
        log.info(f"Model saved to {model_filename}")
        log.info(f"Best Model Info:")
        log.info(f"  Epoch: {best_model_info['best_epoch']}")
        log.info(f"  L2 Metric: {best_model_info['best_test_metric']:.6e}")
        log.info(f"  Loss Components:")
        log.info(f"    - PDE: {best_model_info['best_losses']['pde']:.6e}")
        log.info(f"    - IC_u: {best_model_info['best_losses']['ic_u']:.6e}")
        log.info(f"    - IC_v: {best_model_info['best_losses']['ic_v']:.6e}")
        
        # ============================================================
        # PLOT TRAINING HISTORY
        # ============================================================
        
        # Plot with pretraining history if available
        from plot_2d import plot_training_with_pretraining
        fig_history = plot_training_with_pretraining(history, pretrain_history, val_interval=val_interval)
        training_plot_filename = os.path.join(output_fold, f'pinn_wave_training_{TRAINING_CASE}.png')
        fig_history.savefig(training_plot_filename, dpi=150, bbox_inches='tight')
        log.info(f"Training history (with pretraining) saved to {training_plot_filename}")
    
    # ============================================================
    # PLOT SOLUTION SNAPSHOTS
    # ============================================================
    
    if TRAINING_CASE == 'with_source':
        source_type = 'gaussian'
    else:
        source_type = 'zero'
        
    fig_solution, metrics = plot_solution_2d_comparison(
        model, 
        a_test=a_test, 
        b_test=b_test, 
        source_type=source_type,
        source_amplitude=source_amplitude,
        center_x=center_x_test,
        center_y=center_y_test,
        T_max=T_max,
        gt_data=gt_data
    )
    metrics["training_time"] = str(training_time) if 'training_time' in locals() else None
    
    # Add best model info to metrics if available
    if 'best_model_info' in locals():
        metrics['best_epoch'] = best_model_info['best_epoch']
        metrics['best_test_metric'] = best_model_info['best_test_metric']
        metrics['best_pde_loss'] = best_model_info['best_losses']['pde']
        metrics['best_ic_u_loss'] = best_model_info['best_losses']['ic_u']
        metrics['best_ic_v_loss'] = best_model_info['best_losses']['ic_v']
        # Add phase timing information if available
        if 'phase_times' in best_model_info:
            for phase_name, phase_time in best_model_info['phase_times'].items():
                metrics[f'training_time_{phase_name}'] = phase_time
        # metrics['selection_method'] = best_model_info['selection_method']
        # metrics['test_n_cases'] = best_model_info['n_test_cases']
        # metrics['test_val_interval'] = best_model_info['val_interval']
        # metrics['test_early_stopping_patience'] = best_model_info['early_stopping_patience']
    
    # Save metrics to txt file in output folder
    metrics_txt_path = os.path.join(output_fold, "metrics.txt")
    with open(metrics_txt_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    log.info(f"Metrics saved to {metrics_txt_path}")
    solution_plot_filename = os.path.join(output_fold, f'pinn_wave_solution_{TRAINING_CASE}.png')
    fig_solution.savefig(solution_plot_filename, dpi=150, bbox_inches='tight')
    log.info(f"Solution comparison saved to {solution_plot_filename}")
    
    # ============================================================
    # COMPLETION
    # ============================================================
    log.info("=" * 70)
    log.info("EXPERIMENT COMPLETE")
    log.info("=" * 70)
    log.info(f"All outputs saved to: {output_fold}")
    
    # ============================================================
    # ROLLOUT TEST
    # ===========================================================
    
    # if gt_rollout_data is not None:
    #     print("\n" + "="*80)
    #     print("PERFORMING ROLLOUT TEST")
    #     print("="*80)
        
    #     results = rollout_test_2d(
    #         model,
    #         gt_rollout_data,
    #         n_intervals=10,
    #         dt_interval=1.0,
    #         self_feeding=self_feeding
    #     )
        
    #     # Plot rollout errors
    #     fig_errors = plot_rollout_errors_2d(results)
    #     fig_errors.savefig(f'logs_multibranch_wave2D/rollout_errors_wave2D_{TRAINING_CASE}.png', dpi=150, bbox_inches='tight')
    #     print(f"Rollout errors saved to logs_multibranch_wave2D/rollout_errors_wave2D_{TRAINING_CASE}.png")
        
    #     # Plot rollout snapshots
    #     fig_snapshots = plot_rollout_snapshots_2d(
    #         model, 
    #         results, 
    #         gt_rollout_data,
    #         snapshot_indices=[0, 4, 9]
    #     )
    #     fig_snapshots.savefig(f'logs_multibranch_wave2D/rollout_snapshots_wave2D_{TRAINING_CASE}.png', dpi=150, bbox_inches='tight')
    #     print(f"Rollout snapshots saved to logs_multibranch_wave2D/rollout_snapshots_wave2D_{TRAINING_CASE}.png")
    # else:
    #     print("\nSkipping rollout test (no rollout ground truth available)")
    
    # print("\n" + "="*80)
    # print("COMPLETE")
    # print("="*80)
    # print(f"\nOutputs saved in:")
    # print(f"  - checkpoints/")
    # print(f"  - outputs/")

if __name__ == "__main__":
    main()
