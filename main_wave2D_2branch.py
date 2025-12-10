import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import os
import logging
from datetime import datetime
from hydra import main as hydra_main
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from model_2d import PINNDeepONet_Wave2D
from plot_2d import plot_solution_2d, plot_solution_2d_comparison, plot_training_history

# Setup logging
log = logging.getLogger(__name__)

# Get script directory for absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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
    n_epochs = cfg.training.n_epochs
    es_patience = cfg.training.es_patience
    val_interval = cfg.training.val_interval
    lr = cfg.training.lr
    lr_scheduler_step = cfg.training.lr_scheduler_step
    lr_scheduler_gamma = cfg.training.lr_scheduler_gamma
    log_interval = cfg.training.log_interval
    max_grad_norm = cfg.training.max_grad_norm
    n_batches = cfg.training.n_batches
    
    # Model parameters
    n_sensors_ic = cfg.model.n_sensors_ic
    n_sensors_src = cfg.model.n_sensors_src
    branch_width = cfg.model.branch_width
    trunk_width = cfg.model.trunk_width
    branch_act = eval(cfg.model.branch_act)  # Convert string to nn module
    trunk_act = eval(cfg.model.trunk_act)
    branch_depth = cfg.model.branch_depth
    trunk_depth = cfg.model.trunk_depth
    p = cfg.model.p
    wave_speed = cfg.model.wave_speed
    damping_coeff = cfg.model.damping_coeff
    w_pde = cfg.model.w_pde
    w_ic_u = cfg.model.w_ic_u
    w_ic_v = cfg.model.w_ic_v
    strategy = cfg.model.strategy
    use_fft_trunk = cfg.model.use_fft_trunk
    fft_trunk_args = OmegaConf.to_container(cfg.model.fft_trunk_args) if cfg.model.use_fft_trunk else None
    
    # Dataset parameters
    n_colloc = cfg.data.n_colloc
    n_ic = cfg.data.n_ic
    a_range = tuple(cfg.data.a_range)
    b_range = tuple(cfg.data.b_range)
    center_x_range = tuple(cfg.data.center_x_range)
    center_y_range = tuple(cfg.data.center_y_range)
    n_ic_u = cfg.data.n_ic_u
    n_ic_v = cfg.data.n_ic_v
    
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
    
    model = PINNDeepONet_Wave2D(
        n_sensors_ic=n_sensors_ic,
        n_sensors_src=n_sensors_src,
        branch_depth=branch_depth,
        trunk_depth=trunk_depth,
        branch_width=branch_width,
        trunk_width=trunk_width,
        branch_activation=branch_act,
        trunk_activation=trunk_act,
        p=p,
        use_fft_trunk=use_fft_trunk,
        fft_trunk_args=fft_trunk_args,
        wave_speed=wave_speed,
        damping_coeff=damping_coeff
    )
    
    log.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
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
    
    checkpoint_path = f'checkpoints/pinn_wave2D_{load_from}.pth'
    
    if load_model:
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            log.info(f"Model loaded from {checkpoint_path}")
            history = None
            training_time = None
        else:
            log.error(f"Checkpoint file {checkpoint_path} not found!")
            log.error(f"Set load_model=False to train a new model.")
            return
    else:
        if resume_training:
            if not os.path.exists(checkpoint_path):
                log.error(f"Pretrained model not found: {checkpoint_path}")
                exit(1)
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            log.info(f"✓ Pretrained model loaded for curriculum learning: {checkpoint_path}")
        
        if TRAINING_CASE == 'with_source':
            source_type = 'gaussian'
        else:
            source_type = 'zero'
        
        start_time = datetime.now()
        history, best_model_info = model.train_pinn(
            n_epochs=n_epochs,
            n_colloc=n_colloc,
            n_ic=n_ic,
            a_range = a_range,
            b_range=b_range,
            center_x_range=center_x_range,
            center_y_range=center_y_range,
            T_max=T_max,
            source_type=source_type,
            lr=lr,
            strategy=strategy,
            output_fold=output_fold,
            n_ic_u=n_ic_u,
            n_ic_v=n_ic_v,
            w_pde=w_pde,
            w_ic_u=w_ic_u,
            w_ic_v=w_ic_v,
            early_stopping_patience=es_patience,
            val_interval=val_interval,
            lr_scheduler_gamma=lr_scheduler_gamma,
            lr_scheduler_step=lr_scheduler_step,
            log_interval=log_interval,
            max_grad_norm=max_grad_norm,
            n_batches=n_batches,
        )
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        model_filename = os.path.join(output_fold, 'model.pth')
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
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
        
        fig_history = plot_training_history(history, val_interval=val_interval)
        training_plot_filename = os.path.join(output_fold, f'pinn_wave_training_{TRAINING_CASE}.png')
        fig_history.savefig(training_plot_filename, dpi=150, bbox_inches='tight')
        log.info(f"Training history saved to {training_plot_filename}")
    
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
