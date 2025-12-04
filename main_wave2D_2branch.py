import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import os
from datetime import datetime
from model_2d import PINNDeepONet_Wave2D
from plot_2d import plot_solution_2d, plot_solution_2d_comparison, plot_training_history

# Create output directories
os.makedirs('data', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Get script directory for absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    # ============================================================
    # USER CONFIGURATION
    # ============================================================
    
    TRAINING_CASE = 'no_source'  # Options: 'no_source', 'with_source'
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_fold = os.path.join(SCRIPT_DIR, f'logs_multibranch_wave2D/{TRAINING_CASE}_{timestamp}')
    os.makedirs(output_fold, exist_ok=True)
    
    # Load or train model
    load_from = None #'20251202_113255' 
    load_model = False  # If True, only test a pretrained model (no training)
    resume_training = False  # If True, load a pretrained model and continue training (curriculum learning)
                             # NOTE: If you change fft_trunk_args, set resume_training=False to use new FFT parameters!
                             # The pretrained model's FFT weights are fixed and won't change.
    
    # Training parameters
    n_epochs = 2000
    n_sensors_ic = 40      # Creates 20x20 grid (400 sensors)
    n_sensors_src = 20     # Creates 20x20 grid (400 sensors)
    branch_width = 300
    trunk_width = 512
    branch_act=nn.LeakyReLU()
    trunk_act=nn.Tanh()
    branch_depth = 4
    trunk_depth = 6
    p = 300 
    n_colloc = 800
    n_ic = 50

    lr = 1e-3

    self_feeding = False    # If True, use self-feeding during rollout test

    a_range = (-0.1, 0.6)   # IC displacement amplitude range
    b_range = (-0.1, 0.1)    # IC velocity amplitude
    center_x_range = (0.3, 0.7)  # Source center x-coordinate range
    center_y_range = (0.3, 0.7)  # Source center y-coordinate range
    n_ic_u, n_ic_v = 2, 2

    use_fft_trunk=True
    fft_trunk_args = {
        "input_dim": 3,
        "m_spatial_x": 64,
        "m_spatial_y": 64,
        "m_temporal": 64,
        "sigma_spatial_x": 1.0,
        "sigma_spatial_y": 1.0,
        "sigma_temporal_list": [1.0],
        "seed": 52
    }   
    w_pde, w_ic_u, w_ic_v = 1.0, 50.0, 50.0
    strategy = 'ntk'
    
    # Test parameters - 2D coefficient arrays (matching MATLAB)
    # u(x,y) = 0.5*sin(π*x)*sin(π*y) + 0.3*sin(π*x)*sin(2π*y) + 0.2*sin(2π*x)*sin(π*y) + 0.1*sin(2π*x)*sin(2π*y)
    a_test = torch.tensor([[0.5, 0.3], [0.2, 0.1]]) if TRAINING_CASE == 'no_source' else torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    
    # v(x,y) similar structure with smaller coefficients
    b_test = torch.tensor([[0.2, 0.1], [0.05, 0.025]]) if TRAINING_CASE == 'no_source' else torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    source_amplitude = 0.0 if TRAINING_CASE == 'no_source' else 15.0
    center_x_test = 0.35   # Source center x-coordinate
    center_y_test = 0.65   # Source center y-coordinate
    T_max = 1.0            # Training time horizon
    
    # Save all parameters to JSON for reproducibility
    import json
    params = {
        "TRAINING_CASE": TRAINING_CASE,
        "load_from": load_from,
        "resume_training": resume_training,
        "n_sensors_ic": n_sensors_ic,
        "n_sensors_src": n_sensors_src,
        # "n_sensors_src_t": n_sensors_src_t,
        "branch_width": branch_width,
        "trunk_width": trunk_width,
        "p": p,
        # "wave_speed": wave_speed,
        # "damping_coeff": damping_coeff,
        "w_pde": w_pde,
        "w_ic_u": w_ic_u,
        "w_ic_v": w_ic_v,
        "strategy": strategy,
        "branch_depth": branch_depth,
        "trunk_depth": trunk_depth,
        "trunk_activation": str(trunk_act),
        # "use_fft_branch": use_fft_branch,
        "use_fft_trunk": use_fft_trunk,
        # "fft_branch_params": fft_branch_params,
        "fft_trunk_params": fft_trunk_args,
        "n_epochs": n_epochs,
        "n_colloc": n_colloc,
        "n_ic": n_ic,
        "lr": lr,
        "a_range": a_range,
        "b_range": b_range,
        "n_ic_u": n_ic_u,
        "n_ic_v": n_ic_v,
        "center_range_x": center_x_range,
        "center_range_y": center_y_range,
        "T_max": T_max,
        # "source_type": source_type,
        # "center_t_range": center_t_range,
        "a_test": a_test,
        "b_test": b_test,
        "source_test_center_x": center_x_test,
        "source_test_center_y": center_y_test,
        # "source_test_t": source_test_t,
        # "gt_filename": gt_filename,
        # "T_rollout": T_rollout,
        # "dt_rollout": dt_rollout,
        "self_feeding": self_feeding
    }
    params_path = os.path.join(output_fold, "params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2, default=str)
    print(f"✓ Parameters saved to {params_path}\n")
    
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
        fft_trunk_args=fft_trunk_args
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    print(model)
    
    # ============================================================
    # LOAD GROUND TRUTH (if available)
    # ============================================================
    gt_file = os.path.join(SCRIPT_DIR, f'data/gt_wave2D_{TRAINING_CASE}.csv')
    gt_rollout_file = os.path.join(SCRIPT_DIR, f'data/gt_wave2D_{TRAINING_CASE}_rollout.csv')
    
    gt_data = None
    gt_rollout_data = None
    
    if os.path.exists(gt_file):
        print(f"\nLoading ground truth from {gt_file}")
        gt_data = pd.read_csv(gt_file, header=None).values
        print(f"Ground truth shape: {gt_data.shape}")
        print(f"Columns: [x, y, t, f, u, v]")
    else:
        print(f"\nWarning: Ground truth file {gt_file} not found")
        print("Training will proceed without validation")
    
    if os.path.exists(gt_rollout_file):
        print(f"Loading rollout ground truth from {gt_rollout_file}")
        gt_rollout_data = pd.read_csv(gt_rollout_file, header=None).values
        print(f"Rollout ground truth shape: {gt_rollout_data.shape}")
    
    # ============================================================
    # TRAINING OR LOADING
    # ============================================================
    
    checkpoint_path = f'checkpoints/pinn_wave2D_{load_from}.pth'
    
    if load_model:
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {checkpoint_path}")
            history = None
            training_time = None
        else:
            print(f"Error: Checkpoint file {checkpoint_path} not found!")
            print("Set load_model=False to train a new model.")
            return
    else:
        if resume_training:
            if not os.path.exists(checkpoint_path):
                print(f"✗ Pretrained model not found: {checkpoint_path}")
                exit(1)
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Pretrained model loaded for curriculum learning: {checkpoint_path}\n")
        
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
            w_ic_v=w_ic_v
        )
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        model_filename = os.path.join(SCRIPT_DIR, f'checkpoints/pinn_wave2D_{timestamp}.pth')
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
            'best_model_info': best_model_info,
            'config': {
                'n_sensors_ic': n_sensors_ic,
                'n_sensors_src': n_sensors_src,
                'branch_width': branch_width,
                'trunk_width': trunk_width,
                'branch_depth': branch_depth,
                'trunk_depth': trunk_depth,
                'p': p,
                'TRAINING_CASE': TRAINING_CASE
            }
        }, model_filename)
        
        print(f"\nModel saved to {model_filename}")
        print(f"\nBest Model Info:")
        print(f"  Epoch: {best_model_info['best_epoch']}")
        print(f"  L2 Metric: {best_model_info['best_test_metric']:.6e}")
        print(f"  Loss Components:")
        print(f"    - PDE: {best_model_info['best_losses']['pde']:.6e}")
        print(f"    - IC_u: {best_model_info['best_losses']['ic_u']:.6e}")
        print(f"    - IC_v: {best_model_info['best_losses']['ic_v']:.6e}")
        
        # ============================================================
        # PLOT TRAINING HISTORY
        # ============================================================
        
        fig_history = plot_training_history(history)
        training_plot_filename = os.path.join(SCRIPT_DIR, f'{output_fold}/pinn_wave_training_{TRAINING_CASE}.png')
        fig_history.savefig(training_plot_filename, dpi=150, bbox_inches='tight')
        print(f"Training history saved to {training_plot_filename}")
    
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
        # metrics['test_eval_freq'] = best_model_info['eval_freq']
        # metrics['test_early_stopping_patience'] = best_model_info['early_stopping_patience']
    
    # Save metrics to txt file in output folder
    metrics_txt_path = os.path.join(output_fold, "metrics.txt")
    with open(metrics_txt_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"Metrics saved to {metrics_txt_path}")
    solution_plot_filename = os.path.join(output_fold, f'pinn_wave_solution_{TRAINING_CASE}.png')
    fig_solution.savefig(solution_plot_filename, dpi=150, bbox_inches='tight')
    print(f"Solution comparison saved to {solution_plot_filename}")
    
    # ===========================================================
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
