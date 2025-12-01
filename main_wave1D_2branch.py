import numpy as np
import torch
import matplotlib.pyplot as plt
import warnings
from model import PINNDeepONet_Wave
import torch.nn as nn
from plot import plot_solution, plot_training_history
from test import rollout_test
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Get script directory for absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(52)
    np.random.seed(52)
    
    TRAINING_CASE = 'with_source'  # Options: 'no_source', 'with_source', 'time_source'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_fold = os.path.join(SCRIPT_DIR, f'logs_multibranch_wave/{TRAINING_CASE}_{timestamp}')
    os.makedirs(output_fold, exist_ok=True)

    # Load or train model
    load_from = '20251201_125517'
    load_model = False  # If True, only test a pretrained model (no training)
    resume_training = True  # If True, load a pretrained model and continue training (curriculum learning)

    n_sensors_ic = 20
    n_sensors_src = 20
    n_sensors_src_t = 10  # Always spatiotemporal for curriculum learning
    branch_hidden = 300
    trunk_hidden = 300
    p = 300
    wave_speed = 1.0
    damping_coeff = 1.0

    w_pde, w_ic_u, w_ic_v = 1.0, 10.0, 10.0
    strategy = 'ntk'  # or 'equal_init', 'ema', 'fixed', 'ntk'

    branch_n_hidden=2
    trunk_n_hidden=2
    branch_activation=nn.LeakyReLU()
    trunk_activation=nn.Tanh()
    use_fft_branch=False
    use_fft_trunk=True
    fft_branch_params={}
    fft_trunk_params={
        "input_dim": 2,          # for (x, t)
        "m_spatial": 64,
        "m_temporal": 64,
        "sigma_spatial": 0.10,
        "sigma_temporal_list": [0.1],
        "seed": 52}

    n_epochs = 20000
    n_colloc = 800
    n_ic = 60
    lr = 1e-3
    a_range = (-0.1, 0.6)
    b_range = (-1.2, 2.0)
    n_ic_u, n_ic_v = 3, 5
    # Curriculum learning: control complexity via amplitude and range parameters
    center_range = (0.1, 0.9) if TRAINING_CASE in ['with_source', 'time_source'] else (0.5, 0.5)
    center_t_range = (0.1, 0.9) if TRAINING_CASE == 'time_source' else (0.5, 0.5)
    T_max = 1.0
    source_type = 'gaussian' if TRAINING_CASE in ['with_source', 'time_source'] else 'zero'
    # center_velocity_range = (-0.3, 0.3) if TRAINING_CASE == 'time_source' else None

    a_test = 0.5
    b_test = 2.0

    source_test_center = 0.2  
    source_test_t = 0.5  # Always set for consistent architecture
    # temporal_freq_test = 1.0
    # center_velocity_test = 0.0

    gt_filename = os.path.join(SCRIPT_DIR, f'data/gt_wave1D_{TRAINING_CASE}.csv')
    T_rollout = 10.0 
    dt_rollout = 1.0 
    self_feeding = False

    # Save all parameters to JSON for reproducibility
    import json
    params = {
        "TRAINING_CASE": TRAINING_CASE,
        "n_sensors_ic": n_sensors_ic,
        "n_sensors_src": n_sensors_src,
        "n_sensors_src_t": n_sensors_src_t,
        "branch_hidden": branch_hidden,
        "trunk_hidden": trunk_hidden,
        "p": p,
        "wave_speed": wave_speed,
        "damping_coeff": damping_coeff,
        "w_pde": w_pde,
        "w_ic_u": w_ic_u,
        "w_ic_v": w_ic_v,
        "strategy": strategy,
        "branch_n_hidden": branch_n_hidden,
        "trunk_n_hidden": trunk_n_hidden,
        "branch_activation": str(branch_activation),
        "trunk_activation": str(trunk_activation),
        "use_fft_branch": use_fft_branch,
        "use_fft_trunk": use_fft_trunk,
        "fft_branch_params": fft_branch_params,
        "fft_trunk_params": fft_trunk_params,
        "n_epochs": n_epochs,
        "n_colloc": n_colloc,
        "n_ic": n_ic,
        "lr": lr,
        "a_range": a_range,
        "b_range": b_range,
        "n_ic_u": n_ic_u,
        "n_ic_v": n_ic_v,
        "center_range": center_range,
        "T_max": T_max,
        "source_type": source_type,
        "center_t_range": center_t_range,
        "a_test": a_test,
        "b_test": b_test,
        "source_test_center": source_test_center,
        "source_test_t": source_test_t,
        "gt_filename": gt_filename,
        "T_rollout": T_rollout,
        "dt_rollout": dt_rollout,
        "self_feeding": self_feeding
    }
    params_path = os.path.join(output_fold, "params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2, default=str)
    print(f"✓ Parameters saved to {params_path}\n")

    # Create model
    model = PINNDeepONet_Wave(
        n_sensors_ic=n_sensors_ic,
        n_sensors_src=n_sensors_src,
        n_sensors_src_t=n_sensors_src_t,
        branch_hidden=branch_hidden,
        trunk_hidden=trunk_hidden,
        p=p,
        wave_speed=wave_speed,  
        damping_coeff=damping_coeff,
        branch_n_hidden=branch_n_hidden,
        trunk_n_hidden=trunk_n_hidden,
        branch_activation=branch_activation,
        trunk_activation=trunk_activation,
        use_fft_branch=use_fft_branch,
        use_fft_trunk=use_fft_trunk,
        fft_branch_args=fft_branch_params,
        fft_trunk_args=fft_trunk_params
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training case: {TRAINING_CASE}")
    print()

    # Path to pretrained model (for curriculum learning or testing)
    pretrained_model_path = os.path.join(SCRIPT_DIR, f'checkpoints/pinn_deeponet_wave_{load_from}.pth')
    
    # ==========================================================================
    # TRAINING
    # ==========================================================================

    if load_model:
        # Load existing model for inference/testing only
        if not os.path.exists(pretrained_model_path):
            print(f"✗ Model file not found: {pretrained_model_path}")
            exit(1)
        model.load_state_dict(torch.load(pretrained_model_path))
        print(f"✓ Model loaded: {pretrained_model_path}\n")
        history = None  # No training history when loading model
        training_time=0
    else:
        # Curriculum learning: resume training from a pretrained model if requested
        if resume_training:
            if not os.path.exists(pretrained_model_path):
                print(f"✗ Pretrained model not found: {pretrained_model_path}")
                exit(1)
            model.load_state_dict(torch.load(pretrained_model_path))
            print(f"✓ Pretrained model loaded for curriculum learning: {pretrained_model_path}\n")
        # Train (from scratch or from pretrained)
        start_time = datetime.now()
        history = model.train_pinn(
            n_epochs=n_epochs,
            n_colloc=n_colloc,
            n_ic=n_ic,
            lr=lr,
            a_range=a_range,
            b_range=b_range,
            source_type=source_type,
            T_max=T_max,
            w_pde=w_pde,
            w_ic_u=w_ic_u,
            w_ic_v=w_ic_v,
            strategy=strategy,
            center_range=center_range,
            center_t_range=center_t_range,
            n_ic_u=n_ic_u,
            n_ic_v=n_ic_v,
            output_fold=output_fold
        )
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds() / 60  # in minutes

        model_filename = os.path.join(SCRIPT_DIR, f'checkpoints/pinn_deeponet_wave_{timestamp}.pth')
        torch.save(model.state_dict(), model_filename)
        print(f"\n✓ Model saved: {model_filename}\n")

        # Optionally, save the model as a new pretrained checkpoint for future curriculum learning
        # torch.save(model.state_dict(), pretrained_model_path)

        # Plot training history
        fig1 = plot_training_history(history)
        training_plot_filename = os.path.join(SCRIPT_DIR, f'{output_fold}/pinn_wave_training_{TRAINING_CASE}.png')
        fig1.savefig(training_plot_filename, dpi=150, bbox_inches='tight')
        print(f"✓ Training history saved: {training_plot_filename}")
    # ==========================================================================
    # PLOT RESULTS
    # ==========================================================================

    try:
        gt_data = np.loadtxt(gt_filename, delimiter=',')
        print(f"✓ Loaded ground truth: {gt_filename}")
    except FileNotFoundError:
        print(f"⚠ Ground truth file not found: {gt_filename}")
        print("  Run MATLAB script first to generate ground truth.")
        gt_data = None
    
    # Plot solution
    fig2, metrics = plot_solution(model, 
                         a_test=a_test, 
                         b_test=b_test, 
                         source_type=source_type, 
                         source_center=source_test_center,
                         source_t=source_test_t,
                         T_max=T_max, 
                         gt_data=gt_data)
    
    solution_plot_filename = os.path.join(output_fold, f'pinn_wave_solution_{TRAINING_CASE}.png')
    fig2.savefig(solution_plot_filename, dpi=150, bbox_inches='tight')
    # Save metrics and training time to a text file
    metrics_path = os.path.join(output_fold, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("Metrics:\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nTraining time (minutes): {training_time}\n")
    print(f"✓ Metrics and training time saved: {metrics_path}")
    print(f"✓ Solution plot saved: {solution_plot_filename}")
    # ==========================================================================
    # ROLLOUT TEST (only for with_source case)
    # ==========================================================================
    # if TRAINING_CASE in ['with_source', 'time_source']:
        
    #     # Load rollout ground truth (REQUIRED)
    #     try:
    #         gt_rollout = np.loadtxt(os.path.join(SCRIPT_DIR, f'data/gt_wave1D_{TRAINING_CASE}_rollout.csv'), delimiter=',')
    #         print(f"✓ Loaded rollout ground truth: data/gt_wave1D_{TRAINING_CASE}_rollout.csv")
    #     except FileNotFoundError:
    #         print(f"✗ Rollout ground truth not found: data/gt_wave1D_{TRAINING_CASE}_rollout.csv")
    #         print("  Rollout test requires ground truth data. Skipping...")
    #         gt_rollout = None
        
    #     # Run rollout test only if ground truth is available
    #     if gt_rollout is not None:
    #         u_roll, t_roll, x_roll, fig3 = rollout_test(
    #             model, 
    #             gt_data=gt_rollout,
    #             T_total=T_rollout,
    #             dt_interval=dt_rollout,
    #             self_feeding=self_feeding,
    #         )
            
    #         rollout_plot_filename = os.path.join(output_fold, f'pinn_wave_rollout_{TRAINING_CASE}.png')
    #         fig3.savefig(rollout_plot_filename, dpi=150, bbox_inches='tight')
    #         print(f"✓ Rollout plot saved: {rollout_plot_filename}")
    #         print(f"  Rollout shape: {u_roll.shape}")
    
    # plt.show()

