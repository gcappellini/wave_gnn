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
    
    TRAINING_CASE = 'no_source'  # Change to 'no_source' or 'with_source'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_fold = os.path.join(SCRIPT_DIR, f'logs_multibranch_wave/{TRAINING_CASE}_{timestamp}')
    os.makedirs(output_fold, exist_ok=True)

    # Load or train model
    load_model = False  

    n_sensors_ic = 20
    n_sensors_src = 20  # was 20
    branch_hidden = 300
    trunk_hidden = 300
    p = 300
    wave_speed = 1.0
    damping_coeff = 1.0

    w_pde, w_ic_u, w_ic_v = 1.0, 10.0, 10.0
    strategy = 'fixed'  # or 'equal_init', 'ema', 'fixed', 'ntk'

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

    n_epochs = 10000
    n_colloc = 800
    n_ic = 60
    lr = 1e-3
    a_range = (-0.1, 0.6)
    b_range = (-1.2, 2.0)
    n_ic_u, n_ic_v = 3, 5 # was 2, 2
    center_range = None if TRAINING_CASE == 'no_source' else (0.1, 0.9)
    T_max = 1.0
    source_type = 'gaussian' if TRAINING_CASE == 'with_source' else 'zero'

    a_test = 0.5
    b_test = 2.0

    source_test_center = 0.17  

    gt_filename = os.path.join(SCRIPT_DIR, f'data/gt_wave1D_{TRAINING_CASE}.csv')
    T_rollout = 10.0 
    dt_rollout = 0.5  
    self_feeding = False

    # Save all parameters to JSON for reproducibility
    import json
    params = {
        "TRAINING_CASE": TRAINING_CASE,
        "n_sensors_ic": n_sensors_ic,
        "n_sensors_src": n_sensors_src,
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
        "a_test": a_test,
        "b_test": b_test,
        "source_test_center": source_test_center,
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
    
    # ==========================================================================
    # TRAINING
    # ==========================================================================
    
    if load_model:
        # Load existing model
        model_filename = os.path.join(SCRIPT_DIR, f'checkpoints/pinn_deeponet_wave_{TRAINING_CASE}_{timestamp}.pth')
        model.load_state_dict(torch.load(model_filename))
        print(f"✓ Model loaded: {model_filename}\n")
        history = None  # No training history when loading model
    else:
        history = model.train_pinn(
            n_epochs=n_epochs,
            n_colloc=n_colloc,
            n_ic=n_ic,
            lr=lr,
            a_range=a_range,      # Displacement IC amplitude range
            b_range=b_range,      # Initial velocity range
            source_type=source_type,      # No forcing  
            T_max=T_max,                # Time domain
            w_pde=w_pde,
            w_ic_u=w_ic_u,
            w_ic_v=w_ic_v,
            strategy=strategy,
            center_range=center_range,  # Source center range
            n_ic_u=n_ic_u,
            n_ic_v=n_ic_v,
            output_fold=output_fold
        )
        
        model_filename = os.path.join(SCRIPT_DIR, f'checkpoints/pinn_deeponet_wave_{TRAINING_CASE}_{timestamp}.pth')
        torch.save(model.state_dict(), model_filename)
        print(f"\n✓ Model saved: {model_filename}\n")

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
    fig2 = plot_solution(model, 
                         a_test=a_test, 
                         b_test=b_test, 
                         source_type=source_type, 
                         source_center=source_test_center,
                         T_max=T_max, 
                         gt_data=gt_data)
    
    solution_plot_filename = os.path.join(output_fold, f'pinn_wave_solution_{TRAINING_CASE}.png')
    fig2.savefig(solution_plot_filename, dpi=150, bbox_inches='tight')
    print(f"✓ Solution plot saved: {solution_plot_filename}")
    # ==========================================================================
    # ROLLOUT TEST (only for with_source case)
    # ==========================================================================
    if TRAINING_CASE == 'with_source':
        
        # Load rollout ground truth (REQUIRED)
        try:
            gt_rollout = np.loadtxt(os.path.join(SCRIPT_DIR, f'data/gt_wave1D_{TRAINING_CASE}_rollout.csv'), delimiter=',')
            print(f"✓ Loaded rollout ground truth: data/gt_wave1D_{TRAINING_CASE}_rollout.csv")
        except FileNotFoundError:
            print(f"✗ Rollout ground truth not found: data/gt_wave1D_{TRAINING_CASE}_rollout.csv")
            print("  Rollout test requires ground truth data. Skipping...")
            gt_rollout = None
        
        # Run rollout test only if ground truth is available
        if gt_rollout is not None:
            u_roll, t_roll, x_roll, fig3 = rollout_test(
                model, 
                gt_data=gt_rollout,
                T_total=T_rollout,
                dt_interval=dt_rollout,
                self_feeding=self_feeding,
            )
            
            rollout_plot_filename = os.path.join(output_fold, f'pinn_wave_rollout_{TRAINING_CASE}.png')
            fig3.savefig(rollout_plot_filename, dpi=150, bbox_inches='tight')
            print(f"✓ Rollout plot saved: {rollout_plot_filename}")
            print(f"  Rollout shape: {u_roll.shape}")
    
    plt.show()

