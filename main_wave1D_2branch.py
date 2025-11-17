import numpy as np
import torch
import matplotlib.pyplot as plt
import warnings
from model import PINNDeepONet_Wave
from plot import plot_solution, plot_training_history
from test import rollout_test
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # ==========================================================================
    # TRAINING CASE SELECTION
    # ==========================================================================
    # Choose training scenario:
    # 'no_source': Free wave with zero forcing
    # 'with_source': Wave with Gaussian source (varying amplitude)
    
    TRAINING_CASE = 'with_source'  # Change to 'no_source' or 'with_source'
    
    print("\n" + "="*70)
    print("PHYSICS-INFORMED DEEPONET FOR DAMPED WAVE EQUATION")
    print("="*70)
    print("PDE: u_tt + k*u_t = c² * u_xx + f(x)")
    print("IC:  u(x, 0) = a * sin(π*x)")
    print("     u_t(x, 0) = b * sin(π*x)")
    print("BC:  u(0, t) = u(1, t) = 0 (hard constraint)")
    print("="*70)
    
    # Create model
    model = PINNDeepONet_Wave(
        n_sensors_ic=20,
        n_sensors_src=20,
        branch_hidden=50,
        trunk_hidden=50,
        p=50,
        wave_speed=1.0,  # c=1, so c²=1
        damping_coeff=1.0
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training case: {TRAINING_CASE}")
    print()
    
    # ==========================================================================
    # TRAINING
    # ==========================================================================
    
    if TRAINING_CASE == 'no_source':
        # Case 1: Free wave (no forcing)
        print("Training: Free damped wave (no source)")
        history = model.train_pinn(
            n_epochs=5000,
            n_colloc=200,
            lr=1e-3,
            a_range=(-1.0, 1.0),      # Displacement IC amplitude range
            b_range=(-4.0, 4.0),      # Initial velocity range
            source_type='zero',      # No forcing
            source_amplitude=7.5,    # Default amplitude (not used for zero source)
            T_max=2.0                # Time domain
        )
        
        # Test case
        a_test = 0.5
        b_test = 2.0
        source_test_type = 'zero'
        source_test_amp = 7.5  # Default (not used for zero source)
        source_test_center = 0.5  # Not used for zero source
        T_test = 2.0
        model_filename = 'pinn_deeponet_wave_nosource.pth'
        gt_filename = 'data/gt_wave1D_nosource.csv'
        
    elif TRAINING_CASE == 'with_source':
        # Case 2: Wave with Gaussian source (varying center location during training)
        print("Training: Damped wave with Gaussian source (varying center)")
        print("  Source amplitude: 7.5 (fixed)")
        print("  Source width: 0.3")
        print("  Training center range: [0.1, 0.9]")
        
        history = model.train_pinn(
            n_epochs=5000,
            n_colloc=200,
            lr=1e-3,
            a_range=(-1.0, 1.0),        # Displacement IC amplitude range
            b_range=(-4.0, 4.0),        # Initial velocity range
            source_type='gaussian',    # Gaussian forcing
            source_amplitude=7.5,      # Fixed amplitude
            source_center=0.5,         # Default center (not used when center_range is provided)
            center_range=(0.1, 0.9),   # Vary center location during training
            T_max=2.0                  # Time domain
        )
        
        # Test case: test at center x=0.5 (middle of training range)
        a_test = 0.5
        b_test = 2.0
        source_test_type = 'gaussian'
        source_test_amp = 7.5
        source_test_center = 0.17  # Test at x=0.17
        T_test = 2.0
        model_filename = 'pinn_deeponet_wave_withsource.pth'
        gt_filename = 'data/gt_wave1D_withsource.csv'
        
    else:
        raise ValueError(f"Unknown training case: {TRAINING_CASE}")
    
    # ==========================================================================
    # SAVE MODEL
    # ==========================================================================
    torch.save(model.state_dict(), f'checkpoints/{model_filename}')
    print(f"\n✓ Model saved: checkpoints/{model_filename}\n")
    
    # ==========================================================================
    # PLOT RESULTS
    # ==========================================================================
    print("Generating plots...")
    
    # Try to load ground truth if available
    try:
        gt_data = np.loadtxt(gt_filename, delimiter=',')
        print(f"✓ Loaded ground truth: {gt_filename}")
    except FileNotFoundError:
        print(f"⚠ Ground truth file not found: {gt_filename}")
        print("  Run MATLAB script first to generate ground truth.")
        gt_data = None
    
    # Plot solution
    fig1 = plot_solution(model, 
                         a_test=a_test, 
                         b_test=b_test, 
                         source_type=source_test_type, 
                         source_amplitude=source_test_amp,
                         source_center=source_test_center,
                         T_max=T_test, 
                         gt_data=gt_data)
    
    solution_plot_filename = f'outputs/pinn_wave_solution_{TRAINING_CASE}.png'
    fig1.savefig(solution_plot_filename, dpi=150, bbox_inches='tight')
    print(f"✓ Solution plot saved: {solution_plot_filename}")
    
    # Plot training history
    fig2 = plot_training_history(history)
    training_plot_filename = f'outputs/pinn_wave_training_{TRAINING_CASE}.png'
    fig2.savefig(training_plot_filename, dpi=150, bbox_inches='tight')
    print(f"✓ Training history saved: {training_plot_filename}")
    
    # ==========================================================================
    # ROLLOUT TEST (only for with_source case)
    # ==========================================================================
    if TRAINING_CASE == 'with_source':
        print("\n" + "="*70)
        print("ROLLOUT TEST: 10-second simulation with changing source")
        print("="*70)
        
        # Load rollout ground truth (REQUIRED)
        try:
            gt_rollout = np.loadtxt('data/gt_wave1D_withsource_rollout.csv', delimiter=',')
            print(f"✓ Loaded rollout ground truth: data/gt_wave1D_withsource_rollout.csv")
        except FileNotFoundError:
            print(f"✗ Rollout ground truth not found: data/gt_wave1D_withsource_rollout.csv")
            print("  Rollout test requires ground truth data. Skipping...")
            gt_rollout = None
        
        # Run rollout test only if ground truth is available
        if gt_rollout is not None:
            u_roll, t_roll, x_roll, fig3 = rollout_test(
                model, 
                gt_data=gt_rollout,
                T_total=10.0,
                dt_interval=1.0
            )
            
            rollout_plot_filename = 'outputs/pinn_wave_rollout_withsource.png'
            fig3.savefig(rollout_plot_filename, dpi=150, bbox_inches='tight')
            print(f"✓ Rollout plot saved: {rollout_plot_filename}")
            print(f"  Rollout shape: {u_roll.shape}")
    
    plt.show()
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
