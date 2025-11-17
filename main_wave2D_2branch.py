import torch
import numpy as np
import pandas as pd
import os
from model_2d import PINNDeepONet_Wave2D
from plot_2d import plot_solution_2d, plot_solution_2d_comparison, plot_training_history
from test_2d import rollout_test_2d, plot_rollout_errors_2d, plot_rollout_snapshots_2d

# Create output directories
os.makedirs('data', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

def main():
    # ============================================================
    # USER CONFIGURATION
    # ============================================================
    
    # Select test case: 'no_source' or 'with_source'
    test_case = 'with_source'
    
    # Load or train model
    load_model = False  # Set to True to load existing model instead of training
    
    # Training parameters
    n_epochs = 5000
    n_sensors_ic = 20      # Creates 20x20 grid (400 sensors)
    n_sensors_src = 20     # Creates 20x20 grid (400 sensors)
    branch_hidden = 200
    trunk_hidden = 200
    p = 200 
    n_colloc = 2000
    
    # Test parameters
    a_test = 0.0           # IC displacement amplitude
    b_test = 0.0           # IC velocity amplitude
    source_amplitude = 15.0
    center_x_test = 0.35   # Source center x-coordinate
    center_y_test = 0.65   # Source center y-coordinate
    T_max = 1.0            # Training time horizon
    
    print(f"\nTest case: {test_case}")
    print(f"Training epochs: {n_epochs}")
    print(f"IC sensors: {n_sensors_ic}x{n_sensors_ic} = {n_sensors_ic**2}")
    print(f"Source sensors: {n_sensors_src}x{n_sensors_src} = {n_sensors_src**2}")
    print(f"Collocation points: {n_colloc}")
    print(f"Network sizes: branch_hidden={branch_hidden}, trunk_hidden={trunk_hidden}, p={p}")
    
    # ============================================================
    # INITIALIZE MODEL
    # ============================================================
    
    model = PINNDeepONet_Wave2D(
        n_sensors_ic=n_sensors_ic,
        n_sensors_src=n_sensors_src,
        branch_hidden=branch_hidden,
        trunk_hidden=trunk_hidden,
        p=p
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # ============================================================
    # LOAD GROUND TRUTH (if available)
    # ============================================================
    
    if test_case == 'with_source':
        gt_file = 'data/gt_wave2D_withsource.csv'
        gt_rollout_file = 'data/gt_wave2D_withsource_rollout.csv'
    else:
        gt_file = 'data/gt_wave2D_nosource.csv'
        gt_rollout_file = 'data/gt_wave2D_nosource_rollout.csv'
    
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
    
    checkpoint_path = f'checkpoints/pinn_wave2D_{test_case}.pth'
    
    if load_model:
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {checkpoint_path}")
        else:
            print(f"Error: Checkpoint file {checkpoint_path} not found!")
            print("Set load_model=False to train a new model.")
            return
    else:
        
        if test_case == 'with_source':
            source_type = 'gaussian'
        else:
            source_type = 'zero'
        
        history = model.train_pinn(
            n_epochs=n_epochs,
            n_colloc=n_colloc,
            T_max=T_max,
            source_type=source_type,
            lr=1e-3
        )
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
            'config': {
                'n_sensors_ic': n_sensors_ic,
                'n_sensors_src': n_sensors_src,
                'branch_hidden': branch_hidden,
                'trunk_hidden': trunk_hidden,
                'p': p,
                'test_case': test_case
            }
        }, checkpoint_path)
        
        print(f"\nModel saved to {checkpoint_path}")
        
        # ============================================================
        # PLOT TRAINING HISTORY
        # ============================================================
        
        fig_history = plot_training_history(history)
        fig_history.savefig(f'logs_multibranch_wave2D/training_history_wave2D_{test_case}.png', dpi=150, bbox_inches='tight')
        print(f"Training history saved to logs_multibranch_wave2D/training_history_wave2D_{test_case}.png")
    
    # ============================================================
    # PLOT SOLUTION SNAPSHOTS
    # ============================================================
    
    if test_case == 'with_source':
        source_type = 'gaussian'
    else:
        source_type = 'zero'
        
    if gt_data is not None:
        fig_solution = plot_solution_2d_comparison(
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
        fig_solution.savefig(f'logs_multibranch_wave2D/solution_comparison_wave2D_{test_case}.png', dpi=150, bbox_inches='tight')
        print(f"Solution comparison saved to logs_multibranch_wave2D/solution_comparison_wave2D_{test_case}.png")
    else:
        fig_solution = plot_solution_2d(
            model,
            a_test=a_test,
            b_test=b_test,
            source_type=source_type,
            source_amplitude=source_amplitude,
            center_x=center_x_test,
            center_y=center_y_test,
            T_max=T_max
        )
        fig_solution.savefig(f'logs_multibranch_wave2D/solution_snapshots_wave2D_{test_case}.png', dpi=150, bbox_inches='tight')
        print(f"Solution snapshots saved to logs_multibranch_wave2D/solution_snapshots_wave2D_{test_case}.png")
    
    # ============================================================
    # ROLLOUT TEST
    # ============================================================
    
    if gt_rollout_data is not None:
        print("\n" + "="*80)
        print("PERFORMING ROLLOUT TEST")
        print("="*80)
        
        results = rollout_test_2d(
            model,
            gt_rollout_data,
            n_intervals=10,
            dt_interval=1.0
        )
        
        # Plot rollout errors
        fig_errors = plot_rollout_errors_2d(results)
        fig_errors.savefig(f'logs_multibranch_wave2D/rollout_errors_wave2D_{test_case}.png', dpi=150, bbox_inches='tight')
        print(f"Rollout errors saved to logs_multibranch_wave2D/rollout_errors_wave2D_{test_case}.png")
        
        # Plot rollout snapshots
        fig_snapshots = plot_rollout_snapshots_2d(
            model, 
            results, 
            gt_rollout_data,
            snapshot_indices=[0, 4, 9]
        )
        fig_snapshots.savefig(f'outputs/rollout_snapshots_wave2D_{test_case}.png', dpi=150, bbox_inches='tight')
        print(f"Rollout snapshots saved to outputs/rollout_snapshots_wave2D_{test_case}.png")
    else:
        print("\nSkipping rollout test (no rollout ground truth available)")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nOutputs saved in:")
    print(f"  - checkpoints/")
    print(f"  - outputs/")

if __name__ == "__main__":
    main()
