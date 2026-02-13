"""
Ground Truth Data Generation via MATLAB Simulation

Calls wave_2D_matlab_reference.m to generate 2D wave equation solutions.
Handles MATLAB execution and output validation.
"""

import os
import subprocess
import h5py
import numpy as np
from pathlib import Path


def generate_ground_truth(
    matlab_script_path: str,
    output_mat_file: str,
    script_dir: str,
    config: dict = None,
) -> dict:
    """
    Execute MATLAB simulation to generate 2D wave equation data.
    
    Args:
        matlab_script_path: Path to wave_2D_matlab_reference.m
        output_mat_file: Output path for test_cases.mat
        script_dir: Root script directory
        config: Optional config dict with simulation parameters
                (for future: constant_force case)
    
    Returns:
        dict with keys: {'u_fom': ndarray, 'v_fom': ndarray, 'metadata': dict}
    """
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_mat_file), exist_ok=True)
    
    print("=" * 70)
    print("GENERATING GROUND TRUTH DATA (2D Wave Equation via MATLAB)")
    print("=" * 70)
    print(f"\nScript: {matlab_script_path}")
    print(f"Output: {output_mat_file}")
    
    # Call MATLAB
    matlab_cmd = f"cd '{script_dir}'; {os.path.basename(matlab_script_path)}(1); quit"
    
    try:
        print(f"\nExecuting MATLAB...")
        result = subprocess.run(
            ["matlab", "-batch", matlab_cmd],
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode != 0:
            print(f"WARNING: MATLAB returned code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
        else:
            print("✓ MATLAB execution completed")
    
    except FileNotFoundError:
        print("ERROR: MATLAB not found. Please ensure MATLAB is installed and in PATH.")
        raise
    except subprocess.TimeoutExpired:
        print("ERROR: MATLAB simulation timed out (>600s)")
        raise
    
    # Load and validate output
    if not os.path.exists(output_mat_file):
        raise FileNotFoundError(f"MATLAB failed to create output: {output_mat_file}")
    
    print(f"\n✓ Output file created: {output_mat_file}")
    
    # Load data to validate
    print("\nLoading and validating data...")
    with h5py.File(output_mat_file, 'r') as f:
        u_fom = np.array(f['U_data']).T  # (Nx, Ny, Nt, N_samples)
        v_fom = np.array(f['V_data']).T  # (Nx, Ny, Nt, N_samples) - velocity
        
        Nx, Ny, Nt, N_samples = u_fom.shape
        
        print(f"  Displacement (U): {u_fom.shape}")
        print(f"  Velocity (V): {v_fom.shape}")
        print(f"  Grid: {Nx}×{Ny}, Time steps: {Nt}, Samples: {N_samples}")
        print(f"  Range U: [{u_fom.min():.6e}, {u_fom.max():.6e}]")
        print(f"  Range V: [{v_fom.min():.6e}, {v_fom.max():.6e}]")
    
    metadata = {
        'Nx': Nx,
        'Ny': Ny,
        'Nt': Nt,
        'N_samples': N_samples,
        'matlab_script': os.path.basename(matlab_script_path),
    }
    
    print("\n✓ Ground truth generation complete")
    print("=" * 70)
    
    return {
        'u_fom': u_fom,
        'v_fom': v_fom,
        'metadata': metadata,
    }
