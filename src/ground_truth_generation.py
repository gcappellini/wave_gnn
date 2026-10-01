"""
Ground Truth Data Generation via MATLAB Simulation

Calls wave_2D_matlab_reference.m to generate 2D wave equation solutions.
Handles MATLAB execution and output validation.
"""

import subprocess
import logging
import h5py
import numpy as np
from pathlib import Path

logger = logging.getLogger("wave_gnn")


def run_matlab_dataset_generation(
    matlab_script_path: str,
    output_mat_file: str,
    script_dir: str,
) -> None:
    """Run a MATLAB dataset script and verify that its output was created."""

    script_path = Path(matlab_script_path).resolve()
    output_path = Path(output_mat_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating ground truth with MATLAB: {script_path}")
    command = f"run('{script_path.as_posix()}')"
    result = subprocess.run(
        ["matlab", "-batch", command],
        cwd=script_dir,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.stdout:
        logger.info(result.stdout.rstrip())
    if result.stderr:
        logger.error(result.stderr.rstrip())
    if result.returncode != 0:
        raise RuntimeError(f"MATLAB dataset generation failed for {script_path}")
    if not output_path.exists():
        raise FileNotFoundError(f"MATLAB did not create expected dataset: {output_path}")

    logger.info(f"Created dataset: {output_path}")


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
    
    run_matlab_dataset_generation(matlab_script_path, output_mat_file, script_dir)
    
    # Load and validate output
    if not os.path.exists(output_mat_file):
        raise FileNotFoundError(f"MATLAB failed to create output: {output_mat_file}")
    
    # Load data to validate
    with h5py.File(output_mat_file, 'r') as f:
        u_fom = np.array(f['U_data']).T  # (Nx, Ny, Nt, N_samples)
        v_fom = np.array(f['V_data']).T  # (Nx, Ny, Nt, N_samples) - velocity
        
        Nx, Ny, Nt, N_samples = u_fom.shape
        
        logger.info(f"Displacement (U): {u_fom.shape}")
        logger.info(f"Velocity (V): {v_fom.shape}")
        logger.info(f"Grid: {Nx}x{Ny}, Time steps: {Nt}, Samples: {N_samples}")
        logger.info(f"Range U: [{u_fom.min():.6e}, {u_fom.max():.6e}]")
        logger.info(f"Range V: [{v_fom.min():.6e}, {v_fom.max():.6e}]")
    
    metadata = {
        'Nx': Nx,
        'Ny': Ny,
        'Nt': Nt,
        'N_samples': N_samples,
        'matlab_script': os.path.basename(matlab_script_path),
    }
    
    logger.info("Ground truth generation complete")
    
    return {
        'u_fom': u_fom,
        'v_fom': v_fom,
        'metadata': metadata,
    }
