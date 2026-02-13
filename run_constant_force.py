"""
Complete Pipeline for Constant Force Problem

Orchestrates the full workflow for 2D wave equation WITH forcing:
1. Ground truth generation (MATLAB with forcing)
2. SVD basis extraction
3. Train trunk network
4. Train branch network
5. Optionally: Fine-tune DeepONet jointly
6. Validation and visualization

STRUCTURE: Identical to run_free_evolution.py, only config differs
"""

import os
import sys
import h5py
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# Import modules
from src.ground_truth_generation import generate_ground_truth
from src.svd_analysis import extract_svd_basis
from src.training import train_trunk, train_branch, train_deeponet_joint
from src.plotting import plot_validation_basic


class TeeLogger:
    """Redirect stdout to both console and file."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


@hydra.main(version_base=None, config_path="configs/constant_force", config_name="config")
def main(cfg: DictConfig):
    """
    Execute full pipeline for constant force problem.
    
    Hydra automatically:
    - Loads config from configs/constant_force/config.yaml
    - Creates timestamped output directory: outputs/YYYY-MM-DD/HH-MM-SS
    - Sets working directory to outputs/YYYY-MM-DD/HH-MM-SS
    """
    
    # Setup paths first to determine log location
    script_dir = Path(__file__).parent.absolute()
    data_dir = script_dir / "data"
    models_dir = script_dir / "models"
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    
    # Setup logging to both file and console
    log_file = Path(output_dir) / "run_constant_force.log"
    tee = TeeLogger(log_file)
    sys.stdout = tee
    
    print("\n" + "=" * 70)
    print(f"DEEPONET PIPELINE: {cfg.problem.name.upper()}")
    print("=" * 70)
    print(f"\nConfig:\n{OmegaConf.to_yaml(cfg)}")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"\nWorking directory (output): {output_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Models directory: {models_dir}")
    print(f"Log file: {log_file}")
    print(f"Models directory: {models_dir}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # ====================================================================
    # 1. GROUND TRUTH GENERATION
    # ====================================================================
    matlab_script = script_dir / cfg.problem.matlab.script
    output_mat = data_dir / "test_cases_constant_force.mat"  # Different filename
    
    if not output_mat.exists():
        print("Step 1: Ground Truth Generation (MATLAB with forcing)")
        print("-" * 70)
        gt_data = generate_ground_truth(
            matlab_script_path=str(matlab_script),
            output_mat_file=str(output_mat),
            script_dir=str(script_dir),
            config=OmegaConf.to_container(cfg.problem.matlab)
        )
    else:
        print("Step 1: Loading pre-computed ground truth (constant force)")
        print("-" * 70)
        with h5py.File(output_mat, 'r') as f:
            u_fom = np.array(f['U_data']).T
            v_fom = np.array(f['V_data']).T
        
        Nx, Ny, Nt, N_samples = u_fom.shape
        gt_data = {
            'u_fom': u_fom,
            'v_fom': v_fom,
            'metadata': {
                'Nx': Nx, 'Ny': Ny, 'Nt': Nt, 'N_samples': N_samples
            }
        }
        print(f"✓ Loaded: {output_mat}")
        print(f"  Shape: {u_fom.shape}")
    
    u_fom = gt_data['u_fom']
    
    # ====================================================================
    # 2-6: IDENTICAL TO FREE EVOLUTION
    # ====================================================================
    # (See run_free_evolution.py for full documentation)
    
    print("\nStep 2: SVD Basis Extraction")
    print("-" * 70)
    
    svd_output = data_dir / "svd_basis_data_constant_force.npy"  # Different filename
    
    if not svd_output.exists():
        svd_data = extract_svd_basis(
            u_fom=u_fom,
            n_modes=cfg.svd.n_modes,
            visualize=cfg.svd.visualize,
            output_dir=str(output_dir)
        )
        
        output_dict = {
            'basis': svd_data['basis'],
            'singular_values': svd_data['singular_values'],
            'coefficients': svd_data['coefficients'],
            'grid_info': svd_data['grid_info'],
        }
        np.save(svd_output, output_dict)
        print(f"✓ Saved: {svd_output}")
    else:
        print("Loading pre-computed SVD data...")
        svd_output_dict = np.load(svd_output, allow_pickle=True).item()
        svd_data = svd_output_dict
        print(f"✓ Loaded: {svd_output}")
    
    print("\nStep 3: Train Trunk Network")
    print("-" * 70)
    
    trunk_result = train_trunk(
        config=OmegaConf.to_container(cfg.training),
        svd_data=svd_data,
        device=device,
        output_dir=str(output_dir),
            models_dir=str(models_dir),
    branch_config = OmegaConf.to_container(cfg.training)
    branch_config['n_modes'] = cfg.networks.branch.output_dim
    branch_config['n_sensors'] = cfg.sensors.n_sensors
    branch_config['batch_size'] = cfg.training.batch_size
    branch_config['branch_hidden_dim'] = cfg.networks.branch.hidden_dim
    branch_config['branch_n_layers'] = cfg.networks.branch.n_layers
    
    branch_result = train_branch(
        config=branch_config,
        u_fom=u_fom,
        svd_data=svd_data,
        device=device,
        output_dir=str(output_dir),
            models_dir=str(models_dir),
        output_dir=str(output_dir),
        u_fom=u_fom,
        svd_data=svd_data,
        data_dir=str(data_dir),
        models_dir=str(models_dir),
        device=device,
        n_samples_plot=3
    )
    
    # ====================================================================
    # COMPLETE
    # ====================================================================
    print("\n" + "=" * 70)
    print("✓ CONSTANT FORCE PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print("=" * 70 + "\n")
    
    # Close log file
    sys.stdout = tee.terminal
    tee.close()


if __name__ == "__main__":
    main()
