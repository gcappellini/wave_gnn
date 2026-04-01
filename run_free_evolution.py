"""
Complete Pipeline for Free Evolution Problem

Orchestrates the full workflow:
1. Ground truth generation (MATLAB)
2. SVD basis extraction
3. Train trunk network
4. Train branch network
5. Optionally: Fine-tune DeepONet jointly
6. Validation and visualization
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
from omegaconf import DictConfig, OmegaConf, open_dict

# Import modules
from src.ground_truth_generation import generate_ground_truth
from src.svd_analysis import (
    extract_svd_basis,
    visualize_svd_analysis,
    compute_svd_magnitude_summary,
)
from src.training import train_trunk, train_branch, train_deeponet_joint
from src.plotting import plot_validation_basic, plot_trunk_validation, plot_branch_validation


def _round_to_significant(value: float, significant_digits: int = 6) -> float:
    """Round a float to a fixed number of significant digits."""

    return float(f"{value:.{significant_digits}g}")


def _round_magnitude_summary(summary: dict, significant_digits: int = 6) -> dict:
    """Round nested magnitude summary dictionary values."""

    rounded = {}
    for quantity_name, metrics in summary.items():
        rounded[quantity_name] = {
            metric_name: _round_to_significant(metric_value, significant_digits)
            for metric_name, metric_value in metrics.items()
        }
    return rounded


class TeeLogger:
    """Redirect stdout to both console and file with timestamps."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w')
    
    def write(self, message):
        # Add timestamp to non-empty lines
        lines = message.split('\n')
        timestamped_lines = []
        
        for line in lines:
            if line.strip():  # Non-empty line
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                timestamped_line = f"[{timestamp}] {line}"
            else:  # Empty line
                timestamped_line = line
            timestamped_lines.append(timestamped_line)
        
        timestamped_message = '\n'.join(timestamped_lines)
        
        self.terminal.write(timestamped_message)
        self.log.write(timestamped_message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


@hydra.main(version_base=None, config_path="configs/free_evolution", config_name="config")
def main(cfg: DictConfig):
    """
    Execute full pipeline for free evolution problem.
    
    Hydra automatically:
    - Loads config from configs/free_evolution/config.yaml
    - Creates timestamped output directory: outputs/YYYY-MM-DD/HH-MM-SS
    - Sets working directory to outputs/YYYY-MM-DD/HH-MM-SS
    """
    
    # Setup paths first to determine log location
    script_dir = Path(__file__).parent.absolute()
    data_dir = script_dir / "data"
    models_dir = script_dir / "models"
    hydra_cfg = HydraConfig.get()
    output_dir = Path(hydra_cfg.runtime.output_dir)

    problem_type = cfg.problem.name
    
    # Setup logging to both file and console
    log_file = output_dir / "run_free_evolution.log"
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # ====================================================================
    # 1. GROUND TRUTH GENERATION
    # ====================================================================
    matlab_script = script_dir / cfg.problem.matlab.script
    output_mat = data_dir / "free_evolution.mat"
    
    if not output_mat.exists():
        print("Step 1: Ground Truth Generation (MATLAB)")
        print("-" * 70)
        gt_data = generate_ground_truth(
            matlab_script_path=str(matlab_script),
            output_mat_file=str(output_mat),
            script_dir=str(script_dir),
            config=OmegaConf.to_container(cfg.problem.matlab)
        )
    else:
        print("Step 1: Loading pre-computed ground truth")
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
    v_fom = gt_data.get('v_fom', None)
    
    # ====================================================================
    # 2. SVD BASIS EXTRACTION
    # ====================================================================
    print("\nStep 2: SVD Basis Extraction")
    print("-" * 70)
    
    svd_output = data_dir / "svd_free_evolution.npy"
    
    if not svd_output.exists():
        svd_data = extract_svd_basis(
            u_fom=u_fom,
            n_modes=cfg.svd.n_modes,
            visualize=False,
            output_dir=str(output_dir),
            v_fom=v_fom,
        )
        np.save(svd_output, svd_data)
        print(f"✓ Saved: {svd_output}")
    else:
        print("Loading pre-computed SVD data...")
        svd_output_dict = np.load(svd_output, allow_pickle=True).item()
        svd_data = svd_output_dict
        print(f"✓ Loaded: {svd_output}")

    svd_magnitude_summary = None
    if cfg.svd.visualize:
        svd_magnitude_summary = visualize_svd_analysis(
            svd_data=svd_data,
            output_dir=str(output_dir),
            n_modes=cfg.svd.n_modes,
            u_fom=u_fom,
            v_fom=v_fom,
        )

    if not svd_magnitude_summary:
        svd_magnitude_summary = compute_svd_magnitude_summary(
            svd_data=svd_data,
            n_modes=cfg.svd.n_modes,
            u_fom=u_fom,
            v_fom=v_fom,
        )

    if svd_magnitude_summary:
        svd_magnitude_summary = _round_magnitude_summary(svd_magnitude_summary, significant_digits=6)

        with open_dict(cfg):
            cfg.svd.magnitude_summary = svd_magnitude_summary

        hydra_cfg_path = output_dir / ".hydra" / "config.yaml"
        OmegaConf.save(cfg, str(hydra_cfg_path))

        default_cfg_path = script_dir / "configs" / "free_evolution" / "config.yaml"
        default_cfg = OmegaConf.load(str(default_cfg_path))
        with open_dict(default_cfg):
            default_cfg.svd.magnitude_summary = svd_magnitude_summary
        OmegaConf.save(default_cfg, str(default_cfg_path))

        print(f"✓ Stored SVD magnitude summary in cfg: {hydra_cfg_path}")
        print(f"✓ Updated default config with SVD magnitude summary: {default_cfg_path}")
    
    # ====================================================================
    # 3. TRAIN TRUNK NETWORK
    # ====================================================================
    print("\nStep 3: Train Trunk Network")
    print("-" * 70)
    
    trunk_checkpoint = models_dir / "trunk_svd_free_evolution.pth"
    
    if cfg.training.trunk_n_epochs > 0 and not trunk_checkpoint.exists():
        if 'magnitude_summary' not in cfg.svd:
            raise ValueError(
                "cfg.svd.magnitude_summary is missing. "
                "Run Step 2 (SVD) first to populate scaling parameters."
            )

        # Build config with all required fields
        trunk_config = OmegaConf.to_container(cfg.training)
        trunk_config['n_modes'] = cfg.svd.n_modes
        trunk_config['trunk_hidden_dim'] = cfg.networks.trunk.hidden_dim
        trunk_config['trunk_n_layers'] = cfg.networks.trunk.n_layers
        trunk_config['svd_magnitude_summary'] = OmegaConf.to_container(
            cfg.svd.magnitude_summary,
            resolve=True,
        )
        
        trunk_result = train_trunk(
            config=trunk_config,
            svd_data=svd_data,
            device=device,
            output_dir=str(output_dir),
            models_dir=str(models_dir),
            problem_type=problem_type,
        )
    else:
        print("Loading pre-trained trunk model...")
        print(f"✓ Loaded: {trunk_checkpoint}")
        trunk_result = {'status': 'loaded_from_checkpoint'}
    if cfg.networks.trunk.visualize:
        plot_trunk_validation(output_dir, svd_data, models_dir, device, 
                            problem_type=problem_type)
    
    # ====================================================================
    # 4. TRAIN BRANCH NETWORK
    # ====================================================================
    print("\nStep 4: Train Branch Network")
    print("-" * 70)

    branch_reference_name = f"{problem_type}_branch"
    branch_mat_path = data_dir / f"{branch_reference_name}.mat"
    branch_svd_path = data_dir / f"svd_{branch_reference_name}.npy"

    if not branch_mat_path.exists():
        raise FileNotFoundError(
            f"Branch reference dataset not found: {branch_mat_path}. "
            f"Expected a .mat file named '{branch_reference_name}.mat'."
        )
    if not branch_svd_path.exists():
        raise FileNotFoundError(
            f"Branch reference SVD not found: {branch_svd_path}. "
            f"Expected an .npy file named 'svd_{branch_reference_name}.npy'."
        )

    print(f"Loading branch reference dataset: {branch_mat_path}")
    with h5py.File(branch_mat_path, 'r') as f:
        u_fom_branch = np.array(f['U_data']).T
        v_fom_branch = np.array(f['V_data']).T if 'V_data' in f else None

    print(f"Loading branch reference SVD: {branch_svd_path}")
    svd_data_branch = np.load(branch_svd_path, allow_pickle=True).item()

    branch_svd_magnitude_summary = compute_svd_magnitude_summary(
        svd_data=svd_data_branch,
        n_modes=cfg.svd.n_modes,
        u_fom=u_fom_branch,
        v_fom=v_fom_branch,
    )
    
    branch_checkpoint = models_dir / "branch_svd_free_evolution.pth"
    
    if cfg.training.branch_n_epochs > 0 and not branch_checkpoint.exists():
        if branch_svd_magnitude_summary is None:
            raise ValueError(
                "Branch SVD magnitude summary is missing. "
                "Provide svd_{problem_type}_branch.npy with the required metadata."
            )

        # Build config with all required fields
        branch_config = OmegaConf.to_container(cfg.training)
        branch_config['n_modes'] = cfg.svd.n_modes
        branch_config['n_sensors'] = cfg.sensors.n_sensors
        branch_config['branch_hidden_dim'] = cfg.networks.branch.hidden_dim
        branch_config['branch_n_layers'] = cfg.networks.branch.n_layers
        branch_config['svd_magnitude_summary'] = _round_magnitude_summary(
            branch_svd_magnitude_summary,
            significant_digits=6,
        )
        
        branch_result = train_branch(
            config=branch_config,
            u_fom=u_fom_branch,
            svd_data=svd_data_branch,
            device=device,
            output_dir=str(output_dir),
            models_dir=str(models_dir),
            problem_type='free_evolution',
            v_fom=v_fom_branch,
        )
    else:
        print("Loading pre-trained branch model...")
        print(f"✓ Loaded: {branch_checkpoint}")
        branch_result = {'status': 'loaded_from_checkpoint'}
    
    if cfg.networks.branch.visualize:
        print(output_dir)
        plot_branch_validation(
            output_dir=output_dir,
            u_fom=u_fom_branch,
            v_fom=v_fom_branch,
            svd_data=svd_data_branch,
            models_dir=models_dir,
            device=device,
            problem_type=problem_type,
        )
    
    # # ====================================================================
    # # 5. OPTIONAL: JOINT DEEPONET TRAINING
    # # ====================================================================
    # # if False:  # Toggle to enable
    # print("\nStep 5: Joint DeepONet Training")
    # print("-" * 70)
    
    # deeponet_checkpoint = models_dir / "deeponet_free_evolution.pth"
    
    # if cfg.training.deeponet_n_epochs > 0 and not deeponet_checkpoint.exists():
    #     # Build config with all required fields
    #     deeponet_config = OmegaConf.to_container(cfg.training)
    #     deeponet_config['n_modes'] = cfg.svd.n_modes
    #     deeponet_config['trunk_hidden_dim'] = cfg.networks.trunk.hidden_dim
    #     deeponet_config['trunk_n_layers'] = cfg.networks.trunk.n_layers
    #     deeponet_config['branch_hidden_dim'] = cfg.networks.branch.hidden_dim
    #     deeponet_config['branch_n_layers'] = cfg.networks.branch.n_layers
    #     deeponet_config['n_sensors'] = cfg.sensors.n_sensors
        
    #     trunk_pretrained = output_dir / "trunk_svd_free_evolution.pth"
    #     branch_pretrained = output_dir / "branch_svd_free_evolution.pth"
    #     deeponet_result = train_deeponet_joint(
    #         config=deeponet_config,
    #         u_fom=u_fom,
    #         svd_data=svd_data,
    #         trunk_pretrained_path=str(trunk_pretrained),
    #         branch_pretrained_path=str(branch_pretrained),
    #         device=device,
    #         output_dir=str(output_dir),
    #         models_dir=str(models_dir),
    #         problem_type='free_evolution',
    #         v_fom=v_fom,
    #     )
    # else:
    #     print("Loading pre-trained DeepONet model...")
    #     print(f"✓ Loaded: {deeponet_checkpoint}")
    #     deeponet_result = {'status': 'loaded_from_checkpoint'}
    
    # # ====================================================================
    # # 6. VALIDATION & VISUALIZATION
    # # ====================================================================
    # print("\nStep 6: Validation & Visualization")
    # print("-" * 70)
    
    # plot_validation_basic(
    #     output_dir=str(output_dir),
    #     u_fom=u_fom,
    #     v_fom=v_fom,
    #     svd_data=svd_data,
    #     data_dir=str(data_dir),
    #     models_dir=str(models_dir),
    #     device=device,
    #     n_samples_plot=3,
    #     cfg=cfg,
    #     problem_type='free_evolution'
    # )
    
    # # ====================================================================
    # # COMPLETE
    # # ====================================================================
    # print("\n" + "=" * 70)
    # print("✓ PIPELINE COMPLETE")
    # print("=" * 70)
    # print(f"Output directory: {output_dir}")
    # print("=" * 70 + "\n")
    
    # # Close log file
    # sys.stdout = tee.terminal
    # tee.close()


if __name__ == "__main__":
    main()
