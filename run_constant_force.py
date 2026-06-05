"""Complete pipeline for the merged wave dataset.

Orchestrates the full workflow:
1. Load pre-merged dataset
2. SVD basis extraction
3. Train trunk network
4. Train branch network
5. Optionally: Fine-tune DeepONet jointly
6. Validation and visualization
"""

import os
import sys
import shutil
import h5py
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from src.svd_analysis import (
    extract_svd_basis,
    visualize_svd_analysis,
    compute_svd_magnitude_summary,
)
from src.training import train_trunk, train_branch, train_deeponet_joint
from src.plotting import (
    plot_deeponet_validation,
    plot_deeponet_rollout_validation,
    plot_trunk_validation,
    plot_branch_validation,
)
from src.models import DeepONet, DualHeadMLP, DualHeadSensorBranch


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


def _load_dual_trunk_from_checkpoint(trunk_ckpt_path: Path, device: torch.device) -> tuple[DualHeadMLP, dict]:
    """Load a dual-head trunk model from checkpoint and infer architecture."""

    trunk_ckpt = torch.load(str(trunk_ckpt_path), map_location=device, weights_only=False)
    trunk_state = trunk_ckpt.get('model_state_dict', trunk_ckpt)

    if 'backbone.0.weight' not in trunk_state or 'head_u.weight' not in trunk_state or 'head_v.weight' not in trunk_state:
        raise ValueError(
            f"Checkpoint {trunk_ckpt_path} is not a dual-head trunk checkpoint. "
            "Expected keys: backbone.0.weight, head_u.weight, head_v.weight"
        )

    input_dim = trunk_state['backbone.0.weight'].shape[1]
    hidden_dim = trunk_state['backbone.0.weight'].shape[0]
    n_modes = trunk_state['head_u.weight'].shape[0]
    n_backbone_linears = len([k for k in trunk_state.keys() if k.startswith('backbone.') and k.endswith('.weight')])
    n_layers = n_backbone_linears + 1

    u_output_scale = float(trunk_ckpt.get('u_output_scale', 1.0) or 1.0)
    v_output_scale = float(trunk_ckpt.get('v_output_scale', 1.0) or 1.0)

    trunk = DualHeadMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_modes=n_modes,
        n_layers=n_layers,
        u_output_scale=u_output_scale,
        v_output_scale=v_output_scale,
    ).to(device)
    trunk.load_state_dict(trunk_state)
    trunk.eval()

    metadata = {
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'n_modes': n_modes,
        'n_layers': n_layers,
        'u_output_scale': u_output_scale,
        'v_output_scale': v_output_scale,
    }
    return trunk, metadata


def _load_dual_branch_from_checkpoint(
    branch_ckpt_path: Path,
    device: torch.device,
    n_sensors: int,
) -> tuple[torch.nn.Module, dict]:
    """Load a dual-head branch model (sensor-encoder or legacy MLP) from checkpoint."""

    branch_ckpt = torch.load(str(branch_ckpt_path), map_location=device, weights_only=False)
    branch_state = branch_ckpt.get('model_state_dict', branch_ckpt)

    input_scale = float(branch_ckpt.get('input_scale', 1.0) or 1.0)
    u_output_scale = float(branch_ckpt.get('u_output_scale', 1.0) or 1.0)
    v_output_scale = float(branch_ckpt.get('v_output_scale', 1.0) or 1.0)

    if any(k.startswith('encoder.') for k in branch_state.keys()):
        if 'mlp.backbone.0.weight' not in branch_state or 'mlp.head_u.weight' not in branch_state:
            raise ValueError(
                f"Checkpoint {branch_ckpt_path} has encoder keys but is missing dual MLP head keys."
            )

        hidden_dim = branch_state['mlp.backbone.0.weight'].shape[0]
        n_modes = branch_state['mlp.head_u.weight'].shape[0]
        n_backbone_linears = len([
            k for k in branch_state.keys()
            if k.startswith('mlp.backbone.') and k.endswith('.weight')
        ])
        n_layers = n_backbone_linears + 1
        encoder_channels = branch_state['encoder.0.weight'].shape[0]
        input_channels = branch_state['encoder.0.weight'].shape[1]

        branch = DualHeadSensorBranch(
            n_sensors=n_sensors,
            hidden_dim=hidden_dim,
            n_modes=n_modes,
            n_layers=n_layers,
            input_scale=input_scale,
            input_channels=input_channels,
            encoder_channels=encoder_channels,
            u_output_scale=u_output_scale,
            v_output_scale=v_output_scale,
        ).to(device)
        branch.load_state_dict(branch_state)
        branch.eval()

        metadata = {
            'arch': 'DualHeadSensorBranch',
            'hidden_dim': hidden_dim,
            'n_modes': n_modes,
            'n_layers': n_layers,
            'input_scale': input_scale,
            'u_output_scale': u_output_scale,
            'v_output_scale': v_output_scale,
            'encoder_channels': encoder_channels,
            'input_channels': input_channels,
        }
        return branch, metadata

    if 'backbone.0.weight' in branch_state and 'head_u.weight' in branch_state and 'head_v.weight' in branch_state:
        input_dim = branch_state['backbone.0.weight'].shape[1]
        hidden_dim = branch_state['backbone.0.weight'].shape[0]
        n_modes = branch_state['head_u.weight'].shape[0]
        n_backbone_linears = len([k for k in branch_state.keys() if k.startswith('backbone.') and k.endswith('.weight')])
        n_layers = n_backbone_linears + 1

        branch = DualHeadMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_modes=n_modes,
            n_layers=n_layers,
            input_scale=input_scale,
            u_output_scale=u_output_scale,
            v_output_scale=v_output_scale,
        ).to(device)
        branch.load_state_dict(branch_state)
        branch.eval()

        metadata = {
            'arch': 'DualHeadMLP',
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'n_modes': n_modes,
            'n_layers': n_layers,
            'input_scale': input_scale,
            'u_output_scale': u_output_scale,
            'v_output_scale': v_output_scale,
        }
        return branch, metadata

    raise ValueError(
        f"Checkpoint {branch_ckpt_path} is not a supported dual-head branch checkpoint. "
        "Expected either sensor-encoder keys (encoder.* + mlp.head_u/head_v) or dual MLP keys "
        "(backbone.* + head_u/head_v)."
    )


def _compose_dual_deeponet_checkpoint(
    trunk_ckpt_path: Path,
    branch_ckpt_path: Path,
    deeponet_ckpt_path: Path,
    device: torch.device,
    n_sensors: int,
):
    """Compose and save a dual-head DeepONet checkpoint from pretrained trunk and branch."""

    trunk, trunk_meta = _load_dual_trunk_from_checkpoint(trunk_ckpt_path, device)
    branch, branch_meta = _load_dual_branch_from_checkpoint(branch_ckpt_path, device, n_sensors)

    # Extract sensor normalization stored in the branch checkpoint
    branch_ckpt_raw = torch.load(str(branch_ckpt_path), map_location='cpu', weights_only=False)
    input_norm = branch_ckpt_raw.get('input_normalization', {})
    normalization = {
        'raw_u_min': float(input_norm.get('raw_u_min', 0.0)),
        'raw_u_max': float(input_norm.get('raw_u_max', 1.0)),
        'raw_v_min': float(input_norm.get('raw_v_min', 0.0)),
        'raw_v_max': float(input_norm.get('raw_v_max', 1.0)),
        'raw_f_min': float(input_norm.get('raw_f_min', 0.0)),
        'raw_f_max': float(input_norm.get('raw_f_max', 1.0)),
    }

    deeponet = DeepONet(
        trunk=trunk,
        branch_ic=branch,
        problem_type='free_evolution',
    ).to(device)
    deeponet.eval()

    deeponet_ckpt = {
        'model_state_dict': deeponet.state_dict(),
        'config': {
            'n_modes': trunk_meta['n_modes'],
            'n_sensors': n_sensors,
            'trunk_hidden_dim': trunk_meta['hidden_dim'],
            'trunk_n_layers': trunk_meta['n_layers'],
            'branch_hidden_dim': branch_meta['hidden_dim'],
            'branch_n_layers': branch_meta['n_layers'],
            'branch_architecture': branch_meta['arch'],
            'n_input_channels': int(branch_meta.get('input_channels', 2)),
        },
        'dual': True,
        'input_scale': float(branch_meta.get('input_scale', 1.0)),
        'u_output_scale': float(trunk_meta.get('u_output_scale', 1.0)),
        'v_output_scale': float(trunk_meta.get('v_output_scale', 1.0)),
        'normalization': normalization,
        'composed_from_pretrained': True,
        'source_checkpoints': {
            'trunk': str(trunk_ckpt_path),
            'branch': str(branch_ckpt_path),
        },
    }

    deeponet_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(deeponet_ckpt, str(deeponet_ckpt_path))

    print(f"✓ Composed dual-head DeepONet from pretrained checkpoints")
    print(f"  Trunk checkpoint: {trunk_ckpt_path}")
    print(f"  Branch checkpoint: {branch_ckpt_path}")
    print(f"  Normalization: u=[{normalization['raw_u_min']:.4e}, {normalization['raw_u_max']:.4e}]"
          f", v=[{normalization['raw_v_min']:.4e}, {normalization['raw_v_max']:.4e}]")
    print(f"  Saved composed checkpoint: {deeponet_ckpt_path}")


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


@hydra.main(version_base=None, config_path="configs/constant_force", config_name="config")
def main(cfg: DictConfig):
    run_constant_force_pipeline(cfg)


def run_constant_force_pipeline(
    cfg: DictConfig,
    output_dir: Path = None,
    pretrained_models_dir: Path = None,
    log_file_name: str = "run_constant_force.log",
    use_tee_logger: bool = True,
    save_default_config_summary: bool = True,
    run_validation: bool = True,
):
    """
    Execute full pipeline for merged dataset training.
    
    When called from Hydra entrypoints, output_dir defaults to Hydra runtime output.
    This function can also be reused by curriculum runners with explicit output_dir
    and chained pretrained_models_dir.
    """

    script_dir = Path(__file__).parent.absolute()
    data_dir = script_dir / "data"

    if output_dir is None:
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)
    else:
        output_dir = Path(output_dir)

    if pretrained_models_dir is None:
        pretrained_models_dir = script_dir / "models"
    else:
        pretrained_models_dir = Path(pretrained_models_dir)

    models_dir = output_dir / "checkpoints"
    canonical_models_dir = script_dir / "models"

    problem_type = cfg.problem.name

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    tee = None
    original_stdout = sys.stdout
    log_file = output_dir / log_file_name

    try:
        if use_tee_logger:
            tee = TeeLogger(log_file)
            sys.stdout = tee

        print("\n" + "=" * 70)
        print(f"DEEPONET PIPELINE: {cfg.problem.name.upper()}")
        print("=" * 70)
        print(f"\nConfig:\n{OmegaConf.to_yaml(cfg)}")

        print(f"\nWorking directory (output): {output_dir}")
        print(f"Data directory: {data_dir}")
        print(f"Pretrained models directory: {pretrained_models_dir}")
        print(f"Checkpoints directory: {models_dir}")
        print(f"Canonical models directory: {canonical_models_dir}")
        if use_tee_logger:
            print(f"Log file: {log_file}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}\n")

        # ====================================================================
        # 1. DATASET LOADING
        # ====================================================================
        print("Step 1: Loading merged dataset")
        print("-" * 70)

        merged_path = data_dir / "merged.mat"
        if not merged_path.exists():
            raise FileNotFoundError(
                f"Merged dataset not found: {merged_path}. "
                "Run data/merge_dataset.py first."
            )

        with h5py.File(merged_path, 'r') as f:
            if 'U_data' not in f or 'V_data' not in f:
                raise ValueError(
                    f"Missing required datasets in {merged_path}. Expected U_data and V_data"
                )

            u_fom = np.array(f['U_data']).T
            v_fom = np.array(f['V_data']).T
            f_data = np.array(f['F_data']).T if 'F_data' in f else None
            source_centers = np.array(f['source_centers']).T if 'source_centers' in f else None
            source_signs = np.array(f['source_signs']).T if 'source_signs' in f else None

        if u_fom.shape != v_fom.shape:
            raise ValueError(
                f"U_data/V_data shape mismatch in {merged_path}: {u_fom.shape} vs {v_fom.shape}"
            )

        if f_data is None or source_centers is None or source_signs is None:
            raise ValueError(
                f"Missing required merged keys in {merged_path}. "
                "Expected: F_data, source_centers, source_signs"
            )

        Nx, Ny, Nt, N_samples = u_fom.shape
        gt_data = {
            'u_fom': u_fom,
            'v_fom': v_fom,
            'f_data': f_data,
            'source_centers': source_centers,
            'source_signs': source_signs,
            'metadata': {
                'Nx': Nx,
                'Ny': Ny,
                'Nt': Nt,
                'N_samples': N_samples,
                'source_datasets': ['constant_force', 'forced_sine_dataset', 'free_evolution'],
            }
        }

        print(f"✓ Loaded merged dataset: U{u_fom.shape}, V{v_fom.shape}, F{f_data.shape}")
        print(f"  Source metadata: centers{source_centers.shape}, signs{source_signs.shape}")

        u_fom = gt_data['u_fom']
        v_fom = gt_data.get('v_fom', None)
        f_data = gt_data.get('f_data', None)

        # ====================================================================
        # 2. SVD BASIS EXTRACTION
        # ====================================================================
        print("\nStep 2: SVD Basis Extraction")
        print("-" * 70)

        svd_output = data_dir / "svd_merged.npy"

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
                f_fom=f_data,
            )

        if svd_magnitude_summary:
            svd_magnitude_summary = _round_magnitude_summary(svd_magnitude_summary, significant_digits=6)

            with open_dict(cfg):
                cfg.svd.magnitude_summary = svd_magnitude_summary

            hydra_cfg_path = output_dir / ".hydra" / "config.yaml"
            hydra_cfg_path.parent.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(cfg, str(hydra_cfg_path))

            if save_default_config_summary:
                default_cfg_path = script_dir / "configs" / "constant_force" / "config.yaml"
                default_cfg = OmegaConf.load(str(default_cfg_path))
                with open_dict(default_cfg):
                    default_cfg.svd.magnitude_summary = svd_magnitude_summary
                OmegaConf.save(default_cfg, str(default_cfg_path))
                print(f"✓ Updated default config with SVD magnitude summary: {default_cfg_path}")

            print(f"✓ Stored SVD magnitude summary in cfg: {hydra_cfg_path}")

        # ====================================================================
        # 3. TRAIN TRUNK NETWORK
        # ====================================================================
        print("\nStep 3: Train Trunk Network")
        print("-" * 70)

        trunk_checkpoint = pretrained_models_dir / f"trunk_svd_{problem_type}.pth"

        if cfg.training.trunk_n_epochs > 0 and not trunk_checkpoint.exists():
            if 'magnitude_summary' not in cfg.svd:
                raise ValueError(
                    "cfg.svd.magnitude_summary is missing. "
                    "Run Step 2 (SVD) first to populate scaling parameters."
                )

            trunk_config = OmegaConf.to_container(cfg.training)
            trunk_config['n_modes'] = cfg.svd.n_modes
            trunk_config['trunk_hidden_dim'] = cfg.networks.trunk.hidden_dim
            trunk_config['trunk_n_layers'] = cfg.networks.trunk.n_layers
            trunk_config['svd_magnitude_summary'] = OmegaConf.to_container(
                cfg.svd.magnitude_summary,
                resolve=True,
            )

            train_trunk(
                config=trunk_config,
                svd_data=np.load(data_dir / "svd_merged.npy", allow_pickle=True).item(),
                device=device,
                output_dir=str(output_dir),
                models_dir=str(models_dir),
                problem_type=problem_type,
            )
        else:
            print("Loading pre-trained trunk model...")
            print(f"✓ Loaded: {trunk_checkpoint}")
            if trunk_checkpoint.exists():
                models_dir.mkdir(parents=True, exist_ok=True)
                stage_trunk_checkpoint = models_dir / f"trunk_svd_{problem_type}.pth"
                if stage_trunk_checkpoint.resolve() != trunk_checkpoint.resolve():
                    shutil.copy2(trunk_checkpoint, stage_trunk_checkpoint)
                    print(f"✓ Copied trunk checkpoint to stage checkpoints: {stage_trunk_checkpoint}")

        # Keep a canonical trunk checkpoint in repo-level models for curriculum reuse.
        trained_trunk_checkpoint = models_dir / f"trunk_svd_{problem_type}.pth"
        canonical_trunk_checkpoint = script_dir / "models" / "trunk_svd_merged.pth"
        if trained_trunk_checkpoint.exists():
            canonical_trunk_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(trained_trunk_checkpoint, canonical_trunk_checkpoint)
            print(f"✓ Saved trunk checkpoint copy: {canonical_trunk_checkpoint}")

        if cfg.networks.trunk.visualize:
            svd_data_trunk_val = np.load(data_dir / "svd_merged.npy", allow_pickle=True).item()
            plot_trunk_validation(output_dir, svd_data_trunk_val, canonical_models_dir, device,
                                  problem_type=problem_type)

        # ====================================================================
        # 4. TRAIN BRANCH NETWORK
        # ====================================================================
        print("\nStep 4: Train Branch Network")
        print("-" * 70)

        if f_data is None:
            raise ValueError("merged pipeline requires F_data in merged.mat")

        branch_checkpoint = pretrained_models_dir / f"branch_svd_{problem_type}.pth"

        if cfg.training.branch_n_epochs > 0 and not branch_checkpoint.exists():
            branch_mat_path = data_dir / "merged.mat"
            branch_svd_path = data_dir / "svd_merged.npy"

            if not branch_mat_path.exists():
                raise FileNotFoundError(
                    f"Branch reference dataset not found: {branch_mat_path}. "
                    "Expected data/merged.mat."
                )
            if not branch_svd_path.exists():
                raise FileNotFoundError(
                    f"Branch reference SVD not found: {branch_svd_path}. "
                    "Expected data/svd_merged.npy."
                )

            print(f"Loading branch reference dataset: {branch_mat_path}")
            with h5py.File(branch_mat_path, 'r') as f:
                u_fom_branch = np.array(f['U_data']).T
                v_fom_branch = np.array(f['V_data']).T if 'V_data' in f else None
                f_data_branch = np.array(f['F_data']).T if 'F_data' in f else None

            print(f"Loading branch reference SVD: {branch_svd_path}")
            svd_data_branch = np.load(branch_svd_path, allow_pickle=True).item()

            branch_svd_magnitude_summary = compute_svd_magnitude_summary(
                svd_data=svd_data_branch,
                n_modes=cfg.svd.n_modes,
                u_fom=u_fom_branch,
                v_fom=v_fom_branch,
                f_fom=f_data_branch,
            )

            if branch_svd_magnitude_summary is None:
                raise ValueError(
                    "Branch SVD magnitude summary is missing. "
                    "Provide svd_{problem_type}.npy with the required metadata."
                )

            branch_config = OmegaConf.to_container(cfg.training)
            branch_config['n_modes'] = cfg.svd.n_modes
            branch_config['n_sensors'] = cfg.sensors.n_sensors
            branch_config['branch_hidden_dim'] = cfg.networks.branch.hidden_dim
            branch_config['branch_n_layers'] = cfg.networks.branch.n_layers
            branch_config['svd_magnitude_summary'] = _round_magnitude_summary(
                branch_svd_magnitude_summary,
                significant_digits=6,
            )

            train_branch(
                config=branch_config,
                u_fom=u_fom_branch,
                svd_data=svd_data_branch,
                device=device,
                output_dir=str(output_dir),
                models_dir=str(models_dir),
                problem_type=problem_type,
                v_fom=v_fom_branch,
                f_fom=f_data_branch,
            )
        else:
            print("Loading pre-trained branch model...")
            print(f"✓ Loaded: {branch_checkpoint}")
            if branch_checkpoint.exists():
                models_dir.mkdir(parents=True, exist_ok=True)
                stage_branch_checkpoint = models_dir / f"branch_svd_{problem_type}.pth"
                if stage_branch_checkpoint.resolve() != branch_checkpoint.resolve():
                    shutil.copy2(branch_checkpoint, stage_branch_checkpoint)
                    print(f"✓ Copied branch checkpoint to stage checkpoints: {stage_branch_checkpoint}")

        # Keep a canonical branch checkpoint in repo-level models for curriculum reuse.
        trained_branch_checkpoint = models_dir / f"branch_svd_{problem_type}.pth"
        canonical_branch_checkpoint = script_dir / "models" / "branch_svd_merged.pth"
        if trained_branch_checkpoint.exists():
            canonical_branch_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(trained_branch_checkpoint, canonical_branch_checkpoint)
            print(f"✓ Saved branch checkpoint copy: {canonical_branch_checkpoint}")

        if cfg.networks.branch.visualize:
            branch_mat_path = data_dir / "merged.mat"
            print(f"Loading branch reference dataset: {branch_mat_path}")
            with h5py.File(branch_mat_path, 'r') as f:
                u_fom_branch = np.array(f['U_data']).T
                v_fom_branch = np.array(f['V_data']).T if 'V_data' in f else None
                f_data_branch = np.array(f['F_data']).T if 'F_data' in f else None
            branch_svd_path = data_dir / "svd_merged.npy"
            svd_data_branch = np.load(branch_svd_path, allow_pickle=True).item()

            plot_branch_validation(
                output_dir=output_dir,
                u_fom=u_fom_branch,
                v_fom=v_fom_branch,
                f_fom=f_data_branch,
                svd_data=svd_data_branch,
                models_dir=canonical_models_dir,
                device=device,
                problem_type=problem_type,
            )

        # ====================================================================
        # 5. OPTIONAL: JOINT DEEPONET TRAINING
        # ====================================================================
        print("\nStep 5: Joint DeepONet Training")
        print("-" * 70)

        svd_data_branch = np.load(data_dir / "svd_merged.npy", allow_pickle=True).item()

        if v_fom is None or 'basis_v' not in svd_data_branch:
            raise ValueError(
                "Constant-force DeepONet requires dual-head data (v_fom and svd_data['basis_v']). "
                "Single-head branch/trunk is no longer supported in this pipeline."
            )

        deeponet_checkpoint = pretrained_models_dir / f"deeponet_{problem_type}.pth"

        trunk_pretrained = pretrained_models_dir / f"trunk_svd_{problem_type}.pth"
        branch_pretrained = pretrained_models_dir / f"branch_svd_{problem_type}.pth"

        finetune_deeponet = bool(cfg.training.get('finetune_deeponet', False))
        deeponet_init_ckpt = deeponet_checkpoint if deeponet_checkpoint.exists() else None

        if deeponet_checkpoint.exists() and not finetune_deeponet:
            print("Loading pre-trained DeepONet model...")
            print(f"✓ Loaded: {deeponet_checkpoint}")
        elif cfg.training.deeponet_n_epochs > 0:
            if finetune_deeponet:
                if deeponet_init_ckpt is not None:
                    print(f"Fine-tuning DeepONet from checkpoint: {deeponet_init_ckpt}")
                else:
                    print("finetune_deeponet=true but no existing DeepONet checkpoint found; training from trunk/branch initialization")

            deeponet_config = OmegaConf.to_container(cfg.training)
            deeponet_config['n_modes'] = cfg.svd.n_modes
            deeponet_config['trunk_hidden_dim'] = cfg.networks.trunk.hidden_dim
            deeponet_config['trunk_n_layers'] = cfg.networks.trunk.n_layers
            deeponet_config['branch_hidden_dim'] = cfg.networks.branch.hidden_dim
            deeponet_config['branch_n_layers'] = cfg.networks.branch.n_layers
            deeponet_config['n_sensors'] = cfg.sensors.n_sensors

            train_deeponet_joint(
                config=deeponet_config,
                u_fom=u_fom,
                svd_data=svd_data_branch,
                trunk_pretrained_path=str(trunk_pretrained),
                branch_pretrained_path=str(branch_pretrained),
                deeponet_pretrained_path=str(deeponet_init_ckpt) if deeponet_init_ckpt is not None else None,
                device=device,
                output_dir=str(output_dir),
                models_dir=str(models_dir),
                problem_type=problem_type,
                v_fom=v_fom,
                f_data=f_data,
            )
        else:
            print("DeepONet checkpoint not found; composing from pretrained dual-head trunk and branch...")
            if not trunk_pretrained.exists():
                raise FileNotFoundError(
                    f"Cannot compose DeepONet: trunk checkpoint not found at {trunk_pretrained}"
                )
            if not branch_pretrained.exists():
                raise FileNotFoundError(
                    f"Cannot compose DeepONet: branch checkpoint not found at {branch_pretrained}"
                )

            composed_ckpt_path = models_dir / f"deeponet_{problem_type}.pth"
            os.makedirs(models_dir, exist_ok=True)
            _compose_dual_deeponet_checkpoint(
                trunk_ckpt_path=trunk_pretrained,
                branch_ckpt_path=branch_pretrained,
                deeponet_ckpt_path=composed_ckpt_path,
                device=device,
                n_sensors=cfg.sensors.n_sensors,
            )

        # ====================================================================
        # 6. DEEPONET VALIDATION
        # ====================================================================
        if run_validation:
            print("\nStep 6: DeepONet Validation")
            print("-" * 70)

            _val_ckpt = models_dir / f"deeponet_{problem_type}.pth"
            validation_models_dir = str(models_dir) if _val_ckpt.exists() else str(pretrained_models_dir)

            plot_deeponet_validation(
                output_dir=str(output_dir),
                u_fom=u_fom,
                v_fom=v_fom,
                f_fom=f_data,
                svd_data=svd_data_branch,
                models_dir=validation_models_dir,
                device=device,
                problem_type=problem_type,
                sample_idx=2
            )

            plot_deeponet_rollout_validation(
                output_dir=str(output_dir),
                u_fom=u_fom,
                v_fom=v_fom,
                f_fom=f_data,
                svd_data=svd_data_branch,
                models_dir=validation_models_dir,
                rollout_dt=cfg.validation.rollout_dt,
                device=device,
                problem_type=problem_type,
                sample_idx=2,
            )

        print("\n" + "=" * 70)
        print("✓ PIPELINE COMPLETE")
        print("=" * 70)
        print(f"Output directory: {output_dir}")
        print("=" * 70 + "\n")
    finally:
        if use_tee_logger and tee is not None:
            sys.stdout = original_stdout
            tee.close()


if __name__ == "__main__":
    main()
