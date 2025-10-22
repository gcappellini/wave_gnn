"""
Main script for GCN-based wave equation experiments.
Uses Hydra for configuration management.

Usage:
    python main.py                           # Use default config
    python main.py model=deep_gcn            # Use deep GCN model
    python main.py training=fast             # Use fast training config
    python main.py experiment.seed=123       # Override specific parameter
"""

import os
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np

from train import train_model
from test_gcn import test_model
from dataset import create_dataset
from plot import plot_features_2d
import pickle

log = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    log.info(f"Random seed set to {seed}")


def setup_directories(cfg: DictConfig):
    """Create necessary directories for outputs and checkpoints."""
    output_dir = Path(cfg.experiment.output_dir)
    save_dir = Path(cfg.experiment.save_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    log.info(f"Output directory: {output_dir}")
    log.info(f"Checkpoint directory: {save_dir}")
    
    return output_dir, save_dir


def create_and_save_dataset(cfg: DictConfig, output_dir: Path, plot_dataset=True):
    """Create dataset and save it for reproducibility."""
    log.info("=" * 50)
    log.info("CREATING DATASET")
    log.info("=" * 50)
    
    dataset = create_dataset(num_graphs=cfg.dataset.num_graphs, cfg=cfg)
    
    log.info(f"Dataset created: {len(dataset)} graphs")
    log.info(f"  - Number of graphs: {cfg.dataset.num_graphs}")
    log.info(f"  - Timesteps per graph: {cfg.dataset.num_steps}")
    log.info(f"  - Time step (dt): {cfg.dataset.dt}")
    log.info(f"  - Batch size: {cfg.dataset.batch_size}")
    
    # Split dataset
    n_train = int(cfg.dataset.train_ratio * len(dataset))
    train_set = dataset[:n_train]
    val_set = dataset[n_train:]

    if plot_dataset:
        plot_path = output_dir / "figures/training_dataset.png"
        visual_train_set = np.array([data.x.numpy() for data in train_set]).transpose(2, 0, 1)
        plot_features_2d(dataset[0].coords, visual_train_set, output_file=str(plot_path))
        log.info(f"Training dataset visualization saved to {plot_path}")
    
    log.info(f"  - Training samples: {len(train_set)}")
    log.info(f"  - Validation samples: {len(val_set)}")
    
    return train_set, val_set


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main experiment pipeline."""
    log.info("=" * 50)
    log.info("CONFIGURATION")
    log.info("=" * 50)
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")
    
    # Set seed for reproducibility
    set_seed(cfg.experiment.seed)
    
    # Setup directories
    output_dir, save_dir = setup_directories(cfg)
    
    # Save configuration to output directory
    config_path = output_dir / "config.yaml"
    with open(config_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    log.info(f"Configuration saved to {config_path}")
    
    # Create dataset
    train_set, val_set = create_and_save_dataset(cfg, output_dir, plot_dataset=cfg.run.plot_dataset)
    
    if cfg.run.train:
        # Train model
        log.info("\n" + "=" * 50)
        log.info("TRAINING MODEL")
        log.info("=" * 50)
        
        best_model_path = save_dir / cfg.training.checkpoint_path
        model, metrics, scaler = train_model(
            cfg=cfg,
            train_set=train_set,
            val_set=val_set,
            save_path=str(best_model_path)
        )
        
        log.info(f"Training completed. Best model saved to {best_model_path}")
        log.info(f"Best validation PDE MSE: {metrics['best_val_pde']:.6e}")
        if scaler is not None:
            log.info("✓ Data scaling was enabled during training")
    else:
        best_model_path = save_dir / cfg.training.checkpoint_path
        log.info(f"Skipping training. Using existing model at {best_model_path}")
        # Load scaler if it exists
        scaler_path = save_dir / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            log.info(f"Scaler loaded from {scaler_path}")
        else:
            scaler = None
            log.warning(f"No scaler found at {scaler_path}")
        metrics = {}
    
    # Test model
    log.info("\n" + "=" * 50)
    log.info("TESTING MODEL")
    log.info("=" * 50)

    test_results, test_metrics = test_model(
        cfg=cfg,
        model_path=str(best_model_path),
        output_dir=str(output_dir),
        scaler_path=str(save_dir / "scaler.pkl") if scaler is not None else None
    )
    
    log.info("Testing completed.")
    log.info(f"Results saved to {save_dir}")
    
    # Save final summary
    summary = {
        'config': OmegaConf.to_container(cfg, resolve=True),
        'training_metrics': metrics,
        'test_results': test_results,
        'test_metrics': test_metrics
    }
    
    summary_path = output_dir / "summary.yaml"
    with open(summary_path, 'w') as f:
        f.write(OmegaConf.to_yaml(summary))
    
    log.info(f"\nExperiment summary saved to {summary_path}")
    log.info("=" * 50)
    log.info("EXPERIMENT COMPLETED SUCCESSFULLY")
    log.info("=" * 50)


if __name__ == "__main__":
    main()
