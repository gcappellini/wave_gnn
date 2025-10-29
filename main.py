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
from plot import plot_features_2d, plot_loss_history
import pickle
from datetime import datetime

log = logging.getLogger(__name__)

def convert_to_native_types(obj):
    """
    Recursively convert NumPy types to Python native types for OmegaConf compatibility.
    
    Args:
        obj: Object to convert (can be dict, list, numpy type, etc.)
    
    Returns:
        Object with all numpy types converted to Python native types
    """
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
    
def restore_from_previous_run(run_name: str, save_dir: Path, cfg: DictConfig = None):
    """
    Restore model checkpoint and training metrics from a previous run.
    
    Args:
        run_name: Name of the previous run (e.g., "2024-10-29_14-30-45")
        save_dir: Directory where checkpoints are saved
    
    Returns:
        tuple: (checkpoint_path, metrics_dict) where:
            - checkpoint_path: Path to the model checkpoint (.pt file)
            - metrics_dict: Dictionary containing training metrics including loss_history
    
    Raises:
        FileNotFoundError: If checkpoint or CSV file not found
    """
    checkpoint_path = save_dir / f"best_model_{run_name}.pt"
    csv_path = save_dir / f"best_model_{run_name}.csv"

    # Check if files exist
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Available checkpoints in {save_dir}:\n" +
            "\n".join([f"  - {f.name}" for f in save_dir.glob("best_model_*.pt")])
        )
    
    if not csv_path.exists():
        log.warning(f"Loss history CSV not found: {csv_path}")
        log.warning("Metrics will be loaded from checkpoint only (may not include full loss_history)")
    
    # Load checkpoint to get stored metrics
    log.info(f"Restoring from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract metrics from checkpoint
    metrics = {
        'best_val_pde': checkpoint.get('best_val_pde', float('nan')),
        'final_epoch': checkpoint.get('epoch', 0),
    }
    
    # Try to load full loss history from CSV if available
    if csv_path.exists():
        # try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        loss_history = df.to_dict('records')
        metrics['loss_history'] = loss_history
        log.info(f"Loaded loss history from CSV: {len(loss_history)} epochs")
        if cfg.plot.plot_train_loss:
            plot_path = save_dir / f"loss_history_{run_name}.png"
            plot_loss_history(
                loss_history,
                output_file=str(plot_path),
                figsize=(14, 10),
                dpi=150,
                show_weights=True,
            )
            log.info(f"Loss history plot saved to {plot_path}")
        

        # except Exception as e:
        #     log.warning(f"Failed to load loss history from CSV: {e}")
        #     import csv
        #     try:
        #         with open(csv_path, 'r') as f:
        #             reader = csv.DictReader(f)
        #             loss_history = list(reader)
        #             # Convert string values to float
        #             for record in loss_history:
        #                 for key, value in record.items():
        #                     try:
        #                         record[key] = float(value)
        #                     except (ValueError, TypeError):
        #                         pass
        #             metrics['loss_history'] = loss_history
        #         log.info(f"Loaded loss history from CSV (fallback): {len(loss_history)} epochs")
        #     except Exception as e2:
        #         log.warning(f"Failed to load loss history with fallback: {e2}")
    
    log.info(f"Restored metrics: best_val_pde={metrics['best_val_pde']:.6e}, final_epoch={metrics['final_epoch']}")
    
    return checkpoint_path, metrics


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


def create_and_save_dataset(cfg: DictConfig, output_dir: Path, plot_dataset=True, run_name=None):
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
        plot_path = output_dir / f"figures/training_dataset_{run_name}.png"
        visual_train_set = np.array([data.x.numpy() for data in train_set]).transpose(2, 0, 1)
        plot_features_2d(dataset[0].coords, visual_train_set, dt=1, output_file=str(plot_path), ylabel='Sample Index', ylabel_as_int=True)
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
    
    # Generate fancy run name with timestamp and experiment info
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}"

    # Save configuration to output directory
    config_path = output_dir / f"config_{run_name}.yaml"
    with open(config_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    log.info(f"Configuration saved to {config_path}")
    

    if cfg.run.train:
        # Create dataset
        train_set, val_set = create_and_save_dataset(cfg, output_dir, plot_dataset=cfg.plot.plot_dataset, run_name=run_name)

        # Train model
        log.info("\n" + "=" * 50)
        log.info("TRAINING MODEL")
        log.info("=" * 50)
        
        best_model_path = save_dir / f"best_model_{run_name}.pt"
        model, metrics = train_model(
            cfg=cfg,
            train_set=train_set,
            val_set=val_set,
            save_path=str(best_model_path)
        )
        
        log.info(f"Training completed. Best model saved to {best_model_path}")
        log.info(f"Best validation PDE MSE: {metrics['best_val_pde']:.6e}")
    else:
        resume_from = cfg.run.resume_from
        best_model_path = save_dir / f"best_model_{resume_from}.pt"
        log.info(f"Skipping training. Using existing model at {best_model_path}")
        best_model_path, metrics = restore_from_previous_run(resume_from, save_dir, cfg=cfg)

    # Test model
    log.info("\n" + "=" * 50)
    log.info("TESTING MODEL")
    log.info("=" * 50)

    test_results, test_metrics = test_model(
        cfg=cfg,
        model_path=str(best_model_path),
        output_dir=str(output_dir),
        run_name=run_name
    )
    
    log.info("Testing completed.")
    log.info(f"Results saved to {save_dir}")

    # Convert NumPy types to Python native types for OmegaConf compatibility
    metrics_native = convert_to_native_types(metrics)
    test_results_native = convert_to_native_types(test_results)
    test_metrics_native = convert_to_native_types(test_metrics)
    
    # Save final summary
    summary = {
        'config': OmegaConf.to_container(cfg, resolve=True),
        'training_metrics': metrics,
        'test_results': test_results,
        'test_metrics': test_metrics
    }

    summary_path = output_dir / f"summary_{run_name}.yaml"
    with open(summary_path, 'w') as f:
        f.write(OmegaConf.to_yaml(summary))
    
    log.info(f"\nExperiment summary saved to {summary_path}")
    log.info("=" * 50)
    log.info("EXPERIMENT COMPLETED SUCCESSFULLY")
    log.info("=" * 50)


if __name__ == "__main__":
    main()
