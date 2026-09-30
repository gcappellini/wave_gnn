"""Curriculum runner for merged-dataset DeepONet training.

Executes staged training where each stage reuses the exact pipeline logic from
src/pipeline.py and initializes from the previous stage checkpoints. Stages
whose checkpoint already exists in models/ are skipped by default.
"""

import logging
import re
import shutil
from pathlib import Path

import h5py
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict

from .pipeline import run_constant_force_pipeline

logger = logging.getLogger("wave_gnn")


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "step"


def _resolve_or_create_sample_split(script_dir: Path, cfg: DictConfig) -> Path:
    split_cfg = cfg.curriculum.get('sample_split', {})
    train_fraction = float(split_cfg.get('train_fraction', 0.8))
    seed = int(split_cfg.get('seed', 42))
    split_path_cfg = split_cfg.get('path', None)

    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"curriculum.sample_split.train_fraction must be in (0,1), got {train_fraction}")

    if split_path_cfg:
        split_path = Path(str(split_path_cfg))
        if not split_path.is_absolute():
            split_path = (script_dir / split_path).resolve()
    else:
        split_path = (
            script_dir
            / "data"
            / "splits"
            / f"merged_train{int(train_fraction*100)}_test{int((1-train_fraction)*100)}_seed{seed}.npz"
        )

    if split_path.exists():
        logger.info(f"Using existing sample split: {split_path}")
        return split_path

    merged_path = script_dir / "data" / "merged.mat"
    if not merged_path.exists():
        raise FileNotFoundError(f"Merged dataset not found: {merged_path}")

    with h5py.File(merged_path, 'r') as f:
        if 'U_data' not in f:
            raise ValueError(f"U_data not found in {merged_path}")
        # The sample axis is the largest dimension in merged datasets (typically 5000).
        n_samples = int(max(f['U_data'].shape))

    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples for train/test split, got {n_samples}")

    n_train = int(round(train_fraction * n_samples))
    n_train = min(max(n_train, 1), n_samples - 1)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    train_idx = np.sort(perm[:n_train].astype(np.int64))
    test_idx = np.sort(perm[n_train:].astype(np.int64))

    split_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        split_path,
        train_indices=train_idx,
        test_indices=test_idx,
        seed=np.int64(seed),
        train_fraction=np.float64(train_fraction),
        n_samples=np.int64(n_samples),
    )

    logger.info(f"Created sample split: {split_path}")
    logger.info(f"  Samples train/test: {len(train_idx)}/{len(test_idx)}")
    return split_path


def run_curriculum(cfg: DictConfig, script_dir: Path, root_output_dir: Path) -> Path:
    """Run the full curriculum defined in cfg.curriculum.steps.

    Steps whose published checkpoint (models/step_{idx}_deeponet_merged.pth)
    already exists are skipped unless cfg.pipeline.force_retrain is true.
    Returns the final stage's checkpoints directory.
    """

    script_dir = Path(script_dir)
    root_output_dir = Path(root_output_dir)

    if 'curriculum' not in cfg or 'steps' not in cfg.curriculum:
        raise ValueError("Missing curriculum.steps in curriculum config")

    steps = cfg.curriculum.steps
    if len(steps) == 0:
        raise ValueError("curriculum.steps cannot be empty")

    # Build a base config for stages by removing curriculum-only fields.
    base_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(base_cfg):
        if 'curriculum' in base_cfg:
            del base_cfg['curriculum']

    initial_pretrained_cfg = cfg.curriculum.get('initial_pretrained_models_dir', None)
    if initial_pretrained_cfg:
        initial_pretrained_dir = Path(str(initial_pretrained_cfg))
        if not initial_pretrained_dir.is_absolute():
            initial_pretrained_dir = (script_dir / initial_pretrained_dir).resolve()
    else:
        initial_pretrained_dir = script_dir / "models"

    run_validation_each_step = bool(cfg.curriculum.get('run_validation_each_step', False))
    run_validation_last_step = bool(cfg.curriculum.get('run_validation_last_step', True))
    save_default_config_summary = bool(cfg.curriculum.get('save_default_config_summary', False))
    force_retrain = bool(cfg.get('pipeline', {}).get('force_retrain', False))
    sample_split_path = _resolve_or_create_sample_split(script_dir, cfg)

    models_dir = script_dir / "models"
    current_pretrained_dir = initial_pretrained_dir

    logger.info("MERGED-DATASET CURRICULUM TRAINING")
    logger.info(f"Root output directory: {root_output_dir}")
    logger.info(f"Initial pretrained directory: {current_pretrained_dir}")
    logger.info(f"Sample split file: {sample_split_path}")
    logger.info(f"Total steps: {len(steps)}")

    for stage_idx, step in enumerate(steps):
        step_name = str(step.get('name', f"step_{stage_idx:02d}"))
        step_slug = _slugify(step_name)
        stage_dir = root_output_dir / f"{stage_idx:02d}_{step_slug}"
        published_ckpt = models_dir / f"step_{stage_idx}_deeponet_merged.pth"

        if published_ckpt.exists() and not force_retrain:
            logger.info(f"Curriculum step {stage_idx}: {step_name} — skipping (checkpoint exists: {published_ckpt})")
            current_pretrained_dir = models_dir
            continue

        overrides = step.get('overrides', {})
        stage_cfg = OmegaConf.merge(base_cfg, overrides)
        with open_dict(stage_cfg):
            if 'training' not in stage_cfg:
                stage_cfg.training = {}
            stage_cfg.training.sample_split_file = str(sample_split_path)

        if 'run_validation' in step:
            run_validation = bool(step.run_validation)
        else:
            run_validation = run_validation_each_step or (
                run_validation_last_step and stage_idx == len(steps) - 1
            )

        logger.info(f"Curriculum step {stage_idx}: {step_name}")
        logger.info(f"  Pretrained source: {current_pretrained_dir}")
        logger.info(f"  Stage output: {stage_dir}")
        logger.info(f"  Run validation: {run_validation}")

        run_constant_force_pipeline(
            cfg=stage_cfg,
            output_dir=stage_dir,
            pretrained_models_dir=current_pretrained_dir,
            save_default_config_summary=save_default_config_summary,
            run_validation=run_validation,
        )

        stage_ckpt_dir = stage_dir / "checkpoints"
        stage_deeponet = stage_ckpt_dir / "deeponet_merged.pth"
        if stage_deeponet.exists():
            models_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(stage_deeponet, published_ckpt)
            logger.info(f"Published stage DeepONet checkpoint: {published_ckpt}")

        current_pretrained_dir = stage_ckpt_dir

    logger.info("CURRICULUM COMPLETE")
    logger.info(f"Final checkpoints directory: {current_pretrained_dir}")
    return current_pretrained_dir
