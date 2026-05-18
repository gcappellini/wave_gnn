"""Curriculum runner for merged-dataset DeepONet training.

Executes staged training where each stage reuses the exact pipeline logic from
run_constant_force.py and initializes from the previous stage checkpoints.
"""

import re
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from run_constant_force import run_constant_force_pipeline


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "step"


@hydra.main(version_base=None, config_path="configs/constant_force", config_name="curriculum")
def main(cfg: DictConfig):
    script_dir = Path(__file__).parent.absolute()
    hydra_cfg = HydraConfig.get()
    root_output_dir = Path(hydra_cfg.runtime.output_dir)

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

    current_pretrained_dir = initial_pretrained_dir

    print("\n" + "=" * 70)
    print("MERGED-DATASET CURRICULUM TRAINING")
    print("=" * 70)
    print(f"Root output directory: {root_output_dir}")
    print(f"Initial pretrained directory: {current_pretrained_dir}")
    print(f"Total steps: {len(steps)}")

    for stage_idx, step in enumerate(steps):
        step_name = str(step.get('name', f"step_{stage_idx:02d}"))
        step_slug = _slugify(step_name)
        stage_dir = root_output_dir / f"{stage_idx:02d}_{step_slug}"

        overrides = step.get('overrides', {})
        stage_cfg = OmegaConf.merge(base_cfg, overrides)

        if 'run_validation' in step:
            run_validation = bool(step.run_validation)
        else:
            run_validation = run_validation_each_step or (
                run_validation_last_step and stage_idx == len(steps) - 1
            )

        print("\n" + "-" * 70)
        print(f"Curriculum step {stage_idx}: {step_name}")
        print(f"Pretrained source: {current_pretrained_dir}")
        print(f"Stage output: {stage_dir}")
        print(f"Run validation: {run_validation}")
        print("-" * 70)

        run_constant_force_pipeline(
            cfg=stage_cfg,
            output_dir=stage_dir,
            pretrained_models_dir=current_pretrained_dir,
            log_file_name="run_merged.log",
            use_tee_logger=True,
            save_default_config_summary=save_default_config_summary,
            run_validation=run_validation,
        )

        current_pretrained_dir = stage_dir / "checkpoints"

    print("\n" + "=" * 70)
    print("CURRICULUM COMPLETE")
    print("=" * 70)
    print(f"Final checkpoints directory: {current_pretrained_dir}")
    print(f"Root output directory: {root_output_dir}")


if __name__ == "__main__":
    main()
