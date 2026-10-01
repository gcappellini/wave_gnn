"""Single Hydra entrypoint for dataset preparation, training, and reporting."""

import logging
import subprocess
import sys
from pathlib import Path

import hydra
import h5py
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.curriculum import run_curriculum
from src.ground_truth_generation import run_matlab_dataset_generation
from src.logging_utils import setup_logging
from src.plotting import plot_publication_rollout
from src.reporting import (
    generate_detailed_hyperparameter_table,
    generate_stage_metrics_table,
)
from src.svd_analysis import extract_svd_basis, visualize_svd_analysis


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    script_dir = Path(__file__).resolve().parent
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    logger = setup_logging(output_dir, "main.log")

    logger.info("WAVE GNN EXPERIMENT")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    data_dir = script_dir / "data"
    merged_mat = data_dir / "merged.mat"

    if bool(cfg.pipeline.generate_dataset):
        _generate_missing_datasets(script_dir, logger)

    if bool(cfg.pipeline.get("generate_contact_dataset", False)):
        contact_cfg = cfg.datasets.contact
        contact_output = data_dir / str(contact_cfg.dataset_dir) / str(contact_cfg.output_file)
        if contact_output.exists():
            logger.info(f"Contact dataset exists; skipping MATLAB generation: {contact_output}")
        else:
            run_matlab_dataset_generation(
                matlab_script_path=str(data_dir / "contact_dataset.m"),
                output_mat_file=str(contact_output),
                script_dir=str(data_dir),
            )

    requires_merged_dataset = any(
        bool(cfg.pipeline.get(name, False))
        for name in ("analyze_dataset", "run_curriculum", "generate_tables", "generate_rollout_plots")
    )
    if requires_merged_dataset and not merged_mat.exists():
        raise FileNotFoundError(
            f"Merged dataset not found: {merged_mat}. Set pipeline.generate_dataset=true "
            "or provide data/merged.mat."
        )

    if bool(cfg.pipeline.analyze_dataset):
        _analyze_dataset(cfg, data_dir, output_dir, logger)

    if bool(cfg.pipeline.run_curriculum):
        if "curriculum" not in cfg or "steps" not in cfg.curriculum:
            raise ValueError(
                "Curriculum steps are missing. Run with --config-name curriculum_total "
                "or curriculum_smoke_test."
            )
        run_curriculum(cfg, script_dir, output_dir)

    if bool(cfg.pipeline.generate_tables):
        if "curriculum" not in cfg or "steps" not in cfg.curriculum:
            raise ValueError("Table generation requires curriculum.steps in the selected config")
        generate_stage_metrics_table(
            cfg=cfg,
            script_dir=script_dir,
            output_dir=output_dir / "curriculum_metrics",
            split_file=cfg.pipeline.get("table_split_file"),
            max_samples=cfg.pipeline.get("table_max_samples"),
        )
        generate_detailed_hyperparameter_table(cfg, output_dir)

    if bool(cfg.pipeline.generate_rollout_plots):
        rollout_cfg = cfg.pipeline.rollout
        plot_publication_rollout(
            script_dir=script_dir,
            output_dir=output_dir,
            stage=int(rollout_cfg.stage),
            n_samples_plot=int(rollout_cfg.n_samples),
            split_file=rollout_cfg.get("split_file"),
            random_seed=int(rollout_cfg.random_seed),
            sample_ids=rollout_cfg.get("sample_ids"),
        )

    logger.info("EXPERIMENT COMPLETE")


def _generate_missing_datasets(script_dir: Path, logger: logging.Logger) -> None:
    data_dir = script_dir / "data"
    source_datasets = {
        "constant_force_dataset.m": data_dir / "constant_force.mat",
        "mixed_force_ic_dataset.m": data_dir / "forced_sine_dataset.mat",
        "free_evolution_dataset.m": data_dir / "free_evolution.mat",
    }

    for script_name, output_path in source_datasets.items():
        if output_path.exists():
            logger.info(f"Dataset exists; skipping MATLAB generation: {output_path}")
            continue

        matlab_script = data_dir / script_name
        if not matlab_script.exists():
            raise FileNotFoundError(f"MATLAB dataset script not found: {matlab_script}")

        run_matlab_dataset_generation(
            matlab_script_path=str(matlab_script),
            output_mat_file=str(output_path),
            script_dir=str(data_dir),
        )

    merged_path = data_dir / "merged.mat"
    if not merged_path.exists():
        logger.info("Merging generated source datasets")
        result = subprocess.run(
            [sys.executable, str(data_dir / "merge_dataset.py")],
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
            raise RuntimeError("Dataset merge failed")
        if not merged_path.exists():
            raise FileNotFoundError(f"Dataset merge did not create {merged_path}")


def _analyze_dataset(cfg: DictConfig, data_dir: Path, output_dir: Path, logger: logging.Logger) -> None:
    merged_path = data_dir / "merged.mat"
    svd_path = data_dir / "svd_merged.npy"

    with h5py.File(merged_path, "r") as data:
        u_fom = np.array(data["U_data"]).T
        v_fom = np.array(data["V_data"]).T if "V_data" in data else None

    if svd_path.exists():
        logger.info(f"SVD analysis already exists: {svd_path}")
        if cfg.svd.visualize:
            svd_data = np.load(svd_path, allow_pickle=True).item()
            visualize_svd_analysis(
                svd_data=svd_data,
                output_dir=str(output_dir),
                n_modes=int(cfg.svd.n_modes),
                u_fom=u_fom,
                v_fom=v_fom,
            )
        return

    logger.info("Computing SVD basis")
    svd_data = extract_svd_basis(
        u_fom=u_fom,
        v_fom=v_fom,
        n_modes=int(cfg.svd.n_modes),
        visualize=False,
        output_dir=str(output_dir),
    )
    np.save(svd_path, svd_data)
    logger.info(f"Saved SVD analysis: {svd_path}")
    if cfg.svd.visualize:
        visualize_svd_analysis(
            svd_data=svd_data,
            output_dir=str(output_dir),
            n_modes=int(cfg.svd.n_modes),
            u_fom=u_fom,
            v_fom=v_fom,
        )


if __name__ == "__main__":
    main()
