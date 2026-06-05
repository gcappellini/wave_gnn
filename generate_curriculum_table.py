"""Evaluate curriculum DeepONet checkpoints on the full dataset and build summary tables.

This script replaces the old config-only curriculum table with dataset-wide metrics:
- Validation (one-step): branch input fixed to t=0 state, evaluated at target times.
- Rollout (iterative): branch input updated with model predictions each step.

Outputs:
- sum_up_table.tex
- curriculum_stage_metrics.csv
- curriculum_stage_metrics.json
"""

from __future__ import annotations

import csv
import json
import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from src.plotting import (
    _build_branch_measurement_tensor,
    _load_deeponet_from_checkpoint,
    _load_grid_from_mat,
    _run_deeponet_inference,
)


@dataclass
class StageMetrics:
    stage: int
    step_name: str
    description: str
    checkpoint: str
    avg_u_val: float
    std_u_val: float
    avg_v_val: float
    std_v_val: float
    avg_u_roll: float
    std_u_roll: float
    avg_v_roll: float
    std_v_roll: float


def _nearest_unique_indices(t_vals: np.ndarray, targets: list[float]) -> list[int]:
    idxs = [int(np.argmin(np.abs(t_vals - target))) for target in targets]
    unique_sorted = sorted(set(idxs))
    return unique_sorted


def _format_scientific_tex(value: float) -> str:
    if not np.isfinite(value):
        return "---"
    if value == 0.0:
        return "0"

    magnitude = abs(value)
    if 1e-2 <= magnitude < 1e2:
        return f"{value:.3f}"

    base, exponent = f"{value:.2e}".split("e")
    return rf"{base} \times 10^{{{int(exponent)}}}"


def _escape_tex(text: str) -> str:
    return text.replace("_", r"\_")


def _describe_stage(step_idx: int, step_name: str, training_overrides: dict) -> str:
    if training_overrides.get("trunk_n_epochs", 0) > 0 and training_overrides.get("deeponet_n_epochs", 0) == 0:
        return "Trunk/branch SVD"

    if not bool(training_overrides.get("use_pinn_loss", False)):
        noise_cfg = training_overrides.get("deeponet_branch_input_noise", {}) or {}
        if bool(noise_cfg.get("enabled", False)):
            return "Supervised + branch noise"
        return "Joint supervised DeepONet"

    rollout_cfg = training_overrides.get("deeponet_rollout", {}) or {}
    horizon = int(rollout_cfg.get("horizon", 1) or 1)
    pde_weight = float(training_overrides.get("pde_loss_weight", 0.0) or 0.0)
    wave_weight = float(training_overrides.get("pde_loss_weight_wave", 0.0) or 0.0)
    residual_terms = "R1 + R2" if wave_weight > 0.0 else "R1"
    return f"{residual_terms}, (w_PDE; H)=({pde_weight:.2f}, {horizon})"


def _evaluate_checkpoint(
    checkpoint_path: Path,
    problem_type: str,
    u_fom: np.ndarray,
    v_fom: np.ndarray,
    f_fom: np.ndarray | None,
    t_vals: np.ndarray,
    val_indices: list[int],
    rollout_indices: list[int],
    device: torch.device,
) -> dict[str, float]:
    deeponet, n_sensors, normalization = _load_deeponet_from_checkpoint(
        str(checkpoint_path), device, problem_type
    )

    raw_u_min = float(normalization.get("raw_u_min", 0.0))
    raw_u_max = float(normalization.get("raw_u_max", 1.0))
    raw_v_min = float(normalization.get("raw_v_min", 0.0))
    raw_v_max = float(normalization.get("raw_v_max", 1.0))
    raw_f_min = float(normalization.get("raw_f_min", 0.0))
    raw_f_max = float(normalization.get("raw_f_max", 1.0))

    Nx, Ny, Nt, n_samples = u_fom.shape
    sensor_x = np.linspace(0, Nx - 1, n_sensors, dtype=int)
    sensor_y = np.linspace(0, Ny - 1, n_sensors, dtype=int)
    input_channels = int(getattr(getattr(deeponet, "branch_ic", None), "input_channels", 2))

    x = np.linspace(0.0, 1.0, Nx)
    y = np.linspace(0.0, 1.0, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    coords_abs_cache = {
        idx: np.stack(
            [X.flatten("F"), Y.flatten("F"), np.full((Nx * Ny,), float(t_vals[idx]), dtype=np.float32)],
            axis=1,
        )
        for idx in val_indices
    }

    val_u_per_sample = np.zeros(n_samples, dtype=np.float64)
    val_v_per_sample = np.zeros(n_samples, dtype=np.float64)
    roll_u_per_sample = np.zeros(n_samples, dtype=np.float64)
    roll_v_per_sample = np.zeros(n_samples, dtype=np.float64)

    for sample_idx in range(n_samples):
        u0 = u_fom[:, :, 0, sample_idx].astype(np.float32, copy=False)
        v0 = v_fom[:, :, 0, sample_idx].astype(np.float32, copy=False)

        val_u_errors = []
        val_v_errors = []
        for t_idx in val_indices:
            meas_tensor = _build_branch_measurement_tensor(
                u_field=u0,
                v_field=v0,
                f_fom=f_fom,
                sample_idx=sample_idx,
                sensor_x=sensor_x,
                sensor_y=sensor_y,
                raw_u_min=raw_u_min,
                raw_u_max=raw_u_max,
                raw_v_min=raw_v_min,
                raw_v_max=raw_v_max,
                raw_f_min=raw_f_min,
                raw_f_max=raw_f_max,
                input_channels=input_channels,
                n_sensors=n_sensors,
                device=device,
            )

            u_pred, v_pred = _run_deeponet_inference(
                deeponet,
                meas_tensor,
                coords_abs_cache[t_idx],
                Nx,
                Ny,
                device,
                True,
            )
            u_gt = u_fom[:, :, t_idx, sample_idx]
            v_gt = v_fom[:, :, t_idx, sample_idx]
            val_u_errors.append(float(np.linalg.norm(u_pred - u_gt) / (np.linalg.norm(u_gt) + 1e-12)))
            val_v_errors.append(float(np.linalg.norm(v_pred - v_gt) / (np.linalg.norm(v_gt) + 1e-12)))

        val_u_per_sample[sample_idx] = float(np.mean(val_u_errors))
        val_v_per_sample[sample_idx] = float(np.mean(val_v_errors))

        current_u = u0.astype(np.float32, copy=True)
        current_v = v0.astype(np.float32, copy=True)
        prev_idx = 0
        roll_u_errors = []
        roll_v_errors = []

        for t_idx in rollout_indices:
            local_dt = float(t_vals[t_idx] - t_vals[prev_idx])
            coords_local = np.stack(
                [X.flatten("F"), Y.flatten("F"), np.full((Nx * Ny,), local_dt, dtype=np.float32)],
                axis=1,
            )

            meas_tensor = _build_branch_measurement_tensor(
                u_field=current_u,
                v_field=current_v,
                f_fom=f_fom,
                sample_idx=sample_idx,
                sensor_x=sensor_x,
                sensor_y=sensor_y,
                raw_u_min=raw_u_min,
                raw_u_max=raw_u_max,
                raw_v_min=raw_v_min,
                raw_v_max=raw_v_max,
                raw_f_min=raw_f_min,
                raw_f_max=raw_f_max,
                input_channels=input_channels,
                n_sensors=n_sensors,
                device=device,
            )

            u_pred, v_pred = _run_deeponet_inference(
                deeponet,
                meas_tensor,
                coords_local,
                Nx,
                Ny,
                device,
                True,
            )
            u_gt = u_fom[:, :, t_idx, sample_idx]
            v_gt = v_fom[:, :, t_idx, sample_idx]
            roll_u_errors.append(float(np.linalg.norm(u_pred - u_gt) / (np.linalg.norm(u_gt) + 1e-12)))
            roll_v_errors.append(float(np.linalg.norm(v_pred - v_gt) / (np.linalg.norm(v_gt) + 1e-12)))

            current_u = u_pred.astype(np.float32, copy=True)
            current_v = v_pred.astype(np.float32, copy=True)
            prev_idx = t_idx

        roll_u_per_sample[sample_idx] = float(np.mean(roll_u_errors))
        roll_v_per_sample[sample_idx] = float(np.mean(roll_v_errors))

    return {
        "avg_u_val": float(np.mean(val_u_per_sample)),
        "std_u_val": float(np.std(val_u_per_sample)),
        "avg_v_val": float(np.mean(val_v_per_sample)),
        "std_v_val": float(np.std(val_v_per_sample)),
        "avg_u_roll": float(np.mean(roll_u_per_sample)),
        "std_u_roll": float(np.std(roll_u_per_sample)),
        "avg_v_roll": float(np.mean(roll_v_per_sample)),
        "std_v_roll": float(np.std(roll_v_per_sample)),
    }


def _build_sumup_latex(metrics_rows: list[StageMetrics], val_times: list[float], rollout_times: list[float]) -> str:
    val_times_tex = ", ".join([f"{t_val:.2f}" for t_val in val_times])
    rollout_times_tex = ", ".join([f"{t_val:.2f}" for t_val in rollout_times])
    caption_line = (
        r"  \caption{Average relative $L^2$ errors across all samples for curriculum stage checkpoints. "
        + f"Validation uses $t \\in \\{{{val_times_tex}\\}}$ with one-step prediction from $t=0$; "
        + f"rollout uses iterative prediction over $t \\in \\{{{rollout_times_tex}\\}}$."
        + "}"
    )

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        caption_line,
        r"  \label{tab:curriculum_errors_full_dataset}",
        r"  \begin{tabular}{clcccc}",
        r"    \toprule",
        r"    \textbf{Stage} &",
        r"    \textbf{Description} &",
        r"    $\boldsymbol{\overline{E}_{u}^{\mathrm{val}}}$ &",
        r"    $\boldsymbol{\overline{E}_{v}^{\mathrm{val}}}$ &",
        r"    $\boldsymbol{\overline{E}_{u}^{\mathrm{roll}}}$ &",
        r"    $\boldsymbol{\overline{E}_{v}^{\mathrm{roll}}}$ \\",
        r"    \midrule",
    ]

    for row in metrics_rows:
        lines.append(
            "    "
            + f"{row.stage} & {_escape_tex(row.description)} "
            + f"& {_format_scientific_tex(row.avg_u_val)} "
            + f"& {_format_scientific_tex(row.avg_v_val)} "
            + f"& {_format_scientific_tex(row.avg_u_roll)} "
            + f"& {_format_scientific_tex(row.avg_v_roll)} \\\\"
        )

    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def generate_curriculum_table(
    max_samples: int | None = None,
    split_file: str | None = None,
):
    root = Path(__file__).parent
    config_path = root / "configs" / "constant_force" / "curriculum_total.yaml"
    models_dir = root / "models"
    merged_mat_path = root / "data" / "merged.mat"
    output_csv = root / "curriculum_stage_metrics.csv"
    output_json = root / "curriculum_stage_metrics.json"
    output_tex = root / "sum_up_table.tex"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    steps = config["curriculum"]["steps"]

    with h5py.File(merged_mat_path, "r") as f:
        u_fom = np.array(f["U_data"]).T
        v_fom = np.array(f["V_data"]).T
        f_fom = np.array(f["F_data"]).T if "F_data" in f else None

    total_samples = int(u_fom.shape[-1])
    if split_file is None:
        split_path = root / "data" / "splits" / "merged_train80_test20_seed42.npz"
    else:
        split_path = Path(split_file).expanduser()
        if not split_path.is_absolute():
            split_path = (root / split_path).resolve()

    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    split_data = np.load(split_path, allow_pickle=True)
    test_key = 'test_indices' if 'test_indices' in split_data else 'test_idx'
    if test_key not in split_data:
        raise ValueError(f"Invalid split file {split_path}: missing test_indices")
    test_indices = np.array(split_data[test_key], dtype=np.int64).reshape(-1)
    if test_indices.size == 0:
        raise ValueError(f"Split file {split_path} has empty test indices")
    if test_indices.min() < 0 or test_indices.max() >= total_samples:
        raise ValueError(f"Split file {split_path} has out-of-range test indices")

    u_fom = u_fom[..., test_indices]
    v_fom = v_fom[..., test_indices]
    if f_fom is not None:
        if f_fom.ndim == 3:
            f_fom = f_fom[..., test_indices]
        elif f_fom.ndim == 4:
            f_fom = f_fom[..., test_indices]

    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {max_samples}")
        u_fom = u_fom[..., :max_samples]
        v_fom = v_fom[..., :max_samples]
        if f_fom is not None:
            if f_fom.ndim == 3:
                f_fom = f_fom[..., :max_samples]
            elif f_fom.ndim == 4:
                f_fom = f_fom[..., :max_samples]

    Nx, Ny, Nt, n_samples = u_fom.shape
    t_grid_mat = root / "data" / "free_evolution.mat"
    x_vals, y_vals, t_vals = _load_grid_from_mat(t_grid_mat, Nx, Ny, Nt)
    _ = (x_vals, y_vals)

    # Matches the paper-style summary in the provided example, but over all samples.
    val_targets = [0.44, 1.0]
    rollout_targets = [0.11, 0.33, 0.67, 1.0]
    val_indices = _nearest_unique_indices(t_vals, val_targets)
    rollout_indices = _nearest_unique_indices(t_vals, rollout_targets)
    rollout_indices = [idx for idx in rollout_indices if idx > 0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    problem_type = "merged"

    print("=" * 70)
    print("CURRICULUM STAGE EVALUATION (TEST SPLIT)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Split file: {split_path}")
    print(f"Dataset shape on test split: U{u_fom.shape}, V{v_fom.shape}")
    print(f"Test samples selected: {u_fom.shape[-1]}/{total_samples}")
    print(f"Validation indices: {val_indices} -> {[float(t_vals[idx]) for idx in val_indices]}")
    print(f"Rollout indices: {rollout_indices} -> {[float(t_vals[idx]) for idx in rollout_indices]}")

    metrics_rows: list[StageMetrics] = []

    baseline_ckpt = models_dir / f"deeponet_{problem_type}.pth"
    if baseline_ckpt.exists():
        print("-" * 70)
        print("Baseline: trunk+branch SVD initialization")
        print(f"Checkpoint: {baseline_ckpt.name}")
        baseline_metrics = _evaluate_checkpoint(
            checkpoint_path=baseline_ckpt,
            problem_type=problem_type,
            u_fom=u_fom,
            v_fom=v_fom,
            f_fom=f_fom,
            t_vals=t_vals,
            val_indices=val_indices,
            rollout_indices=rollout_indices,
            device=device,
        )
        metrics_rows.append(
            StageMetrics(
                stage=-1,
                step_name="baseline_trunk_branch_svd",
                description="Baseline trunk+branch SVD",
                checkpoint=baseline_ckpt.name,
                avg_u_val=baseline_metrics["avg_u_val"],
                std_u_val=baseline_metrics["std_u_val"],
                avg_v_val=baseline_metrics["avg_v_val"],
                std_v_val=baseline_metrics["std_v_val"],
                avg_u_roll=baseline_metrics["avg_u_roll"],
                std_u_roll=baseline_metrics["std_u_roll"],
                avg_v_roll=baseline_metrics["avg_v_roll"],
                std_v_roll=baseline_metrics["std_v_roll"],
            )
        )

    for stage_idx, step in enumerate(steps):
        ckpt_path = models_dir / f"step_{stage_idx}_deeponet_{problem_type}.pth"
        if not ckpt_path.exists():
            print(f"⚠ Stage {stage_idx}: checkpoint not found, skipping: {ckpt_path}")
            continue

        training_overrides = step.get("overrides", {}).get("training", {})
        step_name = str(step.get("name", f"step_{stage_idx}"))
        description = _describe_stage(stage_idx, step_name, training_overrides)

        print("-" * 70)
        print(f"Stage {stage_idx}: {step_name}")
        print(f"Checkpoint: {ckpt_path.name}")

        stage_metrics = _evaluate_checkpoint(
            checkpoint_path=ckpt_path,
            problem_type=problem_type,
            u_fom=u_fom,
            v_fom=v_fom,
            f_fom=f_fom,
            t_vals=t_vals,
            val_indices=val_indices,
            rollout_indices=rollout_indices,
            device=device,
        )

        metrics_row = StageMetrics(
            stage=stage_idx,
            step_name=step_name,
            description=description,
            checkpoint=ckpt_path.name,
            avg_u_val=stage_metrics["avg_u_val"],
            std_u_val=stage_metrics["std_u_val"],
            avg_v_val=stage_metrics["avg_v_val"],
            std_v_val=stage_metrics["std_v_val"],
            avg_u_roll=stage_metrics["avg_u_roll"],
            std_u_roll=stage_metrics["std_u_roll"],
            avg_v_roll=stage_metrics["avg_v_roll"],
            std_v_roll=stage_metrics["std_v_roll"],
        )
        metrics_rows.append(metrics_row)

        print(
            f"  avg_u_val={metrics_row.avg_u_val:.4e}, avg_v_val={metrics_row.avg_v_val:.4e}, "
            f"avg_u_roll={metrics_row.avg_u_roll:.4e}, avg_v_roll={metrics_row.avg_v_roll:.4e}"
        )

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stage",
                "step_name",
                "description",
                "checkpoint",
                "avg_u_val",
                "std_u_val",
                "avg_v_val",
                "std_v_val",
                "avg_u_roll",
                "std_u_roll",
                "avg_v_roll",
                "std_v_roll",
            ],
        )
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row.__dict__)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump([row.__dict__ for row in metrics_rows], f, indent=2)

    latex_table = _build_sumup_latex(metrics_rows, val_targets, rollout_targets)
    with open(output_tex, "w", encoding="utf-8") as f:
        f.write(latex_table)

    print("=" * 70)
    print("✓ EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Saved CSV: {output_csv}")
    print(f"Saved JSON: {output_json}")
    print(f"Saved LaTeX: {output_tex}")

    return latex_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate curriculum DeepONet checkpoints and generate summary tables.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests after applying test split.",
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default=None,
        help="Path to split .npz file with train_indices/test_indices (defaults to data/splits/merged_train80_test20_seed42.npz).",
    )
    args = parser.parse_args()
    generate_curriculum_table(
        max_samples=args.max_samples,
        split_file=args.split_file,
    )
