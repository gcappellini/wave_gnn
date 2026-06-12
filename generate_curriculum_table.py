"""Evaluate curriculum DeepONet checkpoints on test split and build robust summary outputs.

Outputs are written to a Hydra-style timestamped directory:
  outputs/YYYY-MM-DD/HH-MM-SS/

Artifacts:
- sum_up_table_robust.tex
- curriculum_stage_metrics_robust.csv
- curriculum_stage_metrics_robust.json
- Per-stage plots for best/worst samples (full-field and rollout)
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
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
    plot_deeponet_rollout_validation,
    plot_deeponet_validation,
)


@dataclass
class StageMetrics:
    stage: int
    step_name: str
    description: str
    checkpoint: str
    n_samples: int
    mean_u_val_rel: float
    std_u_val_rel: float
    median_u_val_rel: float
    p90_u_val_rel: float
    mean_v_val_rel: float
    std_v_val_rel: float
    median_v_val_rel: float
    p90_v_val_rel: float
    mean_u_roll_rel: float
    std_u_roll_rel: float
    median_u_roll_rel: float
    p90_u_roll_rel: float
    mean_v_roll_rel: float
    std_v_roll_rel: float
    median_v_roll_rel: float
    p90_v_roll_rel: float
    mean_u_val_abs_rmse: float
    median_u_val_abs_rmse: float
    p90_u_val_abs_rmse: float
    mean_v_val_abs_rmse: float
    median_v_val_abs_rmse: float
    p90_v_val_abs_rmse: float
    mean_u_roll_abs_rmse: float
    median_u_roll_abs_rmse: float
    p90_u_roll_abs_rmse: float
    mean_v_roll_abs_rmse: float
    median_v_roll_abs_rmse: float
    p90_v_roll_abs_rmse: float
    best_sample_local: int
    worst_sample_local: int


def _nearest_unique_indices(t_vals: np.ndarray, targets: list[float]) -> list[int]:
    idxs = [int(np.argmin(np.abs(t_vals - target))) for target in targets]
    return sorted(set(idxs))


def _summary_stats(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
    }


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


def _describe_stage(step_name: str, training_overrides: dict) -> str:
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


def _resolve_split_path(root: Path, split_file: str | None, config_split_file: str | None = None) -> Path:
    selected = split_file if split_file is not None else config_split_file
    if selected is not None:
        split_path = Path(selected).expanduser()
        if not split_path.is_absolute():
            split_path = (root / split_path).resolve()
        return split_path

    candidates = [
        root / "data" / "splits" / "merged_train80_test19_seed42.npz",
        root / "data" / "splits" / "merged_train80_test20_seed42.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


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
) -> dict:
    deeponet, n_sensors, normalization = _load_deeponet_from_checkpoint(
        str(checkpoint_path), device, problem_type
    )

    raw_u_min = float(normalization.get("raw_u_min", 0.0))
    raw_u_max = float(normalization.get("raw_u_max", 1.0))
    raw_v_min = float(normalization.get("raw_v_min", 0.0))
    raw_v_max = float(normalization.get("raw_v_max", 1.0))
    raw_f_min = float(normalization.get("raw_f_min", 0.0))
    raw_f_max = float(normalization.get("raw_f_max", 1.0))

    Nx, Ny, _, n_samples = u_fom.shape
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

    val_u_rel_per_sample = np.zeros(n_samples, dtype=np.float64)
    val_v_rel_per_sample = np.zeros(n_samples, dtype=np.float64)
    val_u_abs_per_sample = np.zeros(n_samples, dtype=np.float64)
    val_v_abs_per_sample = np.zeros(n_samples, dtype=np.float64)

    roll_u_rel_per_sample = np.zeros(n_samples, dtype=np.float64)
    roll_v_rel_per_sample = np.zeros(n_samples, dtype=np.float64)
    roll_u_abs_per_sample = np.zeros(n_samples, dtype=np.float64)
    roll_v_abs_per_sample = np.zeros(n_samples, dtype=np.float64)

    for sample_idx in range(n_samples):
        u0 = u_fom[:, :, 0, sample_idx].astype(np.float32, copy=False)
        v0 = v_fom[:, :, 0, sample_idx].astype(np.float32, copy=False)

        val_u_rel, val_v_rel, val_u_abs, val_v_abs = [], [], [], []
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

            val_u_rel.append(float(np.linalg.norm(u_pred - u_gt) / (np.linalg.norm(u_gt) + 1e-12)))
            val_v_rel.append(float(np.linalg.norm(v_pred - v_gt) / (np.linalg.norm(v_gt) + 1e-12)))
            val_u_abs.append(float(np.sqrt(np.mean((u_pred - u_gt) ** 2))))
            val_v_abs.append(float(np.sqrt(np.mean((v_pred - v_gt) ** 2))))

        val_u_rel_per_sample[sample_idx] = float(np.mean(val_u_rel))
        val_v_rel_per_sample[sample_idx] = float(np.mean(val_v_rel))
        val_u_abs_per_sample[sample_idx] = float(np.mean(val_u_abs))
        val_v_abs_per_sample[sample_idx] = float(np.mean(val_v_abs))

        current_u = u0.astype(np.float32, copy=True)
        current_v = v0.astype(np.float32, copy=True)
        prev_idx = 0
        roll_u_rel, roll_v_rel, roll_u_abs, roll_v_abs = [], [], [], []

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

            roll_u_rel.append(float(np.linalg.norm(u_pred - u_gt) / (np.linalg.norm(u_gt) + 1e-12)))
            roll_v_rel.append(float(np.linalg.norm(v_pred - v_gt) / (np.linalg.norm(v_gt) + 1e-12)))
            roll_u_abs.append(float(np.sqrt(np.mean((u_pred - u_gt) ** 2))))
            roll_v_abs.append(float(np.sqrt(np.mean((v_pred - v_gt) ** 2))))

            current_u = u_pred.astype(np.float32, copy=True)
            current_v = v_pred.astype(np.float32, copy=True)
            prev_idx = t_idx

        roll_u_rel_per_sample[sample_idx] = float(np.mean(roll_u_rel))
        roll_v_rel_per_sample[sample_idx] = float(np.mean(roll_v_rel))
        roll_u_abs_per_sample[sample_idx] = float(np.mean(roll_u_abs))
        roll_v_abs_per_sample[sample_idx] = float(np.mean(roll_v_abs))

    # Best/worst selection uses deformation rollout relative error only.
    rollout_score = roll_u_rel_per_sample
    best_sample_local = int(np.argmin(rollout_score))
    worst_sample_local = int(np.argmax(rollout_score))

    return {
        "val_u_rel": val_u_rel_per_sample,
        "val_v_rel": val_v_rel_per_sample,
        "val_u_abs": val_u_abs_per_sample,
        "val_v_abs": val_v_abs_per_sample,
        "roll_u_rel": roll_u_rel_per_sample,
        "roll_v_rel": roll_v_rel_per_sample,
        "roll_u_abs": roll_u_abs_per_sample,
        "roll_v_abs": roll_v_abs_per_sample,
        "best_sample_local": best_sample_local,
        "worst_sample_local": worst_sample_local,
    }


def _build_robust_latex(metrics_rows: list[StageMetrics]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \caption{Robust curriculum metrics on the test split. Entries show median (P90) of relative $L^2$ error across samples. Rollout uses all available ground-truth time instants beyond $t=0$.}",
        r"  \label{tab:curriculum_errors_robust}",
        r"  \begin{tabular}{clcccc}",
        r"    \toprule",
        r"    \textbf{Stage} & \textbf{Description} &",
        r"    $\boldsymbol{E_u^{\mathrm{val}}}$ & $\boldsymbol{E_v^{\mathrm{val}}}$ &",
        r"    $\boldsymbol{E_u^{\mathrm{roll}}}$ & $\boldsymbol{E_v^{\mathrm{roll}}}$ \\",
        r"    \midrule",
    ]

    for row in metrics_rows:
        lines.append(
            "    "
            + f"{row.stage} & {_escape_tex(row.description)} "
            + f"& {_format_scientific_tex(row.median_u_val_rel)} ({_format_scientific_tex(row.p90_u_val_rel)}) "
            + f"& {_format_scientific_tex(row.median_v_val_rel)} ({_format_scientific_tex(row.p90_v_val_rel)}) "
            + f"& {_format_scientific_tex(row.median_u_roll_rel)} ({_format_scientific_tex(row.p90_u_roll_rel)}) "
            + f"& {_format_scientific_tex(row.median_v_roll_rel)} ({_format_scientific_tex(row.p90_v_roll_rel)}) \\\\"
        )

    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def _copy_ckpt_for_plot(sample_dir: Path, ckpt_path: Path, problem_type: str):
    sample_dir.mkdir(parents=True, exist_ok=True)
    target_ckpt = sample_dir / f"deeponet_{problem_type}.pth"
    shutil.copy2(ckpt_path, target_ckpt)


def _make_stage_plots(
    stage_plot_dir: Path,
    ckpt_path: Path,
    problem_type: str,
    u_fom: np.ndarray,
    v_fom: np.ndarray,
    f_fom: np.ndarray | None,
    t_vals: np.ndarray,
    best_sample_local: int,
    worst_sample_local: int,
    device: torch.device,
):
    time_instants = [float(t_val) for t_val in t_vals]
    rollout_dt = float(t_vals[1] - t_vals[0]) if len(t_vals) > 1 else None

    for tag, sample_idx in [("best", best_sample_local), ("worst", worst_sample_local)]:
        sample_dir = stage_plot_dir / tag
        _copy_ckpt_for_plot(sample_dir, ckpt_path, problem_type)

        plot_deeponet_validation(
            output_dir=str(sample_dir),
            u_fom=u_fom,
            v_fom=v_fom,
            f_fom=f_fom,
            svd_data={"basis_v": np.zeros((1, 1))},
            models_dir=str(sample_dir),
            device=device,
            problem_type=problem_type,
            sample_idx=int(sample_idx),
            time_instants=time_instants,
        )

        plot_deeponet_rollout_validation(
            output_dir=str(sample_dir),
            u_fom=u_fom,
            v_fom=v_fom,
            f_fom=f_fom,
            svd_data={"basis_v": np.zeros((1, 1))},
            models_dir=str(sample_dir),
            device=device,
            problem_type=problem_type,
            sample_idx=int(sample_idx),
            rollout_dt=rollout_dt,
        )


def generate_curriculum_table(
    split_file: str | None = None,
    max_samples: int | None = None,
):
    root = Path(__file__).parent
    config_path = root / "configs" / "constant_force" / "curriculum_total.yaml"
    models_dir = root / "models"
    merged_mat_path = root / "data" / "merged.mat"

    now = datetime.now()
    output_dir = root / "outputs" / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / "curriculum_stage_metrics_robust.csv"
    output_json = output_dir / "curriculum_stage_metrics_robust.json"
    output_tex = output_dir / "sum_up_table_robust.tex"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    steps = config["curriculum"]["steps"]

    with h5py.File(merged_mat_path, "r") as f:
        u_fom_full = np.array(f["U_data"]).T
        v_fom_full = np.array(f["V_data"]).T
        f_fom_full = np.array(f["F_data"]).T if "F_data" in f else None

    total_samples = int(u_fom_full.shape[-1])

    cfg_split = (config.get("curriculum", {}).get("sample_split", {}) or {}).get("path")
    split_path = _resolve_split_path(root, split_file, cfg_split)
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    split_data = np.load(split_path, allow_pickle=True)
    test_key = "test_indices" if "test_indices" in split_data else "test_idx"
    if test_key not in split_data:
        raise ValueError(f"Invalid split file {split_path}: missing test_indices")

    test_indices = np.array(split_data[test_key], dtype=np.int64).reshape(-1)
    if test_indices.size == 0:
        raise ValueError(f"Split file {split_path} has empty test indices")
    if test_indices.min() < 0 or test_indices.max() >= total_samples:
        raise ValueError(f"Split file {split_path} has out-of-range test indices")

    u_fom = u_fom_full[..., test_indices]
    v_fom = v_fom_full[..., test_indices]
    if f_fom_full is not None:
        if f_fom_full.ndim == 3:
            f_fom = f_fom_full[..., test_indices]
        elif f_fom_full.ndim == 4:
            f_fom = f_fom_full[..., test_indices]
        else:
            f_fom = f_fom_full
    else:
        f_fom = None

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
    _, _, t_vals = _load_grid_from_mat(t_grid_mat, Nx, Ny, Nt)

    val_targets = [0.44, 1.0]
    val_indices = _nearest_unique_indices(t_vals, val_targets)
    # Rollout uses all available ground-truth instants beyond t=0.
    rollout_indices = list(range(1, Nt))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    problem_type = "merged"

    print("=" * 70)
    print("CURRICULUM STAGE EVALUATION (ROBUST TEST-SPLIT)")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
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
        baseline_eval = _evaluate_checkpoint(
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

        s_vu = _summary_stats(baseline_eval["val_u_rel"])
        s_vv = _summary_stats(baseline_eval["val_v_rel"])
        s_ru = _summary_stats(baseline_eval["roll_u_rel"])
        s_rv = _summary_stats(baseline_eval["roll_v_rel"])
        s_vu_abs = _summary_stats(baseline_eval["val_u_abs"])
        s_vv_abs = _summary_stats(baseline_eval["val_v_abs"])
        s_ru_abs = _summary_stats(baseline_eval["roll_u_abs"])
        s_rv_abs = _summary_stats(baseline_eval["roll_v_abs"])

        baseline_row = StageMetrics(
            stage=-1,
            step_name="baseline_trunk_branch_svd",
            description="Baseline trunk+branch SVD",
            checkpoint=baseline_ckpt.name,
            n_samples=n_samples,
            mean_u_val_rel=s_vu["mean"], std_u_val_rel=s_vu["std"], median_u_val_rel=s_vu["median"], p90_u_val_rel=s_vu["p90"],
            mean_v_val_rel=s_vv["mean"], std_v_val_rel=s_vv["std"], median_v_val_rel=s_vv["median"], p90_v_val_rel=s_vv["p90"],
            mean_u_roll_rel=s_ru["mean"], std_u_roll_rel=s_ru["std"], median_u_roll_rel=s_ru["median"], p90_u_roll_rel=s_ru["p90"],
            mean_v_roll_rel=s_rv["mean"], std_v_roll_rel=s_rv["std"], median_v_roll_rel=s_rv["median"], p90_v_roll_rel=s_rv["p90"],
            mean_u_val_abs_rmse=s_vu_abs["mean"], median_u_val_abs_rmse=s_vu_abs["median"], p90_u_val_abs_rmse=s_vu_abs["p90"],
            mean_v_val_abs_rmse=s_vv_abs["mean"], median_v_val_abs_rmse=s_vv_abs["median"], p90_v_val_abs_rmse=s_vv_abs["p90"],
            mean_u_roll_abs_rmse=s_ru_abs["mean"], median_u_roll_abs_rmse=s_ru_abs["median"], p90_u_roll_abs_rmse=s_ru_abs["p90"],
            mean_v_roll_abs_rmse=s_rv_abs["mean"], median_v_roll_abs_rmse=s_rv_abs["median"], p90_v_roll_abs_rmse=s_rv_abs["p90"],
            best_sample_local=int(baseline_eval["best_sample_local"]),
            worst_sample_local=int(baseline_eval["worst_sample_local"]),
        )
        metrics_rows.append(baseline_row)

        _make_stage_plots(
            stage_plot_dir=output_dir / "stage_-1_baseline",
            ckpt_path=baseline_ckpt,
            problem_type=problem_type,
            u_fom=u_fom,
            v_fom=v_fom,
            f_fom=f_fom,
            t_vals=t_vals,
            best_sample_local=baseline_row.best_sample_local,
            worst_sample_local=baseline_row.worst_sample_local,
            device=device,
        )

    for stage_idx, step in enumerate(steps):
        ckpt_path = models_dir / f"step_{stage_idx}_deeponet_{problem_type}.pth"
        if not ckpt_path.exists():
            print(f"⚠ Stage {stage_idx}: checkpoint not found, skipping: {ckpt_path}")
            continue

        training_overrides = step.get("overrides", {}).get("training", {})
        step_name = str(step.get("name", f"step_{stage_idx}"))
        description = _describe_stage(step_name, training_overrides)

        print("-" * 70)
        print(f"Stage {stage_idx}: {step_name}")
        print(f"Checkpoint: {ckpt_path.name}")

        ev = _evaluate_checkpoint(
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

        s_vu = _summary_stats(ev["val_u_rel"])
        s_vv = _summary_stats(ev["val_v_rel"])
        s_ru = _summary_stats(ev["roll_u_rel"])
        s_rv = _summary_stats(ev["roll_v_rel"])
        s_vu_abs = _summary_stats(ev["val_u_abs"])
        s_vv_abs = _summary_stats(ev["val_v_abs"])
        s_ru_abs = _summary_stats(ev["roll_u_abs"])
        s_rv_abs = _summary_stats(ev["roll_v_abs"])

        row = StageMetrics(
            stage=stage_idx,
            step_name=step_name,
            description=description,
            checkpoint=ckpt_path.name,
            n_samples=n_samples,
            mean_u_val_rel=s_vu["mean"], std_u_val_rel=s_vu["std"], median_u_val_rel=s_vu["median"], p90_u_val_rel=s_vu["p90"],
            mean_v_val_rel=s_vv["mean"], std_v_val_rel=s_vv["std"], median_v_val_rel=s_vv["median"], p90_v_val_rel=s_vv["p90"],
            mean_u_roll_rel=s_ru["mean"], std_u_roll_rel=s_ru["std"], median_u_roll_rel=s_ru["median"], p90_u_roll_rel=s_ru["p90"],
            mean_v_roll_rel=s_rv["mean"], std_v_roll_rel=s_rv["std"], median_v_roll_rel=s_rv["median"], p90_v_roll_rel=s_rv["p90"],
            mean_u_val_abs_rmse=s_vu_abs["mean"], median_u_val_abs_rmse=s_vu_abs["median"], p90_u_val_abs_rmse=s_vu_abs["p90"],
            mean_v_val_abs_rmse=s_vv_abs["mean"], median_v_val_abs_rmse=s_vv_abs["median"], p90_v_val_abs_rmse=s_vv_abs["p90"],
            mean_u_roll_abs_rmse=s_ru_abs["mean"], median_u_roll_abs_rmse=s_ru_abs["median"], p90_u_roll_abs_rmse=s_ru_abs["p90"],
            mean_v_roll_abs_rmse=s_rv_abs["mean"], median_v_roll_abs_rmse=s_rv_abs["median"], p90_v_roll_abs_rmse=s_rv_abs["p90"],
            best_sample_local=int(ev["best_sample_local"]),
            worst_sample_local=int(ev["worst_sample_local"]),
        )
        metrics_rows.append(row)

        print(
            f"  rel median(u/v) val={row.median_u_val_rel:.4e}/{row.median_v_val_rel:.4e}, "
            f"roll={row.median_u_roll_rel:.4e}/{row.median_v_roll_rel:.4e}"
        )

        stage_slug = step_name.replace(" ", "_")
        _make_stage_plots(
            stage_plot_dir=output_dir / f"stage_{stage_idx:02d}_{stage_slug}",
            ckpt_path=ckpt_path,
            problem_type=problem_type,
            u_fom=u_fom,
            v_fom=v_fom,
            f_fom=f_fom,
            t_vals=t_vals,
            best_sample_local=row.best_sample_local,
            worst_sample_local=row.worst_sample_local,
            device=device,
        )

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(metrics_rows[0]).keys()) if metrics_rows else [])
        if metrics_rows:
            writer.writeheader()
            for row in metrics_rows:
                writer.writerow(asdict(row))

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump([asdict(row) for row in metrics_rows], f, indent=2)

    latex_table = _build_robust_latex(metrics_rows)
    with open(output_tex, "w", encoding="utf-8") as f:
        f.write(latex_table)

    print("=" * 70)
    print("✓ EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Saved robust CSV: {output_csv}")
    print(f"Saved robust JSON: {output_json}")
    print(f"Saved robust LaTeX: {output_tex}")

    return latex_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate curriculum DeepONet checkpoints and generate robust summary outputs.")
    parser.add_argument(
        "--split-file",
        type=str,
        default=None,
        help="Path to split .npz file with train_indices/test_indices.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for quick checks after applying test split.",
    )
    args = parser.parse_args()
    generate_curriculum_table(
        split_file=args.split_file,
        max_samples=args.max_samples,
    )
