"""Create publication-quality rollout validation plots for DeepONet stage 4.

Plots are deformation-only (u), with 4 rollout instants and 3 columns:
Prediction | Ground Truth | Absolute Error.

The script randomly samples test cases from the split file and saves one figure per sample.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.plotting import (
    _build_branch_measurement_tensor,
    _load_deeponet_from_checkpoint,
    _load_grid_from_mat,
    _run_deeponet_inference,
)


def _resolve_split_path(root: Path, split_file: str | None) -> Path:
    if split_file is not None:
        p = Path(split_file).expanduser()
        if not p.is_absolute():
            p = (root / p).resolve()
        return p

    candidates = [
        root / "data" / "splits" / "merged_train80_test19_seed42.npz",
        root / "data" / "splits" / "merged_train80_test20_seed42.npz",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _snapshot_indices(nt: int, n_snapshots: int = 4) -> list[int]:
    if nt < 2:
        raise ValueError("Need Nt >= 2 for rollout snapshots")
    idx = np.linspace(1, nt - 1, n_snapshots, dtype=int)
    return sorted(set(int(i) for i in idx))


def _plot_sample(
    u_gt_snapshots: list[np.ndarray],
    u_pred_snapshots: list[np.ndarray],
    t_snapshots: list[float],
    sample_id_global: int,
    out_path: Path,
):
    n_rows = len(t_snapshots)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 4.8 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Shared color scale for prediction/ground truth across all shown instants.
    u_all = []
    for gt, pred in zip(u_gt_snapshots, u_pred_snapshots):
        u_all.extend([gt, pred])
    u_vmin = min(arr.min() for arr in u_all)
    u_vmax = max(arr.max() for arr in u_all)

    err_list = [np.abs(pred - gt) for gt, pred in zip(u_gt_snapshots, u_pred_snapshots)]
    err_vmax = max(float(err.max()) for err in err_list)
    err_vmax = max(err_vmax, 1e-12)

    title_fs = 34
    row_title_fs = 24
    tick_fs = 20

    for i, (gt, pred, err, t_val) in enumerate(zip(u_gt_snapshots, u_pred_snapshots, err_list, t_snapshots)):
        im0 = axes[i, 0].imshow(pred, cmap="seismic", origin="lower", vmin=u_vmin, vmax=u_vmax)
        axes[i, 0].set_title(f"Prediction (t={t_val:.3f})", fontsize=row_title_fs, fontweight="bold")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        im1 = axes[i, 1].imshow(gt, cmap="seismic", origin="lower", vmin=u_vmin, vmax=u_vmax)
        axes[i, 1].set_title(f"Ground Truth (t={t_val:.3f})", fontsize=row_title_fs, fontweight="bold")
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

        im2 = axes[i, 2].imshow(err, cmap="YlOrRd", origin="lower", vmin=0.0, vmax=err_vmax)
        l2_rel = np.linalg.norm(pred - gt) / (np.linalg.norm(gt) + 1e-12)
        axes[i, 2].set_title(
            f"|Error| (rel L2={l2_rel:.2e})",
            fontsize=row_title_fs,
            fontweight="bold",
        )
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])

    fig.suptitle(
        f"Sample {sample_id_global}",
        fontsize=title_fs,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout(rect=[0.0, 0.0, 0.86, 0.975])

    # Larger colorbars for publication readability.
    cbar_x = 0.87
    cbar_w = 0.04
    cbar_h = 0.38

    cax_u = fig.add_axes([cbar_x, 0.52, cbar_w, cbar_h])
    cb_u = fig.colorbar(im0, cax=cax_u)
    cb_u.set_label("Deformation u", fontsize=22, fontweight="bold")
    cb_u.ax.tick_params(labelsize=tick_fs)

    cax_e = fig.add_axes([cbar_x, 0.10, cbar_w, cbar_h])
    cb_e = fig.colorbar(im2, cax=cax_e)
    cb_e.set_label("Absolute Error", fontsize=22, fontweight="bold")
    cb_e.ax.tick_params(labelsize=tick_fs)

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(
    stage: int,
    n_samples_plot: int,
    split_file: str | None,
    random_seed: int,
    sample_ids: list[int] | None,
):
    root = Path(__file__).parent
    models_dir = root / "models"
    data_dir = root / "data"

    ckpt_path = models_dir / f"step_{stage}_deeponet_merged.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    merged_mat_path = data_dir / "merged.mat"
    if not merged_mat_path.exists():
        raise FileNotFoundError(f"Missing dataset: {merged_mat_path}")

    split_path = _resolve_split_path(root, split_file)
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with h5py.File(merged_mat_path, "r") as f:
        u_all = np.array(f["U_data"]).T
        v_all = np.array(f["V_data"]).T
        f_all = np.array(f["F_data"]).T if "F_data" in f else None

    split_npz = np.load(split_path, allow_pickle=True)
    test_key = "test_indices" if "test_indices" in split_npz else "test_idx"
    if test_key not in split_npz:
        raise ValueError(f"Split file missing test indices: {split_path}")

    test_indices = np.array(split_npz[test_key], dtype=np.int64).reshape(-1)
    if test_indices.size == 0:
        raise ValueError("No test indices available")

    if sample_ids is not None and len(sample_ids) > 0:
        requested = np.array(sample_ids, dtype=np.int64)
        n_total = u_all.shape[-1]
        out_of_range = [int(i) for i in requested.tolist() if int(i) < 0 or int(i) >= n_total]
        if out_of_range:
            raise ValueError(
                f"Requested sample ids out of dataset range [0, {n_total - 1}]: {out_of_range}"
            )
        chosen_global_ids = requested
    else:
        rng = np.random.default_rng(random_seed)
        n_pick = min(n_samples_plot, test_indices.size)
        chosen_global_ids = rng.choice(test_indices, size=n_pick, replace=False)

    u_fom = u_all[..., chosen_global_ids]
    v_fom = v_all[..., chosen_global_ids]
    if f_all is None:
        f_fom = None
    elif f_all.ndim == 3:
        f_fom = f_all[..., chosen_global_ids]
    elif f_all.ndim == 4:
        f_fom = f_all[..., chosen_global_ids]
    else:
        f_fom = f_all

    nx, ny, nt, n_local = u_fom.shape
    _, _, t_vals = _load_grid_from_mat(data_dir / "free_evolution.mat", nx, ny, nt)

    snap_ids = _snapshot_indices(nt, n_snapshots=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deeponet, n_sensors, normalization = _load_deeponet_from_checkpoint(str(ckpt_path), device, "merged")

    raw_u_min = float(normalization.get("raw_u_min", 0.0))
    raw_u_max = float(normalization.get("raw_u_max", 1.0))
    raw_v_min = float(normalization.get("raw_v_min", 0.0))
    raw_v_max = float(normalization.get("raw_v_max", 1.0))
    raw_f_min = float(normalization.get("raw_f_min", 0.0))
    raw_f_max = float(normalization.get("raw_f_max", 1.0))

    sensor_x = np.linspace(0, nx - 1, n_sensors, dtype=int)
    sensor_y = np.linspace(0, ny - 1, n_sensors, dtype=int)
    input_channels = int(getattr(getattr(deeponet, "branch_ic", None), "input_channels", 2))

    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    now = datetime.now()
    out_dir = root / "outputs" / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S") / "stage4_publication_rollout"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = [
        f"stage={stage}",
        f"checkpoint={ckpt_path}",
        f"split_file={split_path}",
        f"random_seed={random_seed}",
        f"chosen_global_ids={chosen_global_ids.tolist()}",
        f"snapshot_indices={snap_ids}",
        f"snapshot_times={[float(t_vals[i]) for i in snap_ids]}",
    ]

    for local_idx in range(n_local):
        sample_global_id = int(chosen_global_ids[local_idx])

        current_u = u_fom[:, :, 0, local_idx].astype(np.float32, copy=True)
        current_v = v_fom[:, :, 0, local_idx].astype(np.float32, copy=True)
        prev_idx = 0

        pred_by_tidx: dict[int, np.ndarray] = {}
        for t_idx in range(1, nt):
            local_dt = float(t_vals[t_idx] - t_vals[prev_idx])
            coords_local = np.stack(
                [X.flatten("F"), Y.flatten("F"), np.full((nx * ny,), local_dt, dtype=np.float32)],
                axis=1,
            )

            meas_tensor = _build_branch_measurement_tensor(
                u_field=current_u,
                v_field=current_v,
                f_fom=f_fom,
                sample_idx=local_idx,
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
                nx,
                ny,
                device,
                True,
            )

            pred_by_tidx[t_idx] = u_pred
            current_u = u_pred.astype(np.float32, copy=True)
            current_v = v_pred.astype(np.float32, copy=True)
            prev_idx = t_idx

        u_gt_snapshots = [u_fom[:, :, tidx, local_idx] for tidx in snap_ids]
        u_pred_snapshots = [pred_by_tidx[tidx] for tidx in snap_ids]
        t_snapshots = [float(t_vals[tidx]) for tidx in snap_ids]

        fig_path = out_dir / f"sample_{sample_global_id}_rollout_u_pub.pdf"
        _plot_sample(
            u_gt_snapshots=u_gt_snapshots,
            u_pred_snapshots=u_pred_snapshots,
            t_snapshots=t_snapshots,
            sample_id_global=sample_global_id,
            out_path=fig_path,
        )

    (out_dir / "selection_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Saved publication rollout plots to: {out_dir}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Publication rollout plots for stage-4 deformation.")
    parser.add_argument("--stage", type=int, default=4, help="Stage checkpoint index (default: 4)")
    parser.add_argument("--n-samples", type=int, default=4, help="Random test samples to plot (default: 4)")
    parser.add_argument("--split-file", type=str, default=None, help="Optional split .npz path")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for test sample selection")
    parser.add_argument(
        "--sample-ids",
        type=int,
        nargs="+",
        default=[5706, 14653, 14500, 1050],
        help="Exact global sample ids to plot (must belong to test split)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        stage=args.stage,
        n_samples_plot=args.n_samples,
        split_file=args.split_file,
        random_seed=args.seed,
        sample_ids=args.sample_ids,
    )
