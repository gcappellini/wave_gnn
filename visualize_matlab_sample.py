"""Visualize sample 2 from free_evolution_branch.mat as a GIF over all time instants.

This script loads displacement (U_data) and velocity (V_data), then creates an
animation where each frame is one time instant. Each frame shows u(x, y) and
v(x, y) side-by-side.
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def main() -> None:
	data_path = Path("data/free_evolution_branch.mat")
	output_path = Path(f"outputs/sample{sample_idx}_free_evolution_branch.gif")
	output_path.parent.mkdir(parents=True, exist_ok=True)

	if not data_path.exists():
		raise FileNotFoundError(f"Missing file: {data_path}")

	with h5py.File(data_path, "r") as f:
		u_fom = np.array(f["U_data"]).T  # (Nx, Ny, Nt, N_samples)
		v_fom = np.array(f["V_data"]).T  # (Nx, Ny, Nt, N_samples)
		x_grid = np.array(f["x_grid"]).reshape(-1)
		y_grid = np.array(f["y_grid"]).reshape(-1)
		tlist = np.array(f["tlist"]).reshape(-1)

	nx, ny, nt, n_samples = u_fom.shape
	sample_idx = 0
	if sample_idx >= n_samples:
		raise IndexError(f"sample_idx={sample_idx} out of range for N_samples={n_samples}")

	u_sample = u_fom[:, :, :, sample_idx]
	v_sample = v_fom[:, :, :, sample_idx]

	u_vmin, u_vmax = float(u_sample.min()), float(u_sample.max())
	v_vmin, v_vmax = float(v_sample.min()), float(v_sample.max())

	x_min = float(x_grid.min()) if x_grid.size == nx else 0.0
	x_max = float(x_grid.max()) if x_grid.size == nx else 1.0
	y_min = float(y_grid.min()) if y_grid.size == ny else 0.0
	y_max = float(y_grid.max()) if y_grid.size == ny else 1.0
	extent = [x_min, x_max, y_min, y_max]

	fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

	im_u = axes[0].imshow(
		u_sample[:, :, 0],
		origin="lower",
		extent=extent,
		aspect="auto",
		cmap="seismic",
		vmin=u_vmin,
		vmax=u_vmax,
	)
	axes[0].set_title("u(x, y)")
	axes[0].set_xlabel("x")
	axes[0].set_ylabel("y")

	im_v = axes[1].imshow(
		v_sample[:, :, 0],
		origin="lower",
		extent=extent,
		aspect="auto",
		cmap="viridis",
		vmin=v_vmin,
		vmax=v_vmax,
	)
	axes[1].set_title("v(x, y)")
	axes[1].set_xlabel("x")
	axes[1].set_ylabel("y")

	fig.colorbar(im_u, ax=axes[0], fraction=0.046, pad=0.04, label="u")
	fig.colorbar(im_v, ax=axes[1], fraction=0.046, pad=0.04, label="v")

	suptitle = fig.suptitle("")

	def update(frame_idx: int):
		im_u.set_data(u_sample[:, :, frame_idx])
		im_v.set_data(v_sample[:, :, frame_idx])
		t_val = float(tlist[frame_idx]) if tlist.size == nt else float(frame_idx)
		suptitle.set_text(f"free_evolution_branch sample={sample_idx}, frame={frame_idx}, t={t_val:.6f}")
		return im_u, im_v, suptitle

	animation = FuncAnimation(fig, update, frames=nt, interval=500, blit=False)
	animation.save(output_path, writer=PillowWriter(fps=2))
	plt.close(fig)

	print(f"Saved GIF: {output_path}")
	print(f"Shape: U={u_fom.shape}, V={v_fom.shape}")
	print(f"Time steps (Nt): {nt}")


if __name__ == "__main__":
	main()
