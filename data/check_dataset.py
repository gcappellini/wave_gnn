from __future__ import annotations

from pathlib import Path

import h5py


def print_h5_keys_and_shapes(mat_path: Path, label: str) -> None:
	print(f"\n{label}: {mat_path.name}")
	with h5py.File(mat_path, "r") as handle:
		found = False

		def _visitor(name: str, obj: object) -> None:
			nonlocal found
			if isinstance(obj, h5py.Dataset):
				found = True
				print(f"- {name}: {obj.shape}")

		handle.visititems(_visitor)

		if not found:
			print("- No datasets found")


def print_forced_sine_keys_and_shapes(mat_path: Path, label: str) -> None:
	print(f"\n{label}: {mat_path.name}")

	# First try scipy for classic MAT files, then fallback to h5py for v7.3.
	try:
		from scipy.io import loadmat

		data = loadmat(mat_path)
		printed = False
		for key, value in data.items():
			if key.startswith("__"):
				continue
			shape = getattr(value, "shape", None)
			print(f"- {key}: {shape}")
			printed = True

		if printed:
			return
	except Exception:
		pass

	print("- scipy.loadmat unavailable/failed, using h5py fallback")
	print_h5_keys_and_shapes(mat_path, "forced_sine_dataset.mat (h5py)")


def main() -> None:
	base = Path(__file__).resolve().parent
	constant_force_path = base / "free_evolution.mat"
	forced_sine_path = base / "forced_sine_dataset.mat"

	if not constant_force_path.exists():
		raise FileNotFoundError(f"Missing file: {constant_force_path}")
	if not forced_sine_path.exists():
		raise FileNotFoundError(f"Missing file: {forced_sine_path}")

	print_h5_keys_and_shapes(constant_force_path, "constant_force.mat (h5py)")
	print_forced_sine_keys_and_shapes(forced_sine_path, "forced_sine_dataset.mat")


if __name__ == "__main__":
	main()