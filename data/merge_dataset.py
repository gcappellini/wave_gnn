"""
merge_dataset.py
----------------
Load constant_force.mat, forced_sine_dataset.mat, and free_evolution.mat,
homogenize them (fill zero F_data / source metadata for free_evolution),
and save the merged result to merged.mat (HDF5 / MATLAB v7.3 format).

Arrays are written to disk one dataset at a time to avoid loading ~25 GB
of float64 data into memory simultaneously.

Output keys (same convention as source files — MATLAB column-major order
stored in HDF5 row-major, so downstream code using .T keeps working):
  U_data        : (N_total, Nt, Nx, Ny)
  V_data        : (N_total, Nt, Nx, Ny)
  F_data        : (N_total, Nx, Ny)
  source_centers: (2, N_total)
  source_signs  : (1, N_total)
  tlist         : (Nt, 1)  — taken from constant_force.mat
  x_grid        : (Nx, 1)  — taken from constant_force.mat
  y_grid        : (Ny, 1)  — taken from constant_force.mat
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    base = Path(__file__).resolve().parent
    out_path = base / "merged.mat"

    dataset_specs: list[tuple[str, Path]] = [
        ("constant_force",  base / "constant_force.mat"),
        ("forced_sine",     base / "forced_sine_dataset.mat"),
        ("free_evolution",  base / "free_evolution.mat"),
    ]

    # ------------------------------------------------------------------
    # Pass 1: collect shapes and total sample count
    # ------------------------------------------------------------------
    print("Pass 1: inspecting source files ...")
    total_samples = 0
    st_shape: tuple[int, ...] | None = None  # (Nt, Nx, Ny) in HDF5 order
    per_file_n: list[int] = []

    for name, path in dataset_specs:
        if not path.exists():
            raise FileNotFoundError(f"Required dataset not found: {path}")
        with h5py.File(path, "r") as fh:
            shape = fh["U_data"].shape[1:]  # (Nt, Nx, Ny)
            if st_shape is not None and shape != st_shape:
                raise ValueError(
                    f"Spatial/temporal shape mismatch: expected {st_shape}, "
                    f"got {shape} in {path}"
                )
            st_shape = shape
            n = fh["U_data"].shape[0]
            per_file_n.append(n)
            total_samples += n
            print(f"  {name}: {n} samples, U_data shape {fh['U_data'].shape}")

    assert st_shape is not None
    Nt, Nx, Ny = st_shape
    print(f"\nTotal samples: {total_samples}  |  Nt={Nt}, Nx={Nx}, Ny={Ny}")
    u_bytes = total_samples * Nt * Nx * Ny * 8
    f_bytes = total_samples * Nx * Ny * 8
    print(f"Estimated output size: {(2 * u_bytes + f_bytes) / 1e9:.1f} GB\n")

    # ------------------------------------------------------------------
    # Pass 2: create output file and stream data in
    # ------------------------------------------------------------------
    print(f"Writing to {out_path} ...")

    with h5py.File(out_path, "w") as out:
        ds_U = out.create_dataset(
            "U_data", shape=(total_samples, Nt, Nx, Ny), dtype="float64",
            chunks=(1, Nt, Nx, Ny),
        )
        ds_V = out.create_dataset(
            "V_data", shape=(total_samples, Nt, Nx, Ny), dtype="float64",
            chunks=(1, Nt, Nx, Ny),
        )
        ds_F = out.create_dataset(
            "F_data", shape=(total_samples, Nx, Ny), dtype="float64",
            chunks=(1, Nx, Ny),
        )
        ds_sc = out.create_dataset(
            "source_centers", shape=(2, total_samples), dtype="float64",
        )
        ds_ss = out.create_dataset(
            "source_signs", shape=(1, total_samples), dtype="float64",
        )

        offset = 0
        for (name, path), n in zip(dataset_specs, per_file_n):
            sl = slice(offset, offset + n)
            print(f"  [{offset}:{offset + n}]  {name} ...", flush=True)

            with h5py.File(path, "r") as fh:
                ds_U[sl] = fh["U_data"][:]
                ds_V[sl] = fh["V_data"][:]

                if "F_data" in fh:
                    ds_F[sl] = fh["F_data"][:]
                else:
                    # free_evolution has no forcing — fill with zeros
                    ds_F[sl] = np.zeros((n, Nx, Ny), dtype="float64")

                if "source_centers" in fh:
                    ds_sc[:, sl] = fh["source_centers"][:]
                else:
                    ds_sc[:, sl] = np.zeros((2, n), dtype="float64")

                if "source_signs" in fh:
                    ds_ss[:, sl] = fh["source_signs"][:]
                else:
                    ds_ss[:, sl] = np.zeros((1, n), dtype="float64")

                # Copy grid metadata from the first file only
                if offset == 0:
                    for key in ("tlist", "x_grid", "y_grid"):
                        if key in fh:
                            out.create_dataset(key, data=fh[key][:])

            offset += n

        # Store metadata as attributes on U_data
        ds_U.attrs["N_total"] = total_samples
        ds_U.attrs["Nt"] = Nt
        ds_U.attrs["Nx"] = Nx
        ds_U.attrs["Ny"] = Ny

    print(f"\nDone. Saved merged dataset to {out_path}")
    print(f"  U_data / V_data : ({total_samples}, {Nt}, {Nx}, {Ny})")
    print(f"  F_data          : ({total_samples}, {Nx}, {Ny})")
    print(f"  source_centers  : (2, {total_samples})")
    print(f"  source_signs    : (1, {total_samples})")


if __name__ == "__main__":
    main()