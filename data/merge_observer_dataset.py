"""
merge_observer_dataset.py
-------------------------
Merge the per-sample files sim_0001.mat ... sim_5000.mat (observer_wave/data)
into one HDF5 (MATLAB v7.3 layout) file: wave_gnn/data/contact_dataset.mat.

The layout is the one MATLAB's matfile() would produce for the arrays written by
contact_dataset.m (sample axis last in MATLAB, first in HDF5), so downstream
code can keep using h5py + `.T`:

    key         HDF5 shape        after .T (MATLAB order)   dtype
    U_hist      (N, Nt, Ny, Nx)   (Nx, Ny, Nt, N)          float32
    V_hist      (N, Nt, Ny, Nx)   (Nx, Ny, Nt, N)          float32
    Force       (N, 1, Nt)        (Nt, 1, N)               float32
    Contact     (N, 1, Nt)        (Nt, 1, N)               uint8 (logical)
    ToolY_hist  (N, 1, Nt)        (Nt, 1, N)               float32
    CenterX     (N, 1)            (1, N)                   float64
    CenterZ     (N, 1)            (1, N)                   float64
    Time        (1, Nt)           (Nt, 1)                  float64
    sim_id      (N,)                                       int64 (original sim number)

Samples are streamed one at a time; nothing large is held in RAM.

Usage:
    /opt/conda/bin/python merge_observer_dataset.py
    /opt/conda/bin/python merge_observer_dataset.py --limit 20 --out /tmp/test.mat
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

SIM_RE = re.compile(r"sim_(\d+)\.mat$")
KEYS = ("U_hist", "V_hist", "Force", "Contact", "CenterX", "CenterZ", "ToolY_hist", "Time")


def load_sample(path: Path) -> dict[str, np.ndarray]:
    d = loadmat(path, variable_names=KEYS)
    missing = [k for k in KEYS if k not in d]
    if missing:
        raise KeyError(f"{path.name}: missing variables {missing}")
    return d


def main() -> None:
    base = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=base.parent.parent / "observer_wave" / "data")
    ap.add_argument("--out", type=Path, default=base / "contact_dataset.mat")
    ap.add_argument("--limit", type=int, default=None, help="only merge the first N files (testing)")
    args = ap.parse_args()

    files = sorted((p for p in args.src.glob("sim_*.mat") if SIM_RE.search(p.name)),
                   key=lambda p: int(SIM_RE.search(p.name).group(1)))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No sim_*.mat files in {args.src}")
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")

    n = len(files)
    first = load_sample(files[0])
    Nx, Ny, Nt = first["U_hist"].shape
    time_ref = first["Time"].reshape(-1)
    print(f"{n} files from {args.src} | Nx={Nx}, Ny={Ny}, Nt={Nt}")
    print(f"Estimated size: {(2 * n * Nt * Nx * Ny * 4) / 1e9:.1f} GB -> {args.out}")

    tmp = args.out.with_suffix(".mat.tmp")
    sim_ids = np.zeros(n, dtype=np.int64)
    cx = np.zeros((n, 1))
    cz = np.zeros((n, 1))

    try:
        with h5py.File(tmp, "w") as out:
            ds_U = out.create_dataset("U_hist", (n, Nt, Ny, Nx), "float32", chunks=(1, Nt, Ny, Nx))
            ds_V = out.create_dataset("V_hist", (n, Nt, Ny, Nx), "float32", chunks=(1, Nt, Ny, Nx))
            ds_F = out.create_dataset("Force", (n, 1, Nt), "float32")
            ds_C = out.create_dataset("Contact", (n, 1, Nt), "uint8")
            ds_Y = out.create_dataset("ToolY_hist", (n, 1, Nt), "float32")
            ds_C.attrs["MATLAB_class"] = np.bytes_("logical")

            for i, path in enumerate(files):
                d = first if i == 0 else load_sample(path)
                if d["U_hist"].shape != (Nx, Ny, Nt) or d["V_hist"].shape != (Nx, Ny, Nt):
                    raise ValueError(f"{path.name}: unexpected shape {d['U_hist'].shape}")
                if not np.array_equal(d["Time"].reshape(-1), time_ref):
                    raise ValueError(f"{path.name}: Time vector differs from first file")

                # (Nx, Ny, Nt) -> (Nt, Ny, Nx): full axis reversal, same as h5py `.T` convention
                ds_U[i] = d["U_hist"].astype("float32", copy=False).transpose(2, 1, 0)
                ds_V[i] = d["V_hist"].astype("float32", copy=False).transpose(2, 1, 0)
                ds_F[i, 0] = d["Force"].reshape(-1)
                ds_C[i, 0] = d["Contact"].reshape(-1).astype("uint8")
                ds_Y[i, 0] = d["ToolY_hist"].reshape(-1)
                cx[i, 0] = d["CenterX"].item()
                cz[i, 0] = d["CenterZ"].item()
                sim_ids[i] = int(SIM_RE.search(path.name).group(1))

                if (i + 1) % 100 == 0 or i + 1 == n:
                    print(f"  {i + 1}/{n}", flush=True)

            out.create_dataset("CenterX", data=cx)
            out.create_dataset("CenterZ", data=cz)
            out.create_dataset("Time", data=time_ref.reshape(1, -1))
            out.create_dataset("sim_id", data=sim_ids)
            ds_U.attrs["N_total"] = n
            ds_U.attrs["Nt"] = Nt
            ds_U.attrs["Nx"] = Nx
            ds_U.attrs["Ny"] = Ny
        tmp.rename(args.out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    print(f"Done. Saved {args.out}")


if __name__ == "__main__":
    main()
