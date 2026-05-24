"""
Dataset and graph construction utilities for CHINO.

The training data consists of finite-difference solver snapshots of
individual CO2 dissolution plume realizations in the Darcy-Boussinesq
system. Each seed produces a different realization due to the chaotic
nature of convective fingering at high Rayleigh numbers.

Data directory layout:
    data_dir/
        Ra0500/seed000_fd.npz ... seed019_fd.npz
        Ra1000/...
        Ra1577/...

Each .npz file contains:
    c_snaps:  (n_snaps, ny, nx)  concentration snapshots
    t_snaps:  (n_snaps,)         snapshot times
    source:   (ny, nx)           injection source field S(x,y)
    xc:       (nx,)              x cell centers
    yc:       (ny,)              y cell centers
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
):
    """
    Build the static graph topology for a uniform nx x ny MAC grid.

    Edges:
        - k=8 Moore neighborhood (all 8 adjacent cells)
        - Stride-4 long-range edges connecting cells 4 grid spacings apart
          in x and y, providing longer-range local connectivity

    Edge features: [delta_x, delta_y, distance] (normalized by domain size)

    Args:
        nx, ny: grid dimensions
        lx, ly: domain lengths
        device: torch device
        dtype:  floating point dtype

    Returns:
        edge_index: (2, E) long tensor
        edge_attr:  (E, 3) float tensor
        xc:         (nx,) x cell centers
        yc:         (ny,) y cell centers
        x_norm:     (N,) normalized x coordinates (node feature)
        y_norm:     (N,) normalized y coordinates (node feature)
    """
    dx = lx / nx
    dy = ly / ny
    xc = np.linspace(dx / 2, lx - dx / 2, nx)
    yc = np.linspace(dy / 2, ly - dy / 2, ny)

    # Node index map: node_id = iy * nx + ix
    def nid(ix, iy):
        return iy * nx + ix

    src_list, dst_list = [], []
    dx_list, dy_list, dist_list = [], [], []

    # Moore neighborhood offsets
    moore_offsets = [
        (di, dj)
        for di in [-1, 0, 1]
        for dj in [-1, 0, 1]
        if not (di == 0 and dj == 0)
    ]
    # Stride-4 offsets (horizontal and vertical only)
    stride4_offsets = [(-4, 0), (4, 0), (0, -4), (0, 4)]

    all_offsets = moore_offsets + stride4_offsets

    for iy in range(ny):
        for ix in range(nx):
            for di, dj in all_offsets:
                jx, jy = ix + dj, iy + di
                if 0 <= jx < nx and 0 <= jy < ny:
                    src_list.append(nid(ix, iy))
                    dst_list.append(nid(jx, jy))
                    ddx = (xc[jx] - xc[ix]) / lx
                    ddy = (yc[jy] - yc[iy]) / ly
                    dist = np.sqrt(ddx ** 2 + ddy ** 2)
                    dx_list.append(ddx)
                    dy_list.append(ddy)
                    dist_list.append(dist)

    edge_index = torch.tensor(
        [src_list, dst_list], dtype=torch.long, device=device
    )
    edge_attr = torch.tensor(
        np.stack([dx_list, dy_list, dist_list], axis=1),
        dtype=dtype, device=device,
    )

    # Node coordinate features (normalized to [0, 1])
    xx, yy = np.meshgrid(xc / lx, yc / ly)
    x_norm = torch.tensor(xx.flatten(), dtype=dtype, device=device)
    y_norm = torch.tensor(yy.flatten(), dtype=dtype, device=device)

    xc_t = torch.tensor(xc, dtype=dtype, device=device)
    yc_t = torch.tensor(yc, dtype=dtype, device=device)

    return edge_index, edge_attr, xc_t, yc_t, x_norm, y_norm


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RealisationDataset(Dataset):
    """
    Dataset of individual FD realization pairs for supervised training.

    Each sample is a (t_in, t_out) pair from one seed at one Ra value.
    The model is trained to predict the concentration field at t_out given
    the field at t_in and the seed-specific initial condition.

    Args:
        data_dir:      root directory containing Ra0500/, Ra1000/, Ra1577/
        ra_list:       list of Rayleigh numbers to include
        delta_t_max:   maximum allowed time gap t_out - t_in
        c_mean, c_std: normalization constants (computed from training data
                       if not provided)
        t_scale:       time normalization scale
    """

    def __init__(
        self,
        data_dir: str,
        ra_list: list,
        delta_t_max: float = 2.0,
        c_mean: float = None,
        c_std: float = None,
        t_scale: float = 2.0,
    ):
        self.data_dir = data_dir
        self.ra_list = ra_list
        self.delta_t_max = delta_t_max
        self.t_scale = t_scale

        # Load all seed files
        self.ensemble = {}
        all_c = []
        for ra in ra_list:
            ra_dir = os.path.join(data_dir, f"Ra{ra:04d}")
            seed_files = sorted(glob.glob(os.path.join(ra_dir, "seed*.npz")))
            if not seed_files:
                raise FileNotFoundError(
                    f"No seed files found in {ra_dir}. "
                    "Check data_dir in config.yaml."
                )
            seeds = []
            for sf in seed_files:
                d = np.load(sf)
                seeds.append(d["c_snaps"])
                all_c.append(d["c_snaps"])
            t_snaps = np.load(seed_files[0])["t_snaps"]
            source = np.load(seed_files[0])["source"]
            self.ensemble[ra] = {
                "seeds":   seeds,
                "t_snaps": t_snaps,
                "source":  source,
            }

        # Normalization constants
        all_c_flat = np.concatenate([c.flatten() for c in all_c])
        self.c_mean = float(c_mean) if c_mean is not None else float(all_c_flat.mean())
        self.c_std = float(c_std) if c_std is not None else float(all_c_flat.std() + 1e-8)

        # Build sample index: (ra, seed_idx, snap_in, snap_out)
        self.samples = []
        for ra in ra_list:
            d = self.ensemble[ra]
            t = d["t_snaps"]
            n_snaps = len(t)
            n_seeds = len(d["seeds"])
            for si in range(n_seeds):
                for i in range(n_snaps):
                    for j in range(i + 1, n_snaps):
                        if t[j] - t[i] <= delta_t_max + 1e-6:
                            self.samples.append((ra, si, i, j))

        print(f"  RealisationDataset {ra_list} "
              f"Dt_max={delta_t_max}: "
              f"{sum(len(self.ensemble[ra]['seeds']) for ra in ra_list)} seeds "
              f"-> {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ra, seed_idx, snap_in, snap_out = self.samples[idx]
        d = self.ensemble[ra]
        c_seed = d["seeds"][seed_idx]
        t = d["t_snaps"]

        c_in = c_seed[snap_in].flatten()
        c_out = c_seed[snap_out].flatten()
        t_in = t[snap_in]
        t_out = t[snap_out]

        # Normalize
        c_in_n = (c_in - self.c_mean) / self.c_std
        c_out_n = (c_out - self.c_mean) / self.c_std
        t_in_n = t_in / self.t_scale
        t_out_n = t_out / self.t_scale
        ra_n = ra / 1577.0

        n_nodes = c_in_n.shape[0]
        t_col = np.full(n_nodes, t_in_n, dtype=np.float32)
        ra_col = np.full(n_nodes, ra_n, dtype=np.float32)

        node_feats = np.stack([
            c_in_n.astype(np.float32),
            np.zeros(n_nodes, dtype=np.float32),  # s_norm filled at training time
            np.zeros(n_nodes, dtype=np.float32),  # x_norm filled at training time
            np.zeros(n_nodes, dtype=np.float32),  # y_norm filled at training time
            t_col,
            ra_col,
        ], axis=1)

        return (
            torch.tensor(node_feats, dtype=torch.float32),
            torch.tensor(t_out_n,    dtype=torch.float32),
            torch.tensor(c_out_n,    dtype=torch.float32),
            torch.tensor(ra_n,       dtype=torch.float32),
        )
