# Training Data

The CHINO training ensemble is hosted on Zenodo:

**https://zenodo.org/record/20368449**

DOI: [10.5281/zenodo.20368449](https://doi.org/10.5281/zenodo.20368449)

## Download

```bash
# Install zenodo_get
pip install zenodo_get

# Download the full ensemble (~500 MB)
zenodo_get 20368449 -o data/
```

## Directory structure

After download, the data directory should look like:

```
data/
└── ensemble/
    ├── Ra0500/
    │   ├── seed000_fd.npz
    │   ├── seed001_fd.npz
    │   └── ... (20 seeds)
    ├── Ra1000/
    │   └── ... (20 seeds)
    └── Ra1577/
        └── ... (20 seeds)
```

## File format

Each `.npz` file contains:

| Key | Shape | Description |
|---|---|---|
| `c_snaps` | (5, 40, 200) | Concentration snapshots at t=0, 0.5, 1.0, 1.5, 2.0 |
| `t_snaps` | (5,) | Snapshot times |
| `source`  | (40, 200) | Injection source field S(x,y) |
| `xc`      | (200,) | x cell centers |
| `yc`      | (40,) | y cell centers |

## Physical parameters

| Parameter | Value | Description |
|---|---|---|
| Domain | [0,10] x [0,1] | Non-dimensional aquifer |
| Grid | 200 x 40 | MAC staggered finite difference |
| Ra | 500, 1000, 1577 | Rayleigh numbers |
| Pe | 317 | Peclet number |
| Da_S | 1.0 | Dissolution sink |
| Da_inj | 0.005 | Injection rate |
| t_max | 2.0 | Non-dimensional time (= 2000 years physical) |

## Generating new data

To generate additional seeds, use `data/fd_solver.py`:

```bash
python data/fd_solver.py --config config.yaml --ra 1577 --n_seeds 40
```

This requires a GPU and takes approximately 4 minutes for 40 seeds
at Ra=1577 on an NVIDIA A100 or equivalent.

## Reference

Sun, Y., Payton, R.L., Hier-Majumder, S., and Kingdon, A. (2020).
Geological Carbon Sequestration by Reactive Infiltration Instability.
*Frontiers in Earth Science*, 8, 533588.
https://doi.org/10.3389/feart.2020.533588
