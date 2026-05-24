# CHINO — CHanneling Instability Neural Operator

[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sashgeophysics/CHINO/blob/main/notebooks/CHINO_quickstart.ipynb)

CHINO is an attention-augmented graph neural operator for simulating
**reactive infiltration instability** (RII) in porous media. Given the
concentration field of a CO2 dissolution plume at an initial time, CHINO
predicts the field at a later time — capturing the large-scale dynamics
of convective fingering across 2000 years of geological carbon sequestration.

---

## Why CHINO?

Reactive infiltration instability governs a wide range of geophysical processes:
CO2 dissolution in saline aquifers, carbonate dissolution (karst formation),
mantle melt channeling, and reactive fluid flow in fractured reservoirs.
Numerical simulation of these systems is expensive because the fingering
instability develops at fine spatial scales requiring high-resolution grids.

CHINO learns the solution operator of the non-dimensional Darcy-Boussinesq
system from an ensemble of finite-difference simulations, enabling rapid
prediction of new realizations without running the solver.

---

## Key results

| Time | L2 error (Ra=1577) |
|---|---|
| t = 0.5 (500 years) | 0.74 |
| t = 1.0 (1000 years) | 0.43 |
| t = 2.0 (2000 years) | **0.25** |

At t=1.0, the model produces visible finger-like spatial structure.
At t=2.0, the broad swept pattern of merged fingers is correctly
captured with seed-specific left-right asymmetry.

---

## Quickstart

```python
import torch
from huggingface_hub import hf_hub_download
from chino import AttentionMeshGraphNet
from chino.dataset import build_graph

# Download weights
ckpt = hf_hub_download("hier-majumder/CHINO", "chino_best.pt",
                        local_dir="checkpoints/")

# Build model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AttentionMeshGraphNet().to(device)
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()

# Build graph (200x40 MAC grid, domain [0,10]x[0,1])
edge_index, edge_attr, xc, yc, x_norm, y_norm = build_graph(
    nx=200, ny=40, lx=10.0, ly=1.0, device=device
)

# Predict: c_in -> c_out at t_out=2.0
# node_feats: (8000, 6) — [c_in_norm, S_norm, x_norm, y_norm, t_in_norm, Ra_norm]
with torch.no_grad():
    c_pred = model(node_feats, edge_index, edge_attr,
                   t_out_norm=torch.tensor(1.0, device=device))
# c_pred: (8000,) predicted concentration, non-negative
```

---

## Installation

```bash
git clone https://github.com/sashgeophysics/CHINO.git
cd CHINO
conda env create -f environment.yml
conda activate chino
```

---

## Training from scratch

```bash
# 1. Download training data (see data/README.md)
# 2. Train
python scripts/train.py --config config.yaml

# 3. Resume from checkpoint (with 6-hour wall limit)
python scripts/resume.py --config config.yaml --max_hours 6.0

# 4. Evaluate on held-out seeds
python scripts/evaluate.py --config config.yaml
```

All hyperparameters are in `config.yaml`.

---

## Repository structure

```
CHINO/
├── config.yaml          # All hyperparameters
├── environment.yml      # Conda environment
├── chino/
│   ├── model.py         # AttentionMeshGraphNet architecture
│   ├── loss.py          # Anomaly L2 + physics + BC losses
│   └── dataset.py       # RealisationDataset and graph construction
├── scripts/
│   ├── train.py         # Train from scratch
│   ├── resume.py        # Resume with wall-time limit
│   └── evaluate.py      # Evaluation and visualization
├── notebooks/
│   └── CHINO_quickstart.ipynb
├── weights/
│   └── README.md        # Download instructions
├── data/
│   └── README.md        # Data format and download instructions
└── docs/
    └── model_card.md
```

---

## Physical background

CHINO solves the surrogate problem for the non-dimensional
Darcy-Boussinesq system:

```
u = -Ra * dp'/dx
v = -Ra * (dp'/dy + gamma * c)
laplacian(p') = -gamma * Ra * dc/dy
dc/dt + u*dc/dx + v*dc/dy = (1/Pe)*laplacian(c) - Da_S*c + Da_inj*S(x,y)
```

At Ra=1577, convective fingers develop with characteristic width ~0.025
domain units. Pressure couples the entire domain instantaneously via the
Poisson equation — the key physical non-locality that standard
message-passing graph neural networks cannot represent with a finite
number of hops.

CHINO resolves this with full N^2 multi-head self-attention (N=8000 nodes),
giving every node global access in one operation.

---

## Extending CHINO

CHINO is designed to generalize beyond CCS:

- **New parameter regimes:** adjust `ra_list`, `pe`, `da_s`, `da_inj` in
  `config.yaml` and regenerate the training ensemble
- **3D:** replace the 2D MAC grid with a 3D tetrahedral mesh; the
  `AttentionMeshGraphNet` architecture requires only changes to
  `build_graph()` in `chino/dataset.py`
- **Other RII systems:** karst dissolution, mantle melt channels, reactive
  fracture flow — the architecture is physics-agnostic; replace the source
  term and boundary conditions

---

## Citation

```bibtex
@software{hier_majumder_2025_chino,
  author    = {Hier-Majumder, Saswata},
  title     = {CHINO: CHanneling Instability Neural Operator},
  year      = {2025},
  license   = {Apache-2.0},
  url       = {https://github.com/sashgeophysics/CHINO}
}

@article{sun2020geological,
  title   = {Geological Carbon Sequestration by Reactive Infiltration Instability},
  author  = {Sun, Yizhuo and Payton, Ryan L. and Hier-Majumder, Saswata and Kingdon, Andrew},
  journal = {Frontiers in Earth Science},
  volume  = {8},
  pages   = {533588},
  year    = {2020},
  doi     = {10.3389/feart.2020.533588}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
