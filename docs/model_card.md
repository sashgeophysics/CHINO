# Model Card — CHINO v1.0

## Model Description

**CHINO** (CHanneling Instability Neural Operator) is an attention-augmented graph
neural operator for predicting reactive infiltration instability in porous
media. It maps an initial CO2 concentration field to a predicted field at a
later time, capturing the large-scale dynamics of convective fingering in
deep saline aquifers.

| Field | Value |
|---|---|
| Model name | CHINO v1.0 |
| Architecture | Attention-augmented MeshGraphNet |
| License | Apache 2.0 |
| Framework | PyTorch |
| Parameters | 7,409,537 |

---

## Intended Use

CHINO is designed for researchers studying:

- Geological carbon capture and storage (CCS)
- Reactive infiltration instability (RII) in porous media
- Convective dissolution of CO2 in saline aquifers
- Neural operator methods for chaotic PDE systems

The model predicts individual realizations of the concentration field,
not ensemble statistics. It correctly captures large-scale spatial
structure (depth of CO2 penetration, broad convective pattern) but does
not reproduce exact finger positions, which are irreducibly stochastic at
high Rayleigh numbers.

---

## Physical Regime

| Parameter | Value | Physical interpretation |
|---|---|---|
| Rayleigh number Ra | 500, 1000, 1577 | Buoyancy-driven fingering intensity |
| Peclet number Pe | 317 | Advection vs. diffusion ratio |
| Da_S | 1.0 | Dissolution sink rate |
| Da_inj | 0.005 | CO2 injection rate |
| Domain | [0,10] x [0,1] | Non-dim. aquifer (500m x 2000m physical) |
| Grid | 200 x 40 | MAC staggered finite difference |
| t_max | 2.0 | 2000 years of geological sequestration |

---

## Architecture

Each of the 6 processor blocks contains three sequential operations:

1. **Local edge update** (k=8 Moore + stride-4 edges, O(E)): captures
   fine-scale concentration gradients at finger boundaries.
2. **Global self-attention** (4 heads x 64 dim, O(N^2)): every node
   attends to every other node simultaneously, representing the
   instantaneous pressure coupling of the Darcy-Boussinesq system.
3. **Node update MLP**: fuses local messages, global attention, and
   sinusoidal time embedding.

The attention matrix (N=8000 nodes, 4 heads) requires 1 GB of VRAM per
forward pass — negligible on modern GPUs.

**Node input features (6):** c_in, S, x, y, t_in, Ra

**Output:** c(x,y,t_out) >= 0 (Softplus activation)

---

## Training

| Detail | Value |
|---|---|
| Total epochs | 550 |
| Phase 1 (ep. 1-300) | Standard curriculum, w_phys=0.05 |
| Phase 2 (ep. 301-550) | Resumed, w_phys=0.3 |
| Loss | Anomaly relative L2 (0.7) + full L2 (0.3) + physics + BC |
| Optimizer | AdamW, lr=5e-4, cosine schedule |
| Hardware | NVIDIA RTX PRO 6000 Blackwell (95 GB) |
| Wall time | ~13 hours total |

---

## Performance

| Metric | Value | Seed |
|---|---|---|
| L2 at t=0.5 | 0.74 | Seeds 18-19, Ra=1577 |
| L2 at t=1.0 | 0.43 | Seeds 18-19, Ra=1577 |
| L2 at t=2.0 | 0.25 | Seeds 18-19, Ra=1577 |

At t=1.0, the model produces visible finger-like spatial structure that
corresponds spatially to the dominant fingers in the finite-difference
reference. At t=2.0, the broad swept pattern of merged fingers is
captured with correct left-right asymmetry.

---

## Limitations

- **Early-time prediction (t=0.5):** L2=0.74. Finger nucleation at short
  times is controlled by sub-grid perturbations and is not predictable
  deterministically. A diffusion-model-based probabilistic operator
  would be the appropriate extension.
- **Individual finger positions:** Not reproduced. The chaotic nature of
  the Rayleigh-Taylor instability at Ra=1577 means finger positions are
  sensitive to initial conditions at scales below the grid resolution.
  This is a fundamental physical property, not a training failure.
- **Out-of-distribution Ra:** The model has not been tested above Ra=1577
  or below Ra=500.
- **2D only:** The current version is trained on 2D simulations. A 3D
  extension is in development.

---

## Citation

If you use CHINO, please cite:

```bibtex
@software{hier_majumder_2025_chino,
  author    = {Hier-Majumder, Saswata},
  title     = {CHINO: CHanneling Instability Neural Operator},
  year      = {2025},
  license   = {Apache-2.0},
  url       = {https://github.com/sashgeophysics/CHINO}
}

@article{sun2020geological,
  title     = {Geological Carbon Sequestration by Reactive Infiltration Instability},
  author    = {Sun, Yizhuo and Payton, Ryan L. and Hier-Majumder, Saswata and Kingdon, Andrew},
  journal   = {Frontiers in Earth Science},
  volume    = {8},
  pages     = {533588},
  year      = {2020},
  doi       = {10.3389/feart.2020.533588}
}
```
