"""
Evaluate a trained CHINO model on held-out seeds.

Usage:
    python scripts/evaluate.py --config config.yaml

Outputs:
    - L2 error table printed to console
    - FD / CHINO / |error| field plots saved to paths.fig_dir
    - Loss curve saved to paths.fig_dir
"""

import argparse
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.ndimage import zoom

from chino.dataset import RealisationDataset, build_graph
from chino.model import AttentionMeshGraphNet


# ---------------------------------------------------------------------------
# Colormap
# ---------------------------------------------------------------------------

CMAP_C_COLORS = [
    (0.97, 0.97, 0.90),
    (0.85, 0.93, 0.75),
    (0.55, 0.85, 0.65),
    (0.20, 0.70, 0.60),
    (0.10, 0.50, 0.60),
    (0.08, 0.30, 0.55),
    (0.10, 0.10, 0.40),
    (0.15, 0.02, 0.25),
]
CMAP_C = mcolors.LinearSegmentedColormap.from_list("ccs", CMAP_C_COLORS, N=256)
BG = "#0a0a0f"
FG = "#72D169"
ERR_LABEL_C = "#3A1F04"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def predict_field(model, node_feats_np, edge_index, edge_attr,
                  t_out_n, device, dtype, c_mean, c_std, ny, nx):
    nf = torch.tensor(node_feats_np, dtype=dtype, device=device)
    t = torch.tensor(t_out_n, dtype=dtype, device=device)
    with torch.no_grad():
        c_pred = model(nf, edge_index, edge_attr, t)
    return (c_pred.cpu().numpy() * c_std + c_mean).reshape(ny, nx)


def make_node_feats(c_in_norm, s_norm, x_norm, y_norm, t_in_n, ra_n, n_nodes):
    t_col = np.full(n_nodes, t_in_n, dtype=np.float32)
    ra_col = np.full(n_nodes, ra_n, dtype=np.float32)
    return np.stack([
        c_in_norm.astype(np.float32),
        s_norm, x_norm, y_norm,
        t_col, ra_col,
    ], axis=1)


def sci_fmt(val):
    if val == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(val))))
    man = val / 10 ** exp
    return f"{man:.2f}x10^{exp}"


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    os.makedirs(cfg["paths"]["fig_dir"], exist_ok=True)
    nx, ny = cfg["data"]["nx"], cfg["data"]["ny"]
    n_nodes = nx * ny

    # Graph
    edge_index, edge_attr, xc, yc, x_norm, y_norm = build_graph(
        nx, ny, cfg["data"]["lx"], cfg["data"]["ly"], device, dtype
    )
    x_norm_np = x_norm.cpu().numpy()
    y_norm_np = y_norm.cpu().numpy()

    # Load dataset for normalization constants
    dataset = RealisationDataset(
        data_dir=cfg["paths"]["data_dir"],
        ra_list=cfg["data"]["ra_list"],
        delta_t_max=cfg["data"]["delta_t_max"],
    )
    c_mean = dataset.c_mean
    c_std = dataset.c_std
    t_scale = dataset.t_scale
    ensemble = dataset.ensemble

    # Model
    model = AttentionMeshGraphNet(
        node_in_dim=cfg["model"]["node_in_dim"],
        edge_in_dim=cfg["model"]["edge_in_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        n_blocks=cfg["model"]["n_blocks"],
        n_heads=cfg["model"]["n_heads"],
        time_embed_dim=cfg["model"]["time_embed_dim"],
    ).to(device)
    ckpt = os.path.join(cfg["paths"]["model_dir"], "chino_best.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f"Loaded: {ckpt}")

    # Evaluate held-out seeds
    eval_ra = cfg["data"]["ra_list"][-1]  # highest Ra
    held_out = cfg["data"]["held_out_seeds"]
    snap_indices = [1, 2, 4]  # t=0.5, 1.0, 2.0

    d = ensemble[eval_ra]
    t_snaps = d["t_snaps"]
    source_np = d["source"].flatten()
    s_norm_np = (source_np - source_np.mean()) / (source_np.std() + 1e-8)

    print("\n" + "=" * 65)
    print(f"Evaluation — CHINO  |  Ra={eval_ra}  |  held-out seeds")
    print("=" * 65)
    print(f"  {'Seed':<6} {'t=0.5':>8} {'t=1.0':>8} {'t=2.0':>8}")

    for seed_idx in held_out:
        c_seed = d["seeds"][seed_idx]
        c_ic = c_seed[0].flatten()
        c_in_n = (c_ic - c_mean) / c_std
        t_in_n = t_snaps[0] / t_scale
        ra_n = eval_ra / 1577.0

        nf = make_node_feats(c_in_n, s_norm_np, x_norm_np, y_norm_np,
                             t_in_n, ra_n, n_nodes)
        l2s = []
        for si in snap_indices:
            t_out_n = t_snaps[si] / t_scale
            c_pred = predict_field(
                model, nf, edge_index, edge_attr,
                t_out_n, device, dtype, c_mean, c_std, ny, nx,
            )
            c_true = c_seed[si]
            l2 = (np.linalg.norm(c_pred - c_true) /
                  (np.linalg.norm(c_true) + 1e-10))
            l2s.append(l2)

        print(f"  {seed_idx:<6} {l2s[0]:>8.4f} {l2s[1]:>8.4f} {l2s[2]:>8.4f}")

        # Field plots
        _plot_fields(
            model, nf, c_seed, t_snaps, snap_indices,
            edge_index, edge_attr, device, dtype,
            c_mean, c_std, t_scale, ny, nx, xc, yc,
            eval_ra, seed_idx, cfg["paths"]["fig_dir"],
        )

    # Loss curve
    loss_path = os.path.join(cfg["paths"]["model_dir"], "chino_losses.npz")
    if os.path.exists(loss_path):
        _plot_losses(loss_path, cfg["paths"]["fig_dir"])


def _plot_fields(model, nf, c_seed, t_snaps, snap_indices,
                 edge_index, edge_attr, device, dtype,
                 c_mean, c_std, t_scale, ny, nx, xc, yc,
                 eval_ra, seed_idx, fig_dir):
    upsample = 6
    xc_up = np.linspace(float(xc[0]), float(xc[-1]), nx * upsample)
    yc_up = np.linspace(float(yc[0]), float(yc[-1]), ny * upsample)

    n_rows = len(snap_indices) * 3
    fig, axes = plt.subplots(n_rows, 1,
                             figsize=(13.0, 1.4 * n_rows + 0.5),
                             facecolor=BG)
    fig.suptitle(
        f"Ra={eval_ra}  seed {seed_idx}  —  FD / CHINO / |error|",
        color=FG, fontsize=14, fontweight="bold", y=1.005,
    )

    row = 0
    last_im_c = last_im_err = None
    vmax_c = vmax_err = None

    for si in snap_indices:
        t_out_n = t_snaps[si] / t_scale
        c_pred = predict_field(
            model, nf, edge_index, edge_attr,
            t_out_n, device, dtype, c_mean, c_std, ny, nx,
        )
        c_true = c_seed[si]
        diff = np.abs(c_pred - c_true)
        vc = max(c_true.max(), c_pred.max()) * 1.05
        ve = diff.max() * 1.02 + 1e-10
        vmax_c = vc if vmax_c is None else max(vmax_c, vc)
        vmax_err = ve if vmax_err is None else max(vmax_err, ve)

    for si in snap_indices:
        t_val = t_snaps[si]
        t_out_n = t_val / t_scale
        c_pred = predict_field(
            model, nf, edge_index, edge_attr,
            t_out_n, device, dtype, c_mean, c_std, ny, nx,
        )
        c_true = c_seed[si]
        diff = np.abs(c_pred - c_true)
        l2 = (np.linalg.norm(c_pred - c_true) /
              (np.linalg.norm(c_true) + 1e-10))

        specs = [
            (c_true, CMAP_C,  0, vmax_c,   f"FD     t={t_val:.1f}",           False),
            (c_pred, CMAP_C,  0, vmax_c,   f"CHINO  t={t_val:.1f}  L2={l2:.4f}", False),
            (diff,   "hot",   0, vmax_err, f"|e|={sci_fmt(diff.max())}",       True),
        ]
        for field, cmap, vmin, vmax, label, is_err in specs:
            ax = axes[row]
            ax.set_facecolor(BG)
            im = ax.pcolormesh(
                xc_up, yc_up,
                zoom(field, (upsample, upsample), order=1),
                cmap=cmap, vmin=vmin, vmax=vmax,
                shading="auto", rasterized=True,
            )
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(FG)
                sp.set_linewidth(0.5)
            lc = ERR_LABEL_C if is_err else BG
            ax.text(0.99, 0.08, label, transform=ax.transAxes,
                    color=lc, fontsize=11, va="bottom", ha="right",
                    fontweight="bold")
            if not is_err:
                last_im_c = im
            else:
                last_im_err = im
            row += 1

    fig.subplots_adjust(hspace=0.04, bottom=0.10)
    cax1 = fig.add_axes([0.10, 0.055, 0.80, 0.018])
    cb1 = fig.colorbar(last_im_c, cax=cax1, orientation="horizontal")
    cb1.set_ticks([0, vmax_c])
    cb1.ax.set_xticklabels(["0", sci_fmt(vmax_c)], color=FG, fontsize=9)
    cb1.ax.xaxis.set_tick_params(color=FG)
    cb1.set_label("c", color=FG, fontsize=10)

    cax2 = fig.add_axes([0.10, 0.025, 0.80, 0.018])
    cb2 = fig.colorbar(last_im_err, cax=cax2, orientation="horizontal")
    cb2.set_ticks([0, vmax_err])
    cb2.ax.set_xticklabels(["0", sci_fmt(vmax_err)], color=FG, fontsize=9)
    cb2.ax.xaxis.set_tick_params(color=FG)
    cb2.set_label("|Dc|", color=FG, fontsize=10)

    out = os.path.join(fig_dir, f"fig_Ra{eval_ra}_seed{seed_idx:02d}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {out}")


def _plot_losses(loss_path, fig_dir):
    ld = np.load(loss_path)
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
    ax.set_facecolor(BG)
    for key, col, ls, lw in [
        ("total", FG,       "-",  2.5),
        ("l2",    "#A8E89C", "--", 2.0),
        ("phys",  "#F59E0B", "-.", 2.0),
    ]:
        if key in ld:
            ax.plot(ld["epoch"], ld[key], color=col, ls=ls, lw=lw, label=key)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch", color=FG)
    ax.set_ylabel("Loss", color=FG)
    ax.set_title("CHINO training losses", color=FG, fontsize=12)
    ax.tick_params(colors=FG)
    for sp in ax.spines.values():
        sp.set_edgecolor(FG)
    ax.legend(facecolor="#111", edgecolor=FG, labelcolor=FG, fontsize=9)
    out = os.path.join(fig_dir, "chino_losses.png")
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  Loss curve: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CHINO")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    evaluate(load_config(args.config))
