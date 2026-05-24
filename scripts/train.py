"""
Train CHINO from scratch.

Usage:
    python scripts/train.py --config config.yaml

The script reads all hyperparameters from config.yaml and writes
checkpoints to paths.model_dir. Run scripts/resume.py to continue
training from a saved checkpoint.
"""

import argparse
import os
import shutil
import time
import types

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from chino.dataset import RealisationDataset, build_graph
from chino.loss import compute_losses
from chino.model import AttentionMeshGraphNet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def patch_no_checkpointing(model):
    """
    Replace gradient-checkpointed block forwards with direct calls.
    With 95 GB VRAM available, checkpointing is unnecessary and adds
    roughly 35% overhead to the backward pass.
    """
    def fast_forward(self, h, f, edge_index, t_emb_n):
        src, dst = edge_index[0], edge_index[1]
        f_new = self.edge_mlp(
            torch.cat([h[src], h[dst], f], dim=-1)
        ) + f
        msg = torch.zeros_like(h)
        msg.scatter_add_(0, dst.unsqueeze(-1).expand_as(f_new), f_new)
        attn_out = self.attention(h)
        h_new = self.node_mlp(
            torch.cat([h, msg, attn_out, t_emb_n], dim=-1)
        ) + h
        return self.norm(h_new), f_new

    for block in model.blocks:
        block.forward = types.MethodType(fast_forward, block)


def batched_forward(model, node_feats_batch, edge_index, edge_attr, t_out_ns):
    """Run model on a batch, returning stacked predictions (B, N)."""
    return torch.stack([
        model(node_feats_batch[b], edge_index, edge_attr, t_out_ns[b])
        for b in range(node_feats_batch.shape[0])
    ], dim=0)


def make_scheduler(optimizer, cfg, n_epochs):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs - cfg["training"]["lr_warmup_epochs"],
        eta_min=cfg["training"]["lr_min"],
    )


def save_state(history, model, epoch, model_dir, drive_dir=None, tag=None):
    """Save loss history and optionally a tagged checkpoint, then sync."""
    np.savez(
        os.path.join(model_dir, "chino_losses.npz"),
        **{k: [d[k] for d in history] for k in history[0]},
    )
    if tag:
        torch.save(
            model.state_dict(),
            os.path.join(model_dir, f"chino_ep{epoch:04d}.pt"),
        )
    if drive_dir:
        os.makedirs(drive_dir, exist_ok=True)
        for fname in os.listdir(model_dir):
            shutil.copy2(
                os.path.join(model_dir, fname),
                os.path.join(drive_dir, fname),
            )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: dict, drive_dir: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    os.makedirs(cfg["paths"]["model_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["fig_dir"], exist_ok=True)

    # Build graph (shared across all samples)
    edge_index, edge_attr, xc, yc, x_norm, y_norm = build_graph(
        cfg["data"]["nx"], cfg["data"]["ny"],
        cfg["data"]["lx"], cfg["data"]["ly"],
        device, dtype,
    )
    n_nodes = cfg["data"]["nx"] * cfg["data"]["ny"]
    dx = cfg["data"]["lx"] / cfg["data"]["nx"]
    dy = cfg["data"]["ly"] / cfg["data"]["ny"]

    # Model
    model = AttentionMeshGraphNet(
        node_in_dim=cfg["model"]["node_in_dim"],
        edge_in_dim=cfg["model"]["edge_in_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        n_blocks=cfg["model"]["n_blocks"],
        n_heads=cfg["model"]["n_heads"],
        time_embed_dim=cfg["model"]["time_embed_dim"],
    ).to(device)
    patch_no_checkpointing(model)
    print(f"CHINO  |  {model.n_params:,} parameters")

    # Dataset (initial curriculum state)
    ra_start = cfg["curriculum"]["ra_schedule"][0][1]
    dt_start = cfg["curriculum"]["dt_schedule"][0][1]
    dataset = RealisationDataset(
        data_dir=cfg["paths"]["data_dir"],
        ra_list=ra_start,
        delta_t_max=dt_start,
    )
    loader = DataLoader(
        dataset, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=0,
    )
    c_mean = dataset.c_mean
    c_std = dataset.c_std
    t_scale = dataset.t_scale
    source_np = dataset.ensemble[ra_start[-1]]["source"]

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=1e-4,
    )
    n_epochs = 300
    scheduler = make_scheduler(optimizer, cfg, n_epochs)

    history = []
    best_loss = float("inf")
    current_ra = ra_start
    current_dt = dt_start
    t_wall0 = time.time()

    print("=" * 65)
    print(f"Training CHINO")
    print(f"  batch={cfg['training']['batch_size']}"
          f"  hidden={cfg['model']['hidden_dim']}"
          f"  blocks={cfg['model']['n_blocks']}"
          f"  heads={cfg['model']['n_heads']}")
    print(f"  Ra schedule : {cfg['curriculum']['ra_schedule']}")
    print(f"  Dt schedule : {cfg['curriculum']['dt_schedule']}")
    print("=" * 65)

    for epoch in range(1, n_epochs + 1):

        # Ra curriculum
        for start_ep, ra_list in reversed(cfg["curriculum"]["ra_schedule"]):
            if epoch >= start_ep and set(ra_list) != set(current_ra):
                current_ra = ra_list
                dataset = RealisationDataset(
                    cfg["paths"]["data_dir"], current_ra, current_dt,
                    c_mean=c_mean, c_std=c_std,
                )
                loader = DataLoader(
                    dataset, batch_size=cfg["training"]["batch_size"],
                    shuffle=True, num_workers=0,
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = cfg["training"]["lr"] * 0.3
                scheduler = make_scheduler(optimizer, cfg, n_epochs - epoch)
                print(f"  [Ra] Epoch {epoch}: Ra={current_ra}  "
                      f"LR -> {cfg['training']['lr'] * 0.3:.2e}")
                break

        # Time-window curriculum
        for start_ep, dt_max in reversed(cfg["curriculum"]["dt_schedule"]):
            if epoch >= start_ep and abs(dt_max - current_dt) > 1e-6:
                current_dt = dt_max
                dataset = RealisationDataset(
                    cfg["paths"]["data_dir"], current_ra, current_dt,
                    c_mean=c_mean, c_std=c_std,
                )
                loader = DataLoader(
                    dataset, batch_size=cfg["training"]["batch_size"],
                    shuffle=True, num_workers=0,
                )
                print(f"  [dt] Epoch {epoch}: dt_max={dt_max}")
                break

        # Warmup
        warmup = cfg["training"]["lr_warmup_epochs"]
        if epoch <= warmup:
            for pg in optimizer.param_groups:
                pg["lr"] = cfg["training"]["lr"] * epoch / warmup

        # Training epoch
        model.train()
        ep_losses = {k: 0.0 for k in ["total", "l2", "phys", "bc"]}
        n_batches = 0

        for batch in loader:
            node_feats, t_out_n, c_out_n, ra_n = [x.to(device) for x in batch]

            # Fill in static node features
            B = node_feats.shape[0]
            node_feats[:, :, 1] = y_norm.unsqueeze(0).expand(B, -1)  # s_norm placeholder
            node_feats[:, :, 2] = x_norm.unsqueeze(0).expand(B, -1)
            node_feats[:, :, 3] = y_norm.unsqueeze(0).expand(B, -1)

            c_preds = batched_forward(model, node_feats, edge_index, edge_attr, t_out_n)

            batch_total = torch.tensor(0.0, device=device)
            batch_l = batch_p = batch_b = 0.0

            for b in range(B):
                losses = compute_losses(
                    c_preds[b], node_feats[b, :, 0], c_out_n[b],
                    node_feats[b, 0, 4], t_out_n[b],
                    w_l2=cfg["training"]["w_l2"],
                    w_phys=cfg["training"]["w_phys"],
                    w_bc=cfg["training"]["w_bc"],
                    c_mean=c_mean, c_std=c_std, t_scale=t_scale,
                    source_np=source_np,
                    nx=cfg["data"]["nx"], ny=cfg["data"]["ny"],
                    dx=dx, dy=dy,
                    pe=cfg["physics"]["pe"],
                    da_s=cfg["physics"]["da_s"],
                    da_inj=cfg["physics"]["da_inj"],
                    device=device,
                )
                batch_total = batch_total + losses["total"]
                batch_l += losses["l2"].item()
                batch_p += losses["phys"].item()
                batch_b += losses["bc"].item()

            total_mean = batch_total / B
            optimizer.zero_grad()
            total_mean.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["training"]["grad_clip"]
            )
            optimizer.step()

            ep_losses["total"] += total_mean.item()
            ep_losses["l2"]    += batch_l / B
            ep_losses["phys"]  += batch_p / B
            ep_losses["bc"]    += batch_b / B
            n_batches += 1

        for k in ep_losses:
            ep_losses[k] /= max(n_batches, 1)
        if epoch > warmup:
            scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        history.append({"epoch": float(epoch), **ep_losses, "lr": lr_now})

        if ep_losses["total"] < best_loss:
            best_loss = ep_losses["total"]
            torch.save(
                model.state_dict(),
                os.path.join(cfg["paths"]["model_dir"], "chino_best.pt"),
            )

        if epoch % cfg["training"]["ckpt_every"] == 0:
            save_state(
                history, model, epoch,
                cfg["paths"]["model_dir"], drive_dir, tag=True,
            )

        log_every = cfg["training"]["log_every"]
        if epoch % log_every == 0 or epoch == 1:
            print(f"  ep={epoch:4d} | "
                  f"t={time.time()-t_wall0:6.1f}s | "
                  f"lr={lr_now:.2e} | "
                  f"total={ep_losses['total']:.4e} | "
                  f"l2={ep_losses['l2']:.4e} | "
                  f"phys={ep_losses['phys']:.4e}")

    torch.save(
        model.state_dict(),
        os.path.join(cfg["paths"]["model_dir"], "chino_final.pt"),
    )
    save_state(history, model, epoch, cfg["paths"]["model_dir"], drive_dir)
    print(f"\nBest loss: {best_loss:.4e}")
    print("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CHINO from scratch")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--drive_dir", default=None,
                        help="Optional path to sync checkpoints after each save")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, drive_dir=args.drive_dir)
