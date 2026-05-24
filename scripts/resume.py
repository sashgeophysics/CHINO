"""
Resume CHINO training from a checkpoint with a wall-time limit.

Usage:
    python scripts/resume.py --config config.yaml
    python scripts/resume.py --config config.yaml --max_hours 8.0
    python scripts/resume.py --config config.yaml --checkpoint chino_ep0300.pt

The script detects the last completed epoch from chino_losses.npz and
resumes seamlessly. Loss history is saved every ckpt_every epochs so
the epoch counter is always accurate on the next reload.
"""

import argparse
import glob
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
# Helpers (shared with train.py)
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def patch_no_checkpointing(model):
    """Replace block forwards with direct calls (no gradient checkpointing)."""
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
    return torch.stack([
        model(node_feats_batch[b], edge_index, edge_attr, t_out_ns[b])
        for b in range(node_feats_batch.shape[0])
    ], dim=0)


def get_epoch_offset(model_dir: str):
    """
    Return the last completed epoch from chino_losses.npz.
    Falls back to scanning checkpoint filenames if the npz is missing.
    """
    loss_path = os.path.join(model_dir, "chino_losses.npz")
    if os.path.exists(loss_path):
        try:
            prev = np.load(loss_path)
            offset = int(prev["epoch"][-1])
            return prev, offset
        except Exception:
            pass

    # Fallback: scan for chino_epNNNN.pt
    ckpts = glob.glob(os.path.join(model_dir, "chino_ep*.pt"))
    if ckpts:
        nums = []
        for c in ckpts:
            try:
                nums.append(
                    int(os.path.basename(c)
                        .replace("chino_ep", "").replace(".pt", ""))
                )
            except ValueError:
                pass
        if nums:
            return None, max(nums)

    return None, 0


def save_state(history, model, epoch, model_dir, drive_dir=None, tag=None):
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
# Resume loop
# ---------------------------------------------------------------------------

def resume(cfg: dict, checkpoint: str, max_hours: float, drive_dir: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    max_seconds = max_hours * 3600

    os.makedirs(cfg["paths"]["model_dir"], exist_ok=True)

    # Graph
    edge_index, edge_attr, xc, yc, x_norm, y_norm = build_graph(
        cfg["data"]["nx"], cfg["data"]["ny"],
        cfg["data"]["lx"], cfg["data"]["ly"],
        device, dtype,
    )
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

    ckpt_path = os.path.join(cfg["paths"]["model_dir"], checkpoint)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.train()

    # History and epoch offset
    prev_data, epoch_offset = get_epoch_offset(cfg["paths"]["model_dir"])
    history = (
        [{k: float(prev_data[k][i]) for k in prev_data.files}
         for i in range(len(prev_data["epoch"]))]
        if prev_data is not None else []
    )
    best_loss = min((d["total"] for d in history), default=float("inf"))
    phys_min = float("inf")  # reset to avoid stale comparisons

    # Dataset: final curriculum state (all Ra, full dt range)
    ra_list = cfg["curriculum"]["ra_schedule"][-1][1]
    dt_max = cfg["curriculum"]["dt_schedule"][-1][1]
    dataset = RealisationDataset(
        cfg["paths"]["data_dir"], ra_list, dt_max,
    )
    loader = DataLoader(
        dataset, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=0,
    )
    c_mean = dataset.c_mean
    c_std = dataset.c_std
    t_scale = dataset.t_scale
    source_np = dataset.ensemble[ra_list[-1]]["source"]

    # Optimizer: fresh cosine schedule with conservative LR
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=500, eta_min=cfg["training"]["lr_min"]
    )

    print(f"Loaded checkpoint  : {ckpt_path}")
    print(f"Resuming from      : epoch {epoch_offset}")
    print(f"Wall-time limit    : {max_hours} h")
    print(f"LR                 : {cfg['training']['lr']:.1e}")
    print(f"W_L2={cfg['training']['w_l2']}  "
          f"W_PHYS={cfg['training']['w_phys']}  "
          f"W_BC={cfg['training']['w_bc']}")
    print(f"Best loss so far   : {best_loss:.4e}")
    print("=" * 65)

    t_start = time.time()
    local_ep = 0

    while True:
        elapsed = time.time() - t_start
        if elapsed >= max_seconds:
            print(f"\nWall-time limit reached ({elapsed/3600:.2f} h) — stopping.")
            break

        local_ep += 1
        epoch = epoch_offset + local_ep
        model.train()
        ep_losses = {k: 0.0 for k in ["total", "l2", "phys", "bc"]}
        n_batches = 0

        for batch in loader:
            node_feats, t_out_n, c_out_n, ra_n = [x.to(device) for x in batch]

            B = node_feats.shape[0]
            node_feats[:, :, 1] = y_norm.unsqueeze(0).expand(B, -1)
            node_feats[:, :, 2] = x_norm.unsqueeze(0).expand(B, -1)
            node_feats[:, :, 3] = y_norm.unsqueeze(0).expand(B, -1)

            c_preds = batched_forward(
                model, node_feats, edge_index, edge_attr, t_out_n
            )

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
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        history.append({"epoch": float(epoch), **ep_losses, "lr": lr_now})

        raw_phys = ep_losses["phys"] / cfg["training"]["w_phys"]
        phys_min = min(phys_min, raw_phys)

        if ep_losses["total"] < best_loss:
            best_loss = ep_losses["total"]
            torch.save(
                model.state_dict(),
                os.path.join(cfg["paths"]["model_dir"], "chino_best.pt"),
            )

        if local_ep % cfg["training"]["ckpt_every"] == 0:
            save_state(
                history, model, epoch,
                cfg["paths"]["model_dir"], drive_dir, tag=True,
            )
            remaining = (max_seconds - (time.time() - t_start)) / 3600
            print(f"  [ckpt] ep={epoch}  saved  ({remaining:.2f} h remaining)")

        if local_ep % cfg["training"]["log_every"] == 0 or local_ep == 1:
            elapsed_h = (time.time() - t_start) / 3600
            print(f"  ep={epoch:4d} (+{local_ep:3d}) | "
                  f"{elapsed_h:.2f}h/{max_hours:.1f}h | "
                  f"lr={lr_now:.2e} | "
                  f"total={ep_losses['total']:.4e} | "
                  f"l2={ep_losses['l2']:.4e} | "
                  f"phys={ep_losses['phys']:.4e}")

    # Final save
    torch.save(
        model.state_dict(),
        os.path.join(cfg["paths"]["model_dir"], "chino_final.pt"),
    )
    save_state(history, model, epoch, cfg["paths"]["model_dir"], drive_dir)
    print(f"\nFinal epoch  : {epoch}")
    print(f"Best loss    : {best_loss:.4e}")
    print(f"Wall time    : {(time.time()-t_start)/3600:.2f} h")
    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume CHINO training")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="chino_best.pt",
                        help="Checkpoint filename in model_dir")
    parser.add_argument("--max_hours", type=float, default=6.0,
                        help="Wall-time limit in hours")
    parser.add_argument("--drive_dir", default=None,
                        help="Optional path to sync checkpoints after each save")
    args = parser.parse_args()

    cfg = load_config(args.config)
    resume(cfg, args.checkpoint, args.max_hours, drive_dir=args.drive_dir)
