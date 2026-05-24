"""
Loss functions for CHINO training.

Primary loss: anomaly relative L2.
    Penalizes errors in spatial structure (finger positions) rather than
    the smooth background concentration. The anomaly is the field minus
    its spatial mean; the network must learn WHERE concentration varies,
    not just the overall level.

Regularizers:
    Physics residual (transport PDE) and Neumann no-flux boundary conditions.
"""

import torch


def anomaly_l2(
    c_pred: torch.Tensor,
    c_target: torch.Tensor,
    anomaly_weight: float = 0.7,
) -> torch.Tensor:
    """
    Anomaly relative L2 loss between predicted and target concentration fields.

    L = anomaly_weight * ||anomaly_pred - anomaly_true|| / ||anomaly_true||
      + (1 - anomaly_weight) * ||c_pred - c_true|| / ||c_true||

    The anomaly field is defined as the field minus its spatial mean.
    This prevents the trivial solution of predicting the spatial mean
    everywhere, which scores well on standard relative L2 but captures
    no finger structure.

    Args:
        c_pred: (N,) predicted concentration (normalized)
        c_target: (N,) target concentration (normalized)
        anomaly_weight: weight on the anomaly term (default 0.7)

    Returns:
        scalar loss value
    """
    eps = 1e-8

    c_pred_anom = c_pred - c_pred.mean()
    c_tgt_anom = c_target - c_target.mean()

    l2_anom = torch.sqrt(
        ((c_pred_anom - c_tgt_anom) ** 2).sum() /
        (c_tgt_anom ** 2).sum().clamp(min=eps)
    )
    l2_full = torch.sqrt(
        ((c_pred - c_target) ** 2).sum() /
        c_target.pow(2).sum().clamp(min=eps)
    )

    return anomaly_weight * l2_anom + (1.0 - anomaly_weight) * l2_full


def physics_residual(
    c_pred_norm: torch.Tensor,
    c_in_norm: torch.Tensor,
    t_in_n: torch.Tensor,
    t_out_n: torch.Tensor,
    c_mean: float,
    c_std: float,
    t_scale: float,
    source_np,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    pe: float,
    da_s: float,
    da_inj: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Finite-difference transport PDE residual.

    Computes the residual of the non-dimensional advection-diffusion-reaction
    equation:
        dc/dt + u * dc/dx + v * dc/dy = (1/Pe) * laplacian(c)
                                        - Da_S * c + Da_inj * S(x,y)

    The Darcy velocity is implicit; the residual uses a first-order
    finite-difference approximation of dc/dt and the diffusion Laplacian.

    Args:
        c_pred_norm: (N,) predicted normalized concentration
        c_in_norm:   (N,) input normalized concentration
        t_in_n:      scalar normalized input time
        t_out_n:     scalar normalized output time
        c_mean, c_std: normalization constants
        t_scale:     time scale for denormalization
        source_np:   (ny, nx) numpy array of injection source field
        nx, ny:      grid dimensions
        dx, dy:      grid spacing
        pe:          Peclet number
        da_s:        dissolution sink Damkohler number
        da_inj:      injection Damkohler number
        device:      torch device

    Returns:
        scalar mean squared residual
    """
    dt = (t_out_n - t_in_n).clamp(min=1e-4) * t_scale

    c_out = (c_pred_norm * c_std + c_mean).reshape(ny, nx)
    c_in = (c_in_norm * c_std + c_mean).reshape(ny, nx)

    dc_dt = (c_out - c_in) / dt

    lap = torch.zeros_like(c_out)
    lap[:, 1:-1] += (c_out[:, 2:] - 2 * c_out[:, 1:-1] + c_out[:, :-2]) / dx ** 2
    lap[1:-1, :] += (c_out[2:, :] - 2 * c_out[1:-1, :] + c_out[:-2, :]) / dy ** 2

    S = torch.tensor(source_np, dtype=torch.float32, device=device)
    res = dc_dt - (1.0 / pe) * lap + da_s * c_out - da_inj * S

    return res.pow(2).mean()


def bc_residual(
    c_pred_norm: torch.Tensor,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
) -> torch.Tensor:
    """
    Neumann no-flux boundary condition residual on left, right, and bottom walls.

    Penalizes non-zero normal concentration gradients at the three solid
    boundaries. The top boundary is a Dirichlet injection boundary and is
    not included here.

    Args:
        c_pred_norm: (N,) predicted normalized concentration
        nx, ny:      grid dimensions
        dx, dy:      grid spacing

    Returns:
        scalar mean squared BC residual
    """
    c = c_pred_norm.reshape(ny, nx)
    bc_left = ((c[:, 1] - c[:, 0]) / dx).pow(2).mean()
    bc_right = ((c[:, -1] - c[:, -2]) / dx).pow(2).mean()
    bc_bottom = ((c[0, :] - c[1, :]) / dy).pow(2).mean()
    return (bc_left + bc_right + bc_bottom) / 3.0


def compute_losses(
    c_pred: torch.Tensor,
    c_in_norm: torch.Tensor,
    c_out_norm: torch.Tensor,
    t_in_n: torch.Tensor,
    t_out_n: torch.Tensor,
    w_l2: float,
    w_phys: float,
    w_bc: float,
    **physics_kwargs,
) -> dict:
    """
    Compute all loss components for one training sample.

    Args:
        c_pred:    (N,) predicted normalized concentration
        c_in_norm: (N,) input normalized concentration
        c_out_norm:(N,) target normalized concentration
        t_in_n:    scalar normalized input time
        t_out_n:   scalar normalized output time
        w_l2:      weight for anomaly L2 loss
        w_phys:    weight for physics residual
        w_bc:      weight for boundary condition residual
        **physics_kwargs: passed directly to physics_residual()

    Returns:
        dict with keys: total, l2, phys, bc
    """
    l2 = w_l2 * anomaly_l2(c_pred, c_out_norm)
    phys = w_phys * physics_residual(
        c_pred, c_in_norm, t_in_n, t_out_n, **physics_kwargs
    )
    bc = w_bc * bc_residual(
        c_pred,
        physics_kwargs["nx"],
        physics_kwargs["ny"],
        physics_kwargs["dx"],
        physics_kwargs["dy"],
    )
    return {
        "total": l2 + phys + bc,
        "l2":    l2,
        "phys":  phys,
        "bc":    bc,
    }
