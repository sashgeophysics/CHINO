"""
CHINO model architecture: Attention-augmented MeshGraphNet.

The operator maps a seed realization at t_in to the same seed at t_out:

    G_theta : c(x, y, t_in) -> c(x, y, t_out)

Each AttentionMPBlock combines local edge message passing with full
N^2 multi-head self-attention, giving every node a global receptive
field in a single operation. This resolves the finite-hop propagation
failure of standard message-passing GNOs at high Rayleigh numbers,
where the pressure Poisson equation couples the entire domain
instantaneously.

Node input features (6):
    c_in_norm, S_norm, x_norm, y_norm, t_in_norm, Ra_norm

Output:
    c_pred >= 0  (Softplus activation)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def make_mlp(in_dim: int, hidden_dim: int, out_dim: int,
             n_layers: int = 2, layer_norm: bool = True) -> nn.Sequential:
    """Build a multi-layer perceptron with SiLU activations."""
    layers = []
    dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
    for k in range(len(dims) - 1):
        layers.append(nn.Linear(dims[k], dims[k + 1]))
        if k < len(dims) - 2:
            if layer_norm:
                layers.append(nn.LayerNorm(dims[k + 1]))
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Sinusoidal time embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbed(nn.Module):
    """
    Encode a scalar time value as a sinusoidal embedding projected to
    out_dim dimensions via a small MLP.

    The embedding captures the target output time t_out and is injected
    into every processor block so the model represents a continuously
    parameterized family of operators indexed by time.
    """

    def __init__(self, embed_dim: int = 64, out_dim: int = 256):
        super().__init__()
        half = embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) *
            torch.arange(half, dtype=torch.float32) / (half - 1)
        )
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: scalar or (1,) tensor — normalized output time

        Returns:
            embedding: (1, out_dim)
        """
        t = t.view(-1) if t.dim() > 0 else t.unsqueeze(0)
        args = t.unsqueeze(-1) * self.freqs.unsqueeze(0) * 2 * math.pi
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# Multi-head self-attention
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """
    Full O(N^2) multi-head self-attention over all graph nodes.

    Every node attends to every other node simultaneously, giving the
    operator a global receptive field in one step. The attention weights
    are the learned discrete analog of the Green's function of the pressure
    Poisson equation: a_ij encodes how strongly the concentration at node j
    influences the pressure at node i.

    Memory: N x N x H x 4 bytes per forward pass.
    For N=8000, H=4 heads: 4 x 8000^2 x 4B = 1 GB.

    Args:
        hidden_dim: total feature dimension (must be divisible by n_heads)
        n_heads: number of attention heads
    """

    def __init__(self, hidden_dim: int, n_heads: int = 4):
        super().__init__()
        assert hidden_dim % n_heads == 0, \
            "hidden_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k = hidden_dim // n_heads
        self.scale = self.d_k ** -0.5

        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_O = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

        for w in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.xavier_uniform_(w.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (N, hidden_dim) node feature matrix

        Returns:
            attended: (N, hidden_dim)
        """
        N, D = h.shape
        H, dk = self.n_heads, self.d_k

        Q = self.W_Q(h).view(N, H, dk).transpose(0, 1)   # (H, N, dk)
        K = self.W_K(h).view(N, H, dk).transpose(0, 1)
        V = self.W_V(h).view(N, H, dk).transpose(0, 1)

        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # (H, N, N)
        attn = F.softmax(scores, dim=-1)

        out = torch.bmm(attn, V)                          # (H, N, dk)
        out = out.transpose(0, 1).contiguous().view(N, D)
        out = self.W_O(out)

        return self.norm(h + out)


# ---------------------------------------------------------------------------
# Attention-augmented message-passing block
# ---------------------------------------------------------------------------

class AttentionMPBlock(nn.Module):
    """
    One processor block combining local edge message passing with global
    self-attention.

    Step 1 — Local edge update (O(E)):
        f_ij_new = MLP(h_i, h_j, f_ij) + f_ij

    Step 2 — Aggregate local messages:
        msg_i = sum_{j in N(i)} f_ij_new

    Step 3 — Global self-attention (O(N^2)):
        attn_i = MultiHeadSelfAttention(H)[i]

    Step 4 — Node update with residual:
        h_i_new = MLP(h_i, msg_i, attn_i, t_emb) + h_i

    Args:
        hidden_dim: feature dimension throughout
        time_embed_dim: dimension of the time embedding
        n_heads: number of attention heads
    """

    def __init__(self, hidden_dim: int, time_embed_dim: int,
                 n_heads: int = 4):
        super().__init__()
        self.edge_mlp = make_mlp(
            hidden_dim * 2 + hidden_dim, hidden_dim, hidden_dim
        )
        self.attention = MultiHeadSelfAttention(hidden_dim, n_heads)
        self.node_mlp = make_mlp(
            hidden_dim * 3 + time_embed_dim, hidden_dim * 2, hidden_dim
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        f: torch.Tensor,
        edge_index: torch.Tensor,
        t_emb_n: torch.Tensor,
    ):
        """
        Args:
            h: (N, hidden_dim) node states
            f: (E, hidden_dim) edge states
            edge_index: (2, E) source and destination node indices
            t_emb_n: (N, hidden_dim) time embedding broadcast to all nodes

        Returns:
            h_new: (N, hidden_dim)
            f_new: (E, hidden_dim)
        """
        src, dst = edge_index[0], edge_index[1]

        # Step 1-2: local edge update and message aggregation
        f_new = self.edge_mlp(
            torch.cat([h[src], h[dst], f], dim=-1)
        ) + f
        msg = torch.zeros_like(h)
        msg.scatter_add_(0, dst.unsqueeze(-1).expand_as(f_new), f_new)

        # Step 3: global self-attention
        attn_out = self.attention(h)

        # Step 4: node update
        h_new = self.node_mlp(
            torch.cat([h, msg, attn_out, t_emb_n], dim=-1)
        ) + h
        h_new = self.norm(h_new)

        return h_new, f_new


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class AttentionMeshGraphNet(nn.Module):
    """
    CHINO: CHanneling Instability Neural Operator.

    An attention-augmented graph neural operator for predicting individual
    CO2 dissolution plume realizations in the Darcy-Boussinesq system.

    The model takes the concentration field of one seed realization at
    t_in as input and predicts the concentration field of the same seed
    at t_out. The Rayleigh number Ra is provided as a per-node feature,
    allowing the model to generalize across physical regimes within a
    single set of weights.

    Architecture:
        - Node encoder:  MLP(node_in_dim -> hidden_dim), LayerNorm
        - Edge encoder:  MLP(edge_in_dim -> hidden_dim), LayerNorm
        - Time embedding: sinusoidal 64-dim + MLP -> hidden_dim
        - Processor: n_blocks x AttentionMPBlock
        - Decoder: MLP(hidden_dim -> hidden_dim -> 1), Softplus

    Args:
        node_in_dim: number of node input features (default 6)
        edge_in_dim: number of edge input features (default 3)
        hidden_dim: width of all hidden layers (default 256)
        n_blocks: number of processor blocks (default 6)
        time_embed_dim: sinusoidal embedding dimension (default 64)
        n_heads: number of attention heads (default 4)
    """

    def __init__(
        self,
        node_in_dim: int = 6,
        edge_in_dim: int = 3,
        hidden_dim: int = 256,
        n_blocks: int = 6,
        time_embed_dim: int = 64,
        n_heads: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.time_embed = SinusoidalTimeEmbed(
            embed_dim=time_embed_dim, out_dim=hidden_dim
        )
        self.node_encoder = make_mlp(node_in_dim, hidden_dim, hidden_dim)
        self.edge_encoder = make_mlp(edge_in_dim, hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            AttentionMPBlock(hidden_dim, hidden_dim, n_heads)
            for _ in range(n_blocks)
        ])
        self.decoder = make_mlp(
            hidden_dim, hidden_dim, 1, n_layers=3, layer_norm=False
        )
        self.softplus = nn.Softplus()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        t_out_norm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_feats: (N, node_in_dim) — per-node input features
            edge_index: (2, E) — source and destination node indices
            edge_attr:  (E, edge_in_dim) — per-edge geometric features
            t_out_norm: scalar — normalized output time

        Returns:
            c_pred: (N,) — predicted concentration, non-negative
        """
        N = node_feats.shape[0]

        t_scalar = (t_out_norm.view(1) if t_out_norm.dim() == 0
                    else t_out_norm[:1])
        t_emb = self.time_embed(t_scalar)       # (1, hidden_dim)
        t_emb_n = t_emb.expand(N, -1)           # (N, hidden_dim)

        h = self.node_encoder(node_feats)
        f = self.edge_encoder(edge_attr)

        for block in self.blocks:
            h, f = block(h, f, edge_index, t_emb_n)

        out = self.decoder(h).squeeze(-1)
        return self.softplus(out)

    @property
    def n_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())
