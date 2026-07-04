"""A from-scratch selective state-space (Mamba / S6) protein language model.

Implemented from first principles — no ``mamba-ssm`` CUDA-kernel dependency — so
it trains on a 6 GB laptop GPU or plain CPU. The state recurrence is run as a
**work-efficient parallel prefix scan** (Hillis-Steele) in ``⌈log₂ L⌉``
vectorised passes rather than an ``L``-step Python loop, so the GPU is actually
utilised — no fused CUDA kernel required. A reference sequential scan is kept for
clarity and is asserted numerically identical in the tests.

What makes it *selective* (the S6 contribution over classic S4): the SSM
parameters ``Δ`` (step size), ``B`` (input matrix) and ``C`` (output matrix) are
**functions of the input token**, so the model chooses, per position, what to
remember and what to forget. Classic S4 uses fixed ``Δ, B, C`` and cannot do
this content-based routing.

Per Mamba mixer block::

    x, z        = in_proj(u)                      # split into path + gate
    x           = SiLU(causal_depthwise_conv(x))  # local mixing, strictly causal
    Δ, B, C     = selective projections of x       # input-dependent SSM params
    Ā, B̄        = discretize(A, B, Δ)               # zero-order-hold
    h_t         = Ā_t · h_{t-1} + B̄_t · x_t         # linear recurrent state
    y_t         = C_t · h_t + D · x_t
    out         = out_proj(y · SiLU(z))            # gated output

The recurrence only ever looks backward, so the model is inherently causal and
suited to left-to-right (autoregressive) generation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class MambaBlock(nn.Module):
    """A single selective-SSM (Mamba) mixer block.

    Args:
        d_model: residual-stream width.
        d_state: SSM latent state size ``N`` (per channel).
        d_conv: causal depthwise conv kernel width.
        expand: inner-width expansion factor (``d_inner = expand * d_model``).
        dt_rank: rank of the low-rank ``Δ`` projection ("auto" -> ceil(d_model/16)).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Strictly causal depthwise conv: left-pad by (d_conv - 1) and crop the tail.
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )

        # Input-dependent Δ, B, C come from a single projection of x.
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A is parameterised in log space and kept negative (stable, decaying state).
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Args: ``u`` ``(B, L, d_model)``. Returns ``(B, L, d_model)``."""
        _, seq_len, _ = u.shape
        x_and_z = self.in_proj(u)  # (B, L, 2*d_inner)
        x, z = x_and_z.chunk(2, dim=-1)

        # Causal depthwise conv over the time axis, then crop back to L.
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[..., :seq_len]
        x = x.transpose(1, 2)  # (B, L, d_inner)
        x = F.silu(x)

        y = self._selective_scan(x)
        y = y * F.silu(z)
        return self.out_proj(y)

    def _discretize(self, x: torch.Tensor):
        """Return the per-step selective SSM tensors ``(deltaA, deltaB_x, C)``.

        Shapes: ``deltaA``/``deltaB_x`` are ``(B, L, d_inner, N)``; ``C`` is
        ``(B, L, N)``.
        """
        n = self.d_state
        A = -torch.exp(self.A_log)  # (d_inner, N), negative => stable

        proj = self.x_proj(x)  # (B, L, dt_rank + 2N)
        dt, B_mat, C_mat = torch.split(proj, [self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(dt))  # (B, L, d_inner), > 0

        # Discretize: Ā = exp(Δ ⊙ A); B̄ x ≈ Δ ⊙ B ⊙ x  (ZOH, simplified B̄).
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, d_inner, N)
        deltaB_x = delta.unsqueeze(-1) * B_mat.unsqueeze(2) * x.unsqueeze(-1)
        return deltaA, deltaB_x, C_mat

    def _selective_scan(self, x: torch.Tensor) -> torch.Tensor:
        """Work-efficient **parallel** selective scan (Hillis-Steele prefix scan).

        The state recurrence ``h_t = Ā_t · h_{t-1} + B̄_t·x_t`` is a first-order
        linear recurrence, which is *associative*: composing two steps
        ``(a₁,b₁)`` then ``(a₂,b₂)`` gives ``(a₂a₁, a₂b₁+b₂)``. So the whole scan
        can be done as an inclusive parallel prefix-scan over that operator in
        ``⌈log₂ L⌉`` vectorised passes instead of ``L`` sequential Python steps —
        no per-timestep loop, no ``mamba-ssm`` CUDA kernel, and the GPU is
        actually utilised. It is numerically identical to the sequential scan
        (see :meth:`_selective_scan_sequential`) and stable: the multiplicative
        factors ``Ā ∈ (0, 1]`` only ever shrink, so nothing overflows.
        """
        deltaA, deltaB_x, C_mat = self._discretize(x)
        seq_len = x.shape[1]

        a = deltaA          # (B, L, d_inner, N) multiplicative coefficients
        b = deltaB_x        # (B, L, d_inner, N) additive terms
        d = 1
        while d < seq_len:
            # identity element for the first d positions: a=1, b=0 (no-op).
            a_prev = F.pad(a[:, : seq_len - d], (0, 0, 0, 0, d, 0), value=1.0)
            b_prev = F.pad(b[:, : seq_len - d], (0, 0, 0, 0, d, 0), value=0.0)
            b = a * b_prev + b
            a = a * a_prev
            d *= 2

        h = b  # after the scan, b[:, t] == h_t
        y = torch.einsum("bldn,bln->bld", h, C_mat)  # (B, L, d_inner)
        return y + x * self.D  # skip connection

    def _selective_scan_sequential(self, x: torch.Tensor) -> torch.Tensor:
        """Reference O(L) sequential scan — kept for clarity and equivalence tests."""
        b, seq_len, d_inner = x.shape
        deltaA, deltaB_x, C_mat = self._discretize(x)
        h = x.new_zeros(b, d_inner, self.d_state)
        ys = []
        for t in range(seq_len):
            h = deltaA[:, t] * h + deltaB_x[:, t]
            ys.append(torch.einsum("bdn,bn->bd", h, C_mat[:, t]))
        y = torch.stack(ys, dim=1)
        return y + x * self.D


class ResidualMambaLayer(nn.Module):
    """Pre-norm residual wrapper around a :class:`MambaBlock`."""

    def __init__(self, d_model: int, **block_kwargs) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = MambaBlock(d_model, **block_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class ProteinMamba(nn.Module):
    """Autoregressive protein language model built from selective-SSM blocks.

    Small by design (a few M params at the defaults) so it fits a 6 GB laptop
    GPU and trains in resumable batches. Embedding and LM-head weights are tied,
    which regularises the small vocabulary and cuts parameters.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.layers = nn.ModuleList(
            [
                ResidualMambaLayer(
                    d_model, d_state=d_state, d_conv=d_conv, expand=expand
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # weight tying

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Args: ``tokens`` ``(B, L)`` ids. Returns logits ``(B, L, vocab)``."""
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return self.lm_head(x)

    def num_params(self) -> int:
        """Total parameter count, counting tied weights once."""
        seen: set[int] = set()
        total = 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total


def build_mamba(cfg: dict, vocab_size: int, pad_id: int = 0) -> ProteinMamba:
    """Construct a :class:`ProteinMamba` from a plain-dict config node."""
    return ProteinMamba(
        vocab_size=vocab_size,
        d_model=int(cfg.get("d_model", 128)),
        n_layers=int(cfg.get("n_layers", 6)),
        d_state=int(cfg.get("d_state", 16)),
        d_conv=int(cfg.get("d_conv", 4)),
        expand=int(cfg.get("expand", 2)),
        pad_id=pad_id,
    )
