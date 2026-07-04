"""Autoregressive sampling from the NeuroMamba language model.

Left-to-right generation: start from ``<bos>`` and sample one residue at a time,
feeding each new token back in, until ``<eos>`` or ``max_len``. Supports the
standard trio of decoding controls:

* **temperature** — flattens (>1) or sharpens (<1) the next-token distribution.
* **top-k** — restrict to the ``k`` most-likely residues.
* **top-p (nucleus)** — restrict to the smallest set whose mass exceeds ``p``.

The scan re-runs over the growing prefix each step (O(L²) overall). At protein
lengths (~100) on a small model this is milliseconds and keeps the code simple
and correct; a cached-state decode is a documented optimisation, not needed for
a laptop-scale model.
"""
from __future__ import annotations

from neuromamba.tokenizer import ProteinTokenizer


def _filter_logits(logits, top_k: int = 0, top_p: float = 0.0):
    """Apply top-k and/or nucleus (top-p) masking to a ``(B, vocab)`` logit tensor."""
    import torch

    logits = logits.clone()
    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits[logits < kth] = float("-inf")
    if top_p and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cum > top_p
        # shift so the first token over the threshold is kept
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        remove_idx = remove.scatter(-1, sorted_idx, remove)
        logits[remove_idx] = float("-inf")
    return logits


def sample(
    model,
    tokenizer: ProteinTokenizer,
    n_samples: int = 16,
    max_len: int = 120,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.9,
    device=None,
    seed: int | None = None,
) -> list[str]:
    """Generate ``n_samples`` novel sequences. Returns decoded amino-acid strings."""
    import torch

    model.eval()
    param = next(model.parameters())
    device = device or param.device
    if seed is not None:
        torch.manual_seed(seed)

    tokens = torch.full((n_samples, 1), tokenizer.bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(n_samples, dtype=torch.bool, device=device)

    with torch.no_grad():
        for _ in range(max_len):
            logits = model(tokens)[:, -1, :]  # (B, vocab)
            logits[:, tokenizer.pad_id] = float("-inf")  # never emit <pad>
            logits[:, tokenizer.bos_id] = float("-inf")  # never re-emit <bos>
            logits = logits / max(temperature, 1e-6)
            logits = _filter_logits(logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)  # (B, 1)
            nxt[finished] = tokenizer.pad_id  # once done, pad so decode stops
            tokens = torch.cat([tokens, nxt], dim=1)
            finished = finished | (nxt.squeeze(1) == tokenizer.eos_id)
            if bool(finished.all()):
                break

    return [tokenizer.decode(row.tolist()) for row in tokens]


def sample_from_checkpoint(
    ckpt_path,
    n_samples: int = 16,
    device: str = "auto",
    **kwargs,
) -> list[str]:
    """Load a checkpoint saved by ``neuromamba.train`` and sample from it."""
    import torch

    from neuromamba.mamba import build_mamba
    from neuromamba.utils import resolve_device

    device = resolve_device(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    tokenizer = ProteinTokenizer()
    model = build_mamba(ckpt["model_cfg"], tokenizer.vocab_size, tokenizer.pad_id)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    return sample(model, tokenizer, n_samples=n_samples, device=device, **kwargs)
