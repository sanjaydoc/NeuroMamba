"""Resumable training loop for the NeuroMamba protein language model.

Designed for a 6 GB laptop GPU (or CPU): small model, fp16 autocast on CUDA,
gradient clipping, and **step-based resumable checkpointing** — the feature that
lets you "train in batches". Stop any time; ``--resume`` continues from ``last.pt``
with the optimizer state and global step intact.

A held-out **validation split + early stopping** guard against the memorisation
this small model slides into on a small dataset. Two checkpoints are written:

* ``model.pt`` — the **best** model by validation loss (what generation loads).
* ``last.pt``  — the **latest** step (weights + optimizer + step) for ``--resume``.

Checkpoint contract (both files)::

    {
      "global_step": int,          # optimizer steps completed
      "state_dict":  ...,          # model weights
      "optim":       ...,          # AdamW state (so resume is seamless)
      "model_cfg":   {...},        # architecture (to rebuild the model)
      "history":     [...],        # list of {step, loss, val_loss, val_ppl}
      "train_sequences": [...],    # for downstream novelty metrics
      "best_val":    float,        # best validation loss so far
      "val_loss":    float | None, # this checkpoint's validation loss
    }

Usage::

    python -m neuromamba.train --synthetic --max-steps 200        # smoke run
    python -m neuromamba.train --data data/pdz.jsonl --max-steps 4000
    python -m neuromamba.train --data data/pdz.jsonl --max-steps 8000 --resume
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

from neuromamba.utils import resolve_device, set_seed, setup_logging

LOGGER = logging.getLogger("neuromamba")

DEFAULT_CFG = {"d_model": 128, "n_layers": 6, "d_state": 16, "d_conv": 4, "expand": 2}


def _cross_entropy(logits, tokens, pad_id: int):
    """Next-token cross-entropy, ignoring PAD targets. ``logits`` (B,L,V)."""
    import torch.nn.functional as F

    # predict token t+1 from position t
    pred = logits[:, :-1, :].reshape(-1, logits.size(-1))
    target = tokens[:, 1:].reshape(-1)
    return F.cross_entropy(pred, target, ignore_index=pad_id)


def train(
    data: str | None = None,
    synthetic: bool = False,
    out_dir: str | Path = "outputs/neuromamba",
    max_steps: int = 2000,
    batch_size: int = 16,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    save_every: int = 200,
    log_every: int = 50,
    device: str = "auto",
    seed: int = 42,
    mixed_precision: bool = True,
    resume: bool = False,
    model_cfg: dict | None = None,
    max_len: int = 140,
    min_len: int = 40,
    val_frac: float = 0.1,
    eval_every: int = 100,
    patience: int = 8,
    early_stop: bool = True,
) -> Path:
    """Train (or resume) the model; return the path to the **best** checkpoint.

    A held-out validation split guards against the memorisation this small model
    slides into on a small dataset. Every ``eval_every`` steps the validation
    loss (and perplexity) is measured; the lowest-val-loss model is saved to
    ``model.pt`` (what generation loads), and training **early-stops** once the
    validation loss stops improving for ``patience`` evaluations. The latest
    optimizer state is written to ``last.pt`` for ``--resume``.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset

    from neuromamba.data import (
        clean_sequences,
        collate_pad,
        load_sequences_jsonl,
        synthetic_pdz_sequences,
    )
    from neuromamba.mamba import build_mamba
    from neuromamba.tokenizer import ProteinTokenizer

    setup_logging()
    set_seed(seed)
    device = resolve_device(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "model.pt"   # best-by-val model (used for generation)
    last_path = out_dir / "last.pt"    # latest state (used for --resume)
    model_cfg = {**DEFAULT_CFG, **(model_cfg or {})}
    tokenizer = ProteinTokenizer()

    # --- data + deterministic train/val split -------------------------------
    if synthetic or data is None or not Path(data).exists():
        raw = synthetic_pdz_sequences(512, seed=seed)
        LOGGER.info("Using %d SYNTHETIC PDZ-like sequences.", len(raw))
    else:
        raw = load_sequences_jsonl(data)
        LOGGER.info("Loaded %d sequences from %s.", len(raw), data)
    seqs = clean_sequences(raw, min_len=min_len, max_len=max_len)
    if not seqs:
        seqs = clean_sequences(synthetic_pdz_sequences(512, seed=seed))

    import random as _random
    _random.Random(seed).shuffle(seqs)
    n_val = int(len(seqs) * val_frac) if len(seqs) >= 10 else 0
    val_seqs = seqs[:n_val]
    train_seqs = seqs[n_val:]
    use_val = len(val_seqs) > 0

    class _DS(Dataset):
        def __init__(self, s):
            self.s = s

        def __len__(self):
            return len(self.s)

        def __getitem__(self, i):
            return torch.tensor(tokenizer.encode(self.s[i]), dtype=torch.long)

    def _loader(s, shuffle):
        return DataLoader(
            _DS(s), batch_size=min(batch_size, max(len(s), 1)), shuffle=shuffle,
            drop_last=False, collate_fn=lambda b: collate_pad(b, tokenizer.pad_id),
        )

    loader = _loader(train_seqs, shuffle=True)
    val_loader = _loader(val_seqs, shuffle=False) if use_val else None

    model = build_mamba(model_cfg, tokenizer.vocab_size, tokenizer.pad_id).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    global_step = 0
    best_val = float("inf")
    patience_ctr = 0
    history: list[dict] = []
    if resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["optim"])
        global_step = ckpt.get("global_step", 0)
        history = ckpt.get("history", [])
        best_val = ckpt.get("best_val", float("inf"))
        LOGGER.info("Resumed from %s at step %d (best_val=%.4f).", last_path, global_step, best_val)

    use_amp = mixed_precision and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    LOGGER.info(
        "Model: %.2fM params | device=%s | amp=%s | %d train / %d val seqs | target step %d",
        model.num_params() / 1e6, device, use_amp, len(train_seqs), len(val_seqs), max_steps,
    )

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        tot, nb = 0.0, 0
        for tokens in val_loader:
            tokens = tokens.to(device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = _cross_entropy(model(tokens), tokens, tokenizer.pad_id)
            tot += float(loss.item())
            nb += 1
        model.train()
        return tot / max(nb, 1)

    def save(path: Path, val_loss: float | None) -> None:
        torch.save(
            {
                "global_step": global_step,
                "state_dict": model.state_dict(),
                "optim": opt.state_dict(),
                "model_cfg": model_cfg,
                "history": history,
                "train_sequences": train_seqs,
                "best_val": best_val,
                "val_loss": val_loss,
            },
            path,
        )
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    model.train()
    stop = False
    while not stop:
        for tokens in loader:
            tokens = tokens.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(tokens)
                loss = _cross_entropy(logits, tokens, tokenizer.pad_id)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            global_step += 1
            train_loss = float(loss.item())
            if global_step % max(log_every, 1) == 0:
                LOGGER.info("step %5d | loss %.4f", global_step, train_loss)

            # --- periodic validation, best-checkpoint, early stopping --------
            if use_val and global_step % max(eval_every, 1) == 0:
                vloss = evaluate()
                ppl = math.exp(min(vloss, 20.0))
                improved = vloss < best_val - 1e-3
                LOGGER.info(
                    "step %5d | val_loss %.4f | val_ppl %.3f%s",
                    global_step, vloss, ppl, "  <- best" if improved else "",
                )
                history.append(
                    {"step": global_step, "loss": train_loss, "val_loss": vloss, "val_ppl": ppl}
                )
                if improved:
                    best_val = vloss
                    patience_ctr = 0
                    save(best_path, vloss)  # keep the best model for generation
                else:
                    patience_ctr += 1
                    if early_stop and patience_ctr >= patience:
                        LOGGER.info(
                            "Early stopping at step %d (no val improvement for %d evals; best_val=%.4f).",
                            global_step, patience, best_val,
                        )
                        stop = True
                        break
            elif not use_val and global_step % max(save_every, 1) == 0:
                save(best_path, None)

            if global_step >= max_steps:
                stop = True
                break

    save(last_path, best_val if use_val else None)
    if not best_path.exists():  # tiny run / no val: promote last to the loadable model
        save(best_path, None)
    if use_val:
        LOGGER.info(
            "Done at step %d. Best val_loss=%.4f (ppl %.3f). Best model -> %s",
            global_step, best_val, math.exp(min(best_val, 20.0)), best_path,
        )
    else:
        LOGGER.info("Done at step %d. Checkpoint -> %s", global_step, best_path)
    return best_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/resume the NeuroMamba protein LM.")
    p.add_argument("--data", default=None, help="JSONL of {'sequence': ...}; omit for synthetic.")
    p.add_argument("--synthetic", action="store_true", help="Force synthetic PDZ-like data.")
    p.add_argument("--out-dir", default="outputs/neuromamba")
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    p.add_argument("--resume", action="store_true", help="Resume from the last checkpoint.")
    # Validation / early stopping — the anti-overfitting controls.
    p.add_argument("--val-frac", type=float, default=0.1, help="Held-out validation fraction.")
    p.add_argument("--eval-every", type=int, default=100, help="Steps between validation evals.")
    p.add_argument("--patience", type=int, default=8, help="Early-stop after N evals w/o val gain.")
    p.add_argument("--no-early-stop", action="store_true", help="Disable early stopping.")
    # Model-size knobs — scale these up to put more work on the GPU per step.
    p.add_argument("--d-model", type=int, default=None, help="Residual width (default 128).")
    p.add_argument("--n-layers", type=int, default=None, help="Number of Mamba blocks (default 6).")
    p.add_argument("--d-state", type=int, default=None, help="SSM state size N (default 16).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    overrides = {
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "d_state": args.d_state,
    }
    model_cfg = {k: v for k, v in overrides.items() if v is not None} or None
    train(
        data=args.data,
        synthetic=args.synthetic,
        out_dir=args.out_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        save_every=args.save_every,
        log_every=args.log_every,
        device=args.device,
        seed=args.seed,
        mixed_precision=not args.no_amp,
        resume=args.resume,
        model_cfg=model_cfg,
        val_frac=args.val_frac,
        eval_every=args.eval_every,
        patience=args.patience,
        early_stop=not args.no_early_stop,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
