"""Resumable training loop for the NeuroMamba protein language model.

Designed for a 6 GB laptop GPU (or CPU): small model, fp16 autocast on CUDA,
gradient clipping, and **step-based resumable checkpointing** — the feature that
lets you "train in batches". Stop any time; ``--resume`` continues from the last
checkpoint with the optimizer state and global step intact.

Checkpoint contract (``ckpt``)::

    {
      "global_step": int,          # optimizer steps completed
      "state_dict":  ...,          # model weights
      "optim":       ...,          # AdamW state (so resume is seamless)
      "model_cfg":   {...},        # architecture (to rebuild the model)
      "history":     [...],        # list of {step, loss} for plotting
      "train_sequences": [...],    # for downstream novelty metrics
    }

Usage::

    python -m neuromamba.train --synthetic --max-steps 200        # smoke run
    python -m neuromamba.train --data data/pdz.jsonl --max-steps 2000
    python -m neuromamba.train --data data/pdz.jsonl --max-steps 4000 --resume
"""
from __future__ import annotations

import argparse
import json
import logging
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
) -> Path:
    """Train (or resume) the model; return the path to the latest checkpoint."""
    import torch
    from torch.utils.data import DataLoader

    from neuromamba.data import build_sequence_dataset, collate_pad
    from neuromamba.mamba import build_mamba
    from neuromamba.tokenizer import ProteinTokenizer

    setup_logging()
    set_seed(seed)
    device = resolve_device(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "model.pt"
    model_cfg = {**DEFAULT_CFG, **(model_cfg or {})}

    tokenizer = ProteinTokenizer()
    dataset, train_seqs = build_sequence_dataset(
        tokenizer, data_path=data, synthetic=synthetic,
        min_len=min_len, max_len=max_len, seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        drop_last=False,
        collate_fn=lambda b: collate_pad(b, tokenizer.pad_id),
    )

    model = build_mamba(model_cfg, tokenizer.vocab_size, tokenizer.pad_id).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    global_step = 0
    history: list[dict] = []
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["optim"])
        global_step = ckpt.get("global_step", 0)
        history = ckpt.get("history", [])
        LOGGER.info("Resumed from %s at step %d.", ckpt_path, global_step)

    use_amp = mixed_precision and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    LOGGER.info(
        "Model: %.2fM params | device=%s | amp=%s | %d train seqs | target step %d",
        model.num_params() / 1e6, device, use_amp, len(train_seqs), max_steps,
    )

    def save() -> None:
        torch.save(
            {
                "global_step": global_step,
                "state_dict": model.state_dict(),
                "optim": opt.state_dict(),
                "model_cfg": model_cfg,
                "history": history,
                "train_sequences": train_seqs,
            },
            ckpt_path,
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
            if global_step % max(log_every, 1) == 0:
                LOGGER.info("step %5d | loss %.4f", global_step, float(loss.item()))
                history.append({"step": global_step, "loss": float(loss.item())})
            if global_step % max(save_every, 1) == 0:
                save()
            if global_step >= max_steps:
                stop = True
                break

    save()
    LOGGER.info("Done at step %d. Checkpoint -> %s", global_step, ckpt_path)
    return ckpt_path


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
    return p.parse_args()


def main() -> int:
    args = parse_args()
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
