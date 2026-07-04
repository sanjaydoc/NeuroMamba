"""Generate novel PDZ-domain sequences and evaluate them.

Loads a trained NeuroMamba checkpoint, samples a library of sequences, scores
them with the built-in proxy oracle, and reports generation-quality metrics
(validity / uniqueness / novelty / diversity). Writes a JSONL of scored designs
and a JSON summary.

If PyTorch or a checkpoint is unavailable, it transparently falls back to the
torch-free order-k **Markov baseline** so the *generate -> score* story runs
anywhere — and so the SSM has an explicit baseline to beat.

Usage::

    python scripts/generate.py --ckpt outputs/neuromamba/model.pt --n 64
    python scripts/generate.py --markov --data data/pdz.jsonl --n 64   # baseline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuromamba import oracle  # noqa: E402
from neuromamba.data import load_sequences_jsonl, synthetic_pdz_sequences  # noqa: E402
from neuromamba.metrics import evaluate  # noqa: E402
from neuromamba.utils import setup_logging  # noqa: E402

LOGGER = setup_logging()


def _load_training(data: str | None) -> list[str]:
    if data and Path(data).exists():
        return load_sequences_jsonl(data)
    return synthetic_pdz_sequences(512)


def generate_markov(n: int, data: str | None, max_len: int, order: int, seed: int):
    from neuromamba.markov import MarkovGenerator

    train = _load_training(data)
    gen = MarkovGenerator(order=order).fit(train)
    LOGGER.info("Markov(order=%d) baseline fitted on %d sequences.", order, len(train))
    return gen.sample(n, max_len=max_len, seed=seed), train


def generate_mamba(ckpt: str, n: int, max_len: int, temperature: float,
                   top_p: float, top_k: int, seed: int):
    import torch

    from neuromamba.generate import sample
    from neuromamba.mamba import build_mamba
    from neuromamba.tokenizer import ProteinTokenizer
    from neuromamba.utils import resolve_device

    device = resolve_device("auto")
    state = torch.load(ckpt, map_location=device)
    tok = ProteinTokenizer()
    model = build_mamba(state["model_cfg"], tok.vocab_size, tok.pad_id).to(device)
    model.load_state_dict(state["state_dict"])
    LOGGER.info("Loaded checkpoint %s (step %s).", ckpt, state.get("global_step"))
    samples = sample(
        model, tok, n_samples=n, max_len=max_len,
        temperature=temperature, top_p=top_p, top_k=top_k, seed=seed,
    )
    return samples, state.get("train_sequences", [])


def main() -> int:
    p = argparse.ArgumentParser(description="Generate + score novel PDZ sequences.")
    p.add_argument("--ckpt", default="outputs/neuromamba/model.pt")
    p.add_argument("--data", default=None, help="Training JSONL for novelty comparison.")
    p.add_argument("--out-dir", default="outputs/generated")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--max-len", type=int, default=130)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--markov", action="store_true", help="Use the torch-free Markov baseline.")
    p.add_argument("--markov-order", type=int, default=3)
    args = p.parse_args()

    use_markov = args.markov or not Path(args.ckpt).exists()
    if use_markov and not args.markov:
        LOGGER.warning("No checkpoint at %s; using Markov baseline.", args.ckpt)

    if use_markov:
        samples, train = generate_markov(
            args.n, args.data, args.max_len, args.markov_order, args.seed
        )
        source = f"markov(order={args.markov_order})"
    else:
        samples, train = generate_mamba(
            args.ckpt, args.n, args.max_len,
            args.temperature, args.top_p, args.top_k, args.seed,
        )
        source = "neuromamba"
        if not train and args.data:
            train = _load_training(args.data)

    # Score + evaluate.
    scores = oracle.score_batch(samples)
    metrics = evaluate(samples, training=train)
    metrics["mean_proxy_stability"] = round(
        sum(s.stability for s in scores) / max(len(scores), 1), 4
    )
    metrics["mean_proxy_solubility"] = round(
        sum(s.solubility for s in scores) / max(len(scores), 1), 4
    )
    metrics["mean_proxy_pdz_binding"] = round(
        sum(s.pdz_binding for s in scores) / max(len(scores), 1), 4
    )
    metrics["source"] = source

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "designs.jsonl").open("w") as fh:
        for seq, sc in zip(samples, scores, strict=True):
            fh.write(json.dumps({"sequence": seq, **sc.__dict__}) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    LOGGER.info("=== Generation report (%s) ===", source)
    for k, v in metrics.items():
        LOGGER.info("  %-26s %s", k, v)
    LOGGER.info("Wrote designs + metrics -> %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
