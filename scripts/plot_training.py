"""Plot the training vs. validation loss curve (with the early-stop point).

Reads a ``history.json`` written by ``neuromamba.train`` — a list of
``{"step", "loss", "val_loss", "val_ppl"}`` records — and renders a train/val
loss curve that marks the best (lowest-val-loss) checkpoint and shades the
region where the model starts to overfit. If no history file is found it falls
back to the recorded RTX 3000 run so the figure is always reproducible.

Usage::

    python scripts/plot_training.py                                   # defaults
    python scripts/plot_training.py --history outputs/neuromamba/history.json
    python scripts/plot_training.py --out docs/training_curve.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# The RTX 3000 run on 646 real PDZ sequences (fallback when no history.json).
_RTX3000 = [
    {"step": 100, "loss": 2.1358, "val_loss": 2.0334},
    {"step": 200, "loss": 0.8595, "val_loss": 1.3738},
    {"step": 300, "loss": 0.3597, "val_loss": 1.2135},
    {"step": 400, "loss": 0.2120, "val_loss": 1.2437},
    {"step": 500, "loss": 0.1751, "val_loss": 1.2527},
    {"step": 600, "loss": 0.1847, "val_loss": 1.3046},
    {"step": 700, "loss": 0.1615, "val_loss": 1.3476},
    {"step": 800, "loss": 0.1474, "val_loss": 1.3900},
    {"step": 900, "loss": 0.1097, "val_loss": 1.4166},
]


def load_history(path: Path) -> list[dict]:
    if path.exists():
        recs = json.loads(path.read_text())
        recs = [r for r in recs if r.get("val_loss") is not None]
        if recs:
            return recs
    return _RTX3000


def main() -> int:
    p = argparse.ArgumentParser(description="Plot train/val loss with the early-stop point.")
    p.add_argument("--history", default="outputs/neuromamba/history.json")
    p.add_argument("--out", default="docs/training_curve.png")
    args = p.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = load_history(Path(args.history))
    steps = [r["step"] for r in recs]
    train = [r["loss"] for r in recs]
    val = [r["val_loss"] for r in recs]
    best_i = min(range(len(val)), key=lambda i: val[i])
    best_step, best_val = steps[best_i], val[best_i]

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # shade the overfitting region (after the val-loss minimum)
    if best_i < len(steps) - 1:
        ax.axvspan(best_step, steps[-1], color="#d62728", alpha=0.06)
        ax.text(
            (best_step + steps[-1]) / 2, max(max(train), max(val)) * 0.96,
            "overfitting\n(val loss rising)", ha="center", va="top",
            fontsize=9, color="#b03030",
        )

    ax.plot(steps, train, "-o", color="#1f77b4", lw=2, ms=4, label="train loss")
    ax.plot(steps, val, "-o", color="#d62728", lw=2, ms=4, label="validation loss")

    ax.axvline(best_step, color="#2ca02c", ls="--", lw=1.5)
    ax.plot([best_step], [best_val], "*", color="#2ca02c", ms=16, zorder=5)
    ax.annotate(
        f"best model — early-stop\nstep {best_step}, val ppl {2.718281828 ** best_val:.2f}",
        xy=(best_step, best_val), xytext=(best_step + 60, best_val + 0.35),
        fontsize=9, color="#1a7a1a",
        arrowprops=dict(arrowstyle="->", color="#2ca02c"),
    )

    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("NeuroMamba — train vs. validation loss (RTX 3000, 646 real PDZ sequences)")
    ax.legend(loc="center right", frameon=False)
    ax.grid(True, alpha=0.25)
    ax.margins(x=0.02)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print(f"Wrote {out}  (best val at step {best_step}, val_loss {best_val:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
