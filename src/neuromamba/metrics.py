"""Generation-quality metrics for NeuroMamba samples.

All torch-free (pure Python + ``difflib``) so they run in CI and on a bare
machine. The four questions a generative protein model must answer:

* **validity**  — are samples real, canonical, sensibly-sized sequences?
* **uniqueness** — does the model collapse to a few motifs, or produce variety?
* **novelty**   — are samples *new*, or memorised copies of the training set?
* **diversity** — how different are samples from *each other*?

Sequence identity uses ``difflib.SequenceMatcher`` ratio, a length-robust
similarity in ``[0, 1]`` that needs no alignment dependency — adequate for the
short, same-family sequences here.
"""
from __future__ import annotations

from difflib import SequenceMatcher

from neuromamba.tokenizer import AA_SET


def identity(a: str, b: str) -> float:
    """Similarity ratio in ``[0, 1]`` between two sequences (1.0 == identical)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def is_valid(seq: str, min_len: int = 20, max_len: int = 200) -> bool:
    """True if non-empty, canonical residues only, and within a length window."""
    return (
        bool(seq)
        and min_len <= len(seq) <= max_len
        and all(c in AA_SET for c in seq)
    )


def max_identity_to_set(seq: str, reference: list[str]) -> float:
    """Highest identity of ``seq`` to any sequence in ``reference``."""
    if not reference:
        return 0.0
    return max(identity(seq, r) for r in reference)


def evaluate(
    samples: list[str],
    training: list[str] | None = None,
    novelty_threshold: float = 0.8,
    min_len: int = 20,
    max_len: int = 200,
) -> dict:
    """Summarise a batch of generated sequences.

    Args:
        samples: generated amino-acid strings.
        training: training sequences to measure novelty against (optional).
        novelty_threshold: a sample is "novel" if its max identity to the
            training set is below this.

    Returns:
        A metrics dict (fractions in ``[0, 1]``; higher is better except
        ``mean_max_train_identity``, where lower means more novel).
    """
    n = len(samples)
    if n == 0:
        return {"n": 0}

    valid = [s for s in samples if is_valid(s, min_len, max_len)]
    lengths = [len(s) for s in samples]
    unique = set(samples)

    out: dict = {
        "n": n,
        "validity": len(valid) / n,
        "uniqueness": len(unique) / n,
        "mean_length": sum(lengths) / n,
        "min_length": min(lengths),
        "max_length": max(lengths),
    }

    # Internal diversity: 1 - mean pairwise identity over a capped sample of pairs.
    if len(valid) >= 2:
        pairs, total = 0, 0.0
        cap = 40  # keep it O(cap^2), not O(n^2), for large batches
        subset = valid[:cap]
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                total += identity(subset[i], subset[j])
                pairs += 1
        out["diversity"] = 1.0 - (total / pairs) if pairs else 0.0

    # Novelty vs. the training set.
    if training:
        max_ids = [max_identity_to_set(s, training) for s in valid] or [1.0]
        mean_max = sum(max_ids) / len(max_ids)
        out["mean_max_train_identity"] = mean_max
        out["novel_fraction"] = sum(
            1 for m in max_ids if m < novelty_threshold
        ) / len(max_ids)

    return out
