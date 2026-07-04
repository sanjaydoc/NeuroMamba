"""Torch-free order-k Markov fallback generator.

A dependency-free baseline generator so the *train -> generate -> score*
pipeline (and CI) runs on a machine without PyTorch, and so the Mamba model has
an honest baseline to beat. It learns per-context next-residue frequencies from
the training sequences and samples new ones left-to-right.

This is deliberately weak — a k-gram model has no long-range state, which is
exactly the capability the selective-SSM adds. Comparing NeuroMamba's novelty /
proxy scores against this baseline makes the SSM's contribution measurable.
"""
from __future__ import annotations

import random
from collections import defaultdict

from neuromamba.tokenizer import AA_ALPHABET

_END = "$"  # end-of-sequence sentinel


class MarkovGenerator:
    """Order-``k`` character Markov chain over amino acids."""

    def __init__(self, order: int = 3) -> None:
        self.order = order
        self.table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.lengths: list[int] = []
        self._fitted = False

    def fit(self, sequences: list[str]) -> MarkovGenerator:
        for seq in sequences:
            seq = seq.strip().upper()
            if not seq:
                continue
            self.lengths.append(len(seq))
            padded = seq + _END
            for i in range(len(padded)):
                ctx = seq[max(0, i - self.order) : i]
                nxt = padded[i]
                self.table[ctx][nxt] += 1
        self._fitted = True
        return self

    def _sample_next(self, ctx: str, rng: random.Random) -> str:
        # back off to shorter contexts when the full one is unseen
        for start in range(len(ctx)):
            key = ctx[start:]
            if key in self.table:
                counts = self.table[key]
                choices, weights = zip(*counts.items(), strict=True)
                return rng.choices(choices, weights=weights, k=1)[0]
        if "" in self.table:
            counts = self.table[""]
            choices, weights = zip(*counts.items(), strict=True)
            return rng.choices(choices, weights=weights, k=1)[0]
        return rng.choice(AA_ALPHABET)

    def sample(
        self, n: int, max_len: int = 140, seed: int | None = None
    ) -> list[str]:
        """Generate ``n`` sequences; stops at the end sentinel or ``max_len``."""
        rng = random.Random(seed)
        if not self._fitted:
            raise RuntimeError("MarkovGenerator.sample called before fit().")
        out: list[str] = []
        for _ in range(n):
            seq = ""
            for _ in range(max_len):
                nxt = self._sample_next(seq[-self.order :], rng)
                if nxt == _END:
                    break
                seq += nxt
            out.append(seq)
        return out
