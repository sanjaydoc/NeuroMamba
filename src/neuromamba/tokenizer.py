"""Amino-acid tokenizer for the PDZ-Mamba protein language model.

A minimal, dependency-free tokenizer over the canonical 20 amino acids plus
three special tokens. Autoregressive generation needs an explicit ``BOS`` to
start from and an ``EOS`` to stop at; ``PAD`` fills ragged batches and is
ignored by the loss.

Vocabulary (indices are stable and part of the checkpoint contract)::

    0: <pad>   1: <bos>   2: <eos>
    3..22:     A C D E F G H I K L M N P Q R S T V W Y
"""
from __future__ import annotations

# Canonical 20 amino acids (single-letter), fixed order.
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = frozenset(AA_ALPHABET)

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
SPECIAL_TOKENS = (PAD, BOS, EOS)


class ProteinTokenizer:
    """Reversible mapping between amino-acid strings and integer id lists."""

    def __init__(self) -> None:
        self.itos: list[str] = list(SPECIAL_TOKENS) + list(AA_ALPHABET)
        self.stoi: dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD]
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, sequence: str, add_special: bool = True) -> list[int]:
        """Encode an amino-acid string to token ids.

        Non-canonical characters are skipped (they never appear in the 20-AA
        vocabulary), which keeps a stray ``X``/``U`` from crashing training.
        """
        ids = [self.stoi[c] for c in sequence.strip().upper() if c in self.stoi]
        if add_special:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: list[int], stop_at_eos: bool = True) -> str:
        """Decode token ids back to an amino-acid string, dropping specials."""
        out: list[str] = []
        for i in ids:
            tok = self.itos[int(i)]
            if tok == EOS and stop_at_eos:
                break
            if tok in SPECIAL_TOKENS:
                continue
            out.append(tok)
        return "".join(out)
