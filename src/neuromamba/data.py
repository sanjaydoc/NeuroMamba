"""Sequence data layer for NeuroMamba.

Two sources, following a "real-data-first, synthetic fallback" pattern:

* **Real** — PDZ-domain sequences (Pfam ``PF00595`` / InterPro ``IPR001478``)
  fetched by ``scripts/download_pdz_family.py`` and stored as a JSONL of
  ``{"sequence": ...}`` records. PDZ domains scaffold synaptic signalling
  (e.g. PSD-95), which is why they are a fitting neuro-relevant demo target.
* **Synthetic** — a clearly-labelled generator of PDZ-*like* sequences built
  from the domain's known secondary-structure layout (β1-β2-β3-α1-β4-α2-β5-β6),
  so training, tests, and CI run before anything is downloaded.

The dataset yields token tensors; ``collate_pad`` builds a padded batch.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from neuromamba.tokenizer import AA_ALPHABET, ProteinTokenizer

LOGGER = logging.getLogger("neuromamba")


# --- synthetic PDZ-like sequences -------------------------------------------
# Coarse residue propensities per structural element of a canonical PDZ fold.
# Not a validated model — a labelled placeholder so the pipeline runs offline.
_ELEMENTS = [
    ("EKLVIVYAT", 6),   # β1  (buried hydrophobic + charged edge)
    ("GLGFSILG", 5),    # β2  (carboxylate-binding loop region, Gly-rich)
    ("KPGDVILA", 6),    # β3
    ("EDLRKALEQ", 8),   # α1  (helical, charged)
    ("GDRILAVNG", 7),   # β4
    ("SVEELRKAE", 9),   # α2  (helical)
    ("KVTLTVLR", 6),    # β5
    ("GPKEGDVI", 6),    # β6
]


def synthetic_pdz_sequences(n: int, seed: int = 0, jitter: float = 0.15) -> list[str]:
    """Generate ``n`` PDZ-like sequences from the element propensities.

    ``jitter`` is the per-residue probability of substituting a random amino
    acid, injecting the noise a real family shows. Lengths land near the ~90-aa
    PDZ domain size.
    """
    rng = random.Random(seed)
    seqs: list[str] = []
    for _ in range(n):
        chars: list[str] = []
        for motif, length in _ELEMENTS:
            for _ in range(length):
                if rng.random() < jitter:
                    chars.append(rng.choice(AA_ALPHABET))
                else:
                    chars.append(rng.choice(motif))
        trim = rng.randint(0, 4)  # small length variation
        seqs.append("".join(chars[: len(chars) - trim]))
    return seqs


# --- IO ---------------------------------------------------------------------
def load_sequences_jsonl(path: str | Path) -> list[str]:
    """Load sequences from a JSONL file of ``{"sequence": ...}`` records."""
    path = Path(path)
    seqs: list[str] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            seq = rec.get("sequence") if isinstance(rec, dict) else rec
            if seq:
                seqs.append(str(seq).strip().upper())
    return seqs


def save_sequences_jsonl(sequences: list[str], path: str | Path) -> Path:
    """Write sequences to a JSONL file, returning the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for seq in sequences:
            fh.write(json.dumps({"sequence": seq}) + "\n")
    return path


def clean_sequences(
    sequences: list[str], min_len: int = 40, max_len: int = 140
) -> list[str]:
    """Uppercase, drop non-canonical residues, length-filter, and de-duplicate."""
    aa_set = set(AA_ALPHABET)
    seen: set[str] = set()
    out: list[str] = []
    for seq in sequences:
        seq = "".join(c for c in seq.strip().upper() if c in aa_set)
        if min_len <= len(seq) <= max_len and seq not in seen:
            seen.add(seq)
            out.append(seq)
    return out


# --- torch Dataset ----------------------------------------------------------
def build_sequence_dataset(
    tokenizer: ProteinTokenizer,
    data_path: str | Path | None = None,
    synthetic: bool = False,
    n_synthetic: int = 512,
    min_len: int = 40,
    max_len: int = 140,
    seed: int = 0,
):
    """Return a torch ``Dataset`` of tokenised sequences plus the raw string list.

    Falls back to synthetic data when ``synthetic`` is set or no readable
    ``data_path`` is given. Returns ``(dataset, sequences)`` so downstream
    novelty metrics can compare generations against the exact training set.
    """
    from torch.utils.data import Dataset

    if synthetic or data_path is None or not Path(data_path).exists():
        raw = synthetic_pdz_sequences(n_synthetic, seed=seed)
        LOGGER.info("Using %d SYNTHETIC PDZ-like sequences.", len(raw))
    else:
        raw = load_sequences_jsonl(data_path)
        LOGGER.info("Loaded %d sequences from %s.", len(raw), data_path)
    seqs = clean_sequences(raw, min_len=min_len, max_len=max_len)
    if not seqs:  # degenerate real file -> keep the loop runnable
        seqs = clean_sequences(synthetic_pdz_sequences(n_synthetic, seed=seed))
        LOGGER.warning("No usable real sequences; fell back to synthetic.")

    class SequenceDataset(Dataset):
        def __init__(self, sequences: list[str]) -> None:
            self.sequences = sequences
            self.tok = tokenizer

        def __len__(self) -> int:
            return len(self.sequences)

        def __getitem__(self, idx: int):
            import torch

            ids = self.tok.encode(self.sequences[idx], add_special=True)
            return torch.tensor(ids, dtype=torch.long)

    return SequenceDataset(seqs), seqs


def collate_pad(batch, pad_id: int):
    """Pad a list of 1-D token tensors to a batch ``(B, L_max)``."""
    import torch

    max_len = max(t.numel() for t in batch)
    out = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, t in enumerate(batch):
        out[i, : t.numel()] = t
    return out
