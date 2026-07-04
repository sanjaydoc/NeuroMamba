"""Self-contained in-silico proxy scorers for generated sequences.

These are cheap, transparent, **in-silico proxies — not validated wet-lab
assays**. They exist so NeuroMamba demonstrates a full *generate -> score ->
select* loop with zero heavy dependencies (no ESM, no GPU). Each returns a
"higher-is-better" score in a comparable range; ``score`` bundles them.

The design mirrors a real oracle interface: swap any proxy for a learned model
or an experimental readout and the loop is unchanged. Proxies used here:

* **stability**   — hydrophobic-core content + secondary-structure propensity
  balance (a folded domain needs a hydrophobic core and mixed α/β content).
* **solubility**  — penalises high average hydrophobicity (GRAVY) and long
  hydrophobic runs (aggregation-prone), rewards moderate net charge.
* **pdz_binding** — presence and quality of the canonical PDZ carboxylate-
  binding "GLGF" loop (R/K...G-Φ-G-Φ), which forms the peptide-binding groove.

All pure-Python / torch-free.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from neuromamba.tokenizer import AA_SET

# Kyte-Doolittle hydropathy index.
_KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
# Chou-Fasman-style coarse helix / sheet formers.
_HELIX = set("AELMQKRH")
_SHEET = set("VIYCWFT")
_HYDROPHOBIC = set("AVILMFWC")
_POS, _NEG = set("KR"), set("DE")


def _gravy(seq: str) -> float:
    return sum(_KD.get(c, 0.0) for c in seq) / max(len(seq), 1)


def _fraction(seq: str, members: set) -> float:
    return sum(1 for c in seq if c in members) / max(len(seq), 1)


def _longest_hydrophobic_run(seq: str) -> int:
    best = cur = 0
    for c in seq:
        cur = cur + 1 if c in _HYDROPHOBIC else 0
        best = max(best, cur)
    return best


def stability_proxy(seq: str) -> float:
    """Reward a hydrophobic core and balanced α/β secondary-structure content."""
    if not seq:
        return 0.0
    core = _fraction(seq, _HYDROPHOBIC)          # want a real core (~0.3-0.45)
    core_term = 1.0 - abs(core - 0.38) / 0.38
    helix, sheet = _fraction(seq, _HELIX), _fraction(seq, _SHEET)
    balance = 1.0 - abs(helix - sheet)           # want both, not all-helix/all-sheet
    ss_total = min((helix + sheet) / 0.6, 1.0)   # want decent structured fraction
    return max(0.0, 0.5 * core_term + 0.25 * balance + 0.25 * ss_total)


def solubility_proxy(seq: str) -> float:
    """Penalise hydrophobicity / aggregation, reward moderate net charge."""
    if not seq:
        return 0.0
    gravy = _gravy(seq)                           # lower = more soluble
    gravy_term = 1.0 - min(abs(gravy + 0.2) / 2.0, 1.0)
    run = _longest_hydrophobic_run(seq)           # long runs aggregate
    run_term = 1.0 - min(run / 8.0, 1.0)
    net = abs(_fraction(seq, _POS) - _fraction(seq, _NEG)) * len(seq)
    charge_term = 1.0 - min(net / (0.25 * len(seq) + 1e-6), 1.0)
    return max(0.0, 0.45 * gravy_term + 0.35 * run_term + 0.20 * charge_term)


def pdz_binding_proxy(seq: str) -> float:
    """Score presence/quality of the PDZ carboxylate-binding 'GLGF' loop.

    The canonical motif is R/K-X-X-X-G-Φ-G-Φ (Φ = hydrophobic); the G-Φ-G-Φ
    core (classically 'GLGF') lines the groove that grips a peptide C-terminus.
    We scan for the best-matching window and score partial matches too.
    """
    if len(seq) < 4:
        return 0.0
    best = 0.0
    for i in range(len(seq) - 3):
        w = seq[i : i + 4]
        s = 0.0
        s += 0.25 if w[0] == "G" else 0.0
        s += 0.25 if w[1] in _HYDROPHOBIC else 0.0
        s += 0.25 if w[2] == "G" else 0.0
        s += 0.25 if w[3] in _HYDROPHOBIC else 0.0
        # bonus for an upstream basic residue (the R/K anchor)
        if s >= 0.75 and i >= 1 and seq[max(0, i - 4) : i].count("R") + \
                seq[max(0, i - 4) : i].count("K") > 0:
            s = min(1.0, s + 0.1)
        best = max(best, s)
    return best


@dataclass
class ProxyScore:
    """Bundle of proxy objectives (all higher-is-better)."""

    stability: float = 0.0
    solubility: float = 0.0
    pdz_binding: float = 0.0
    details: dict = field(default_factory=dict)

    def as_vector(self) -> list[float]:
        return [self.stability, self.solubility, self.pdz_binding]

    @property
    def mean(self) -> float:
        return sum(self.as_vector()) / 3.0


def score(seq: str) -> ProxyScore:
    """Score one sequence on all three proxies."""
    seq = "".join(c for c in seq.strip().upper() if c in AA_SET)
    return ProxyScore(
        stability=round(stability_proxy(seq), 4),
        solubility=round(solubility_proxy(seq), 4),
        pdz_binding=round(pdz_binding_proxy(seq), 4),
        details={"length": len(seq), "gravy": round(_gravy(seq), 3)},
    )


def score_batch(sequences: list[str]) -> list[ProxyScore]:
    return [score(s) for s in sequences]
