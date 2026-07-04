"""Fetch real PDZ-domain sequences to train NeuroMamba.

Primary source — the **UniProt REST API**: query reviewed entries carrying the
PDZ domain (Pfam ``PF00595`` / InterPro ``IPR001478``) and slice out each
annotated PDZ domain region using UniProt's own feature coordinates. This yields
genuine, single-domain, ~80–100-residue sequences — the sparse biological data
the model learns from.

Fallback — if the network is unavailable, write the package's clearly-labelled
**synthetic** PDZ-like sequences instead, so the pipeline and CI still run. The
output is always a JSONL of ``{"sequence": ...}`` records.

Usage::

    python scripts/download_pdz_family.py                       # ~400 domains
    python scripts/download_pdz_family.py --count 800 --out data/pdz.jsonl
    python scripts/download_pdz_family.py --synthetic           # force offline
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuromamba.data import (  # noqa: E402
    clean_sequences,
    save_sequences_jsonl,
    synthetic_pdz_sequences,
)
from neuromamba.utils import setup_logging  # noqa: E402

LOGGER = setup_logging()

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"


def fetch_pdz_domains(count: int, timeout: int = 60) -> list[str]:
    """Query UniProt and extract PDZ-domain subsequences via feature coordinates."""
    params = {
        "query": "xref:pfam-PF00595 AND reviewed:true",
        "fields": "accession,sequence,ft_domain",
        "format": "json",
        "size": str(min(count, 500)),
    }
    url = UNIPROT_URL + "?" + urllib.parse.urlencode(params)
    LOGGER.info("Querying UniProt for PDZ-domain proteins ...")
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        payload = json.loads(resp.read().decode())

    domains: list[str] = []
    for entry in payload.get("results", []):
        seq = entry.get("sequence", {}).get("value", "")
        if not seq:
            continue
        for feat in entry.get("features", []):
            if feat.get("type") != "Domain":
                continue
            if "PDZ" not in feat.get("description", "").upper():
                continue
            loc = feat.get("location", {})
            start = loc.get("start", {}).get("value")
            end = loc.get("end", {}).get("value")
            if isinstance(start, int) and isinstance(end, int) and end > start:
                domains.append(seq[start - 1 : end])  # UniProt is 1-indexed
    LOGGER.info("Extracted %d PDZ-domain regions from UniProt.", len(domains))
    return domains


def main() -> int:
    p = argparse.ArgumentParser(description="Download PDZ-domain sequences.")
    p.add_argument("--count", type=int, default=400)
    p.add_argument("--out", default="data/pdz.jsonl")
    p.add_argument("--min-len", type=int, default=40)
    p.add_argument("--max-len", type=int, default=140)
    p.add_argument("--synthetic", action="store_true", help="Skip the network; use synthetic data.")
    args = p.parse_args()

    seqs: list[str] = []
    if not args.synthetic:
        try:
            seqs = fetch_pdz_domains(args.count)
        except Exception as exc:  # noqa: BLE001 - network is best-effort
            LOGGER.warning("UniProt fetch failed (%s); using synthetic fallback.", exc)

    if not seqs:
        LOGGER.warning("Writing SYNTHETIC PDZ-like sequences (clearly labelled).")
        seqs = synthetic_pdz_sequences(max(args.count, 512))

    seqs = clean_sequences(seqs, min_len=args.min_len, max_len=args.max_len)
    out = save_sequences_jsonl(seqs, args.out)
    LOGGER.info("Wrote %d sequences -> %s", len(seqs), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
