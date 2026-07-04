"""Torch-free tests: tokenizer, data, metrics, proxy oracle, Markov baseline.

These run on a bare machine (no PyTorch), mirroring CI's lightweight lane.
"""
from __future__ import annotations

from neuromamba import oracle
from neuromamba.data import (
    clean_sequences,
    load_sequences_jsonl,
    save_sequences_jsonl,
    synthetic_pdz_sequences,
)
from neuromamba.markov import MarkovGenerator
from neuromamba.metrics import evaluate, identity, is_valid, max_identity_to_set
from neuromamba.tokenizer import AA_SET, ProteinTokenizer


# --- tokenizer --------------------------------------------------------------
def test_tokenizer_roundtrip():
    tok = ProteinTokenizer()
    seq = "MKTAYIAKQR"
    ids = tok.encode(seq)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    assert tok.decode(ids) == seq


def test_tokenizer_skips_noncanonical():
    tok = ProteinTokenizer()
    # X and U are not canonical; they should be dropped, not crash.
    assert tok.decode(tok.encode("MKXUAY")) == "MKAY"


def test_tokenizer_vocab_size():
    tok = ProteinTokenizer()
    assert tok.vocab_size == 23  # 20 AAs + pad/bos/eos


# --- data -------------------------------------------------------------------
def test_synthetic_sequences_are_canonical():
    seqs = synthetic_pdz_sequences(20, seed=1)
    assert len(seqs) == 20
    for s in seqs:
        assert set(s) <= AA_SET
        assert 40 <= len(s) <= 140


def test_clean_sequences_filters_and_dedups():
    raw = ["MKTAY", "MKTAY", "mktz9ay", "A" * 5, "A" * 200]
    cleaned = clean_sequences(raw, min_len=4, max_len=50)
    assert "MKTAY" in cleaned
    assert cleaned.count("MKTAY") == 1  # dedup
    assert all(len(s) <= 50 for s in cleaned)


def test_jsonl_roundtrip(tmp_path):
    seqs = ["MKTAY", "GLGFSIL"]
    path = save_sequences_jsonl(seqs, tmp_path / "seqs.jsonl")
    assert load_sequences_jsonl(path) == seqs


# --- metrics ----------------------------------------------------------------
def test_identity_bounds():
    assert identity("ABC", "ABC") == 1.0
    assert identity("ABC", "XYZ") == 0.0
    assert 0.0 < identity("ABCDE", "ABXDE") < 1.0


def test_is_valid():
    assert is_valid("A" * 30)
    assert not is_valid("A" * 5)          # too short
    assert not is_valid("ABC123")         # non-canonical


def test_novelty_detects_memorisation():
    train = ["MKTAYIAKQRQ", "GLGFSILGGED"]
    # An exact copy has max identity 1.0 (not novel).
    assert max_identity_to_set("MKTAYIAKQRQ", train) == 1.0


def test_evaluate_summary_keys():
    train = synthetic_pdz_sequences(30, seed=2)
    samples = synthetic_pdz_sequences(20, seed=99)
    m = evaluate(samples, training=train)
    for key in ("validity", "uniqueness", "diversity", "novel_fraction"):
        assert key in m
    assert 0.0 <= m["validity"] <= 1.0


# --- proxy oracle -----------------------------------------------------------
def test_oracle_scores_in_range():
    sc = oracle.score("GLGFSILGGEDKPGDVILAEDLRKALEQ")
    for v in sc.as_vector():
        assert 0.0 <= v <= 1.0
    assert "length" in sc.details


def test_oracle_rewards_glgf_motif():
    with_motif = oracle.pdz_binding_proxy("AAAGLGFAAA")
    without = oracle.pdz_binding_proxy("AAAAAAAAAA")
    assert with_motif > without


def test_oracle_empty_sequence():
    sc = oracle.score("")
    assert sc.mean == 0.0


# --- Markov baseline --------------------------------------------------------
def test_markov_generates_valid_sequences():
    train = synthetic_pdz_sequences(50, seed=3)
    gen = MarkovGenerator(order=3).fit(train)
    samples = gen.sample(10, max_len=140, seed=7)
    assert len(samples) == 10
    for s in samples:
        assert set(s) <= AA_SET


def test_markov_requires_fit():
    import pytest

    with pytest.raises(RuntimeError):
        MarkovGenerator().sample(1)
