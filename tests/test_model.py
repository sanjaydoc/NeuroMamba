"""PyTorch tests for the Mamba model, training, checkpoint resume, and sampling.

Skipped cleanly when PyTorch is absent, so the torch-free core suite still runs.
The load-bearing test is **causality**: changing a token must never alter the
logits at earlier positions — the property that makes autoregressive generation
and next-token training valid at all.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from neuromamba.mamba import ProteinMamba, build_mamba  # noqa: E402
from neuromamba.tokenizer import ProteinTokenizer  # noqa: E402


def _small_model(seed: int = 0) -> ProteinMamba:
    torch.manual_seed(seed)
    tok = ProteinTokenizer()
    return build_mamba(
        {"d_model": 32, "n_layers": 2, "d_state": 8, "d_conv": 4, "expand": 2},
        tok.vocab_size,
        tok.pad_id,
    )


def test_forward_shape():
    model = _small_model()
    tokens = torch.randint(3, 23, (2, 17))
    logits = model(tokens)
    assert logits.shape == (2, 17, model.vocab_size)


def test_causality():
    """Logits at position t must not depend on tokens after t."""
    model = _small_model().eval()
    tokens = torch.randint(3, 23, (1, 12))
    with torch.no_grad():
        base = model(tokens)
        changed = tokens.clone()
        changed[0, -1] = (changed[0, -1] + 1)  # flip the LAST token
        after = model(changed)
    # everything before the last position is unchanged
    assert torch.allclose(base[:, :-1], after[:, :-1], atol=1e-5)
    # the last position DID change (sanity: the model actually uses the input)
    assert not torch.allclose(base[:, -1], after[:, -1], atol=1e-5)


def test_weight_tying_and_param_count():
    model = _small_model()
    assert model.lm_head.weight is model.embedding.weight
    assert model.num_params() > 0


def test_loss_decreases_on_overfit():
    """A few steps of AdamW on one batch should reduce the loss."""
    from neuromamba.train import _cross_entropy

    model = _small_model().train()
    tok = ProteinTokenizer()
    batch = torch.stack(
        [torch.tensor(tok.encode("GLGFSILGGEDKPGDVILAEDLRKALEQ")) for _ in range(4)]
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = None
    for _ in range(25):
        opt.zero_grad()
        loss = _cross_entropy(model(batch), batch, tok.pad_id)
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss.item())
    assert float(loss.item()) < first


def test_sampling_produces_valid_sequences():
    from neuromamba.generate import sample

    model = _small_model().eval()
    tok = ProteinTokenizer()
    seqs = sample(model, tok, n_samples=5, max_len=40, top_p=0.9, seed=1)
    assert len(seqs) == 5
    for s in seqs:
        assert all(c in tok.stoi for c in s)  # only decodable residues


def test_train_and_resume(tmp_path):
    """End-to-end: train a few steps, then resume and confirm the step advances."""
    from neuromamba.train import train

    out = tmp_path / "run"
    cfg = {"d_model": 32, "n_layers": 2, "d_state": 8}
    ckpt = train(
        synthetic=True, out_dir=out, max_steps=10, batch_size=8,
        save_every=5, log_every=100, device="cpu", model_cfg=cfg, seed=0,
    )
    assert ckpt.exists()
    state = torch.load(ckpt, map_location="cpu")
    assert state["global_step"] == 10
    assert "train_sequences" in state and state["train_sequences"]

    # Resume for 10 more steps -> global step should reach 20.
    train(
        synthetic=True, out_dir=out, max_steps=20, batch_size=8,
        save_every=5, log_every=100, device="cpu", model_cfg=cfg, seed=0,
        resume=True,
    )
    state2 = torch.load(ckpt, map_location="cpu")
    assert state2["global_step"] == 20
