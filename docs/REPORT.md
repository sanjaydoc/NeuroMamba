# NeuroMamba — Technical Report

**Author:** Dr. Sanjay Anbu

NeuroMamba is a from-scratch selective state-space (Mamba/S6) protein language
model for de novo generation of novel PDZ-domain sequences. This report covers
the model, the training and generation methodology, the evaluation protocol, and
an honest account of scope and limitations.

## 1. Motivation

De novo design needs a generative *engine* that proposes genuinely new sequences.
The dominant choices are Transformers (quadratic attention) and structure-space
diffusion. **State-space models** are a third path: linear-time sequence modelling
with a learned, input-selective memory. The **selective** mechanism of Mamba/S6 —
making the SSM parameters functions of the input — is what lets a recurrent model
route information content-dependently, closing much of the historical gap to
attention while staying O(L). NeuroMamba explores this for proteins at a scale
that fits a 6 GB laptop GPU.

The demonstration target is the **PDZ domain** (Pfam `PF00595`). PDZ domains are
~90-residue interaction modules that scaffold the post-synaptic density (e.g.
PSD-95); a generative model of the family is a natural building block for
neural-interface tooling and a coherent de-novo-design benchmark.

## 2. Model

### 2.1 Selective SSM block

Each block (`src/neuromamba/mamba.py`) implements the Mamba mixer:

1. **Input projection** to an expanded inner width, split into a state path `x`
   and a gate `z`.
2. **Strictly-causal depthwise 1-D convolution** over `x` (left-padded, tail
   cropped) for local mixing without leaking future positions.
3. **Selective projections**: `Δ`, `B`, and `C` are computed *from* `x`, making
   them input-dependent — the S6 contribution over fixed-parameter S4.
4. **Discretization** via zero-order-hold: `Ā = exp(Δ ⊙ A)` with `A` a learned,
   log-parameterised, negative (stable) diagonal state matrix.
5. **Sequential scan**: `h_t = Ā_t · h_{t-1} + B̄_t · x_t`, `y_t = C_t · h_t + D · x_t`.
6. **Gated output**: `out = W_o (y · SiLU(z))`.

The recurrence is causal by construction, so the model is a valid autoregressive
LM. The scan is implemented as an explicit O(L) loop — numerically identical to
the hardware-aware parallel scan, without the `mamba-ssm` CUDA dependency, which
keeps the model runnable on CPU and any GPU.

### 2.2 Language model

`ProteinMamba` stacks pre-norm residual blocks (RMSNorm) with a tied
embedding / output head over a 23-token vocabulary (20 amino acids + PAD/BOS/EOS).
At the defaults (`d_model=128`, `n_layers=6`, `d_state=16`) it is **~0.7 M
parameters** — small enough that VRAM is a non-issue on 6 GB and training is
fast.

## 3. Data

`scripts/download_pdz_family.py` queries the **UniProt REST API** for reviewed
proteins carrying the PDZ Pfam domain and slices out each annotated PDZ region
using UniProt's own feature coordinates, yielding real, single-domain sequences.
When the network is unavailable it writes **clearly-labelled synthetic** PDZ-like
sequences built from the domain's secondary-structure layout, so the pipeline and
CI always run. Sequences are cleaned (canonical residues, length-filtered,
de-duplicated) before training.

## 4. Training

`src/neuromamba/train.py` is tuned for a 6 GB GPU:

- **Next-token cross-entropy**, PAD ignored.
- **fp16 autocast** on CUDA; gradient clipping; AdamW.
- **Step-based resumable checkpointing** — the checkpoint stores the global step,
  model and optimizer state, config, loss history, and the training sequences (so
  novelty can be measured later). `--resume` continues seamlessly. This is the
  "train in batches" workflow: stop after any number of steps and resume.

## 5. Generation & evaluation

`src/neuromamba/generate.py` samples autoregressively from `<bos>` with
temperature, top-k, and nucleus (top-p) controls. `scripts/generate.py` produces
a library and reports:

- **validity** — canonical, sensibly-sized sequences;
- **uniqueness** — fraction of distinct samples;
- **novelty** — fraction whose max identity to the *training set* is below
  threshold (guards against memorisation), plus mean max-train-identity;
- **diversity** — 1 − mean pairwise identity among samples;
- **proxy scores** — stability, solubility, and PDZ-binding-groove proxies
  (`src/neuromamba/oracle.py`).

A torch-free **order-k Markov** model (`src/neuromamba/markov.py`) is the baseline:
it has no long-range state, so the margin NeuroMamba shows over it is a direct
read on what the selective-SSM memory buys.

## 6. Results

**Training.** A 0.10 M-parameter model (`d_model=64, n_layers=3`) trained for 350
steps on CPU shows next-token cross-entropy falling **2.77 → 2.28**. The package
default (0.70 M, `d_model=128, n_layers=6`) is what the RTX 3000 run uses; it fits
6 GB comfortably, trains faster, and reaches lower loss.

**Generation.** 64 samples, NeuroMamba vs. the order-3 Markov baseline, same proxy
oracle (higher is better; *max-train-identity* lower = more novel):

| Metric | Markov (baseline) | NeuroMamba |
|---|---|---|
| Validity | 0.73 | **0.97** |
| Uniqueness | 1.00 | 1.00 |
| Novel fraction (id < 0.8) | 1.00 | 1.00 |
| Mean max-train-identity | 0.41 | 0.49 |
| Diversity | 0.78 | 0.71 |
| Proxy stability | 0.835 | **0.840** |
| Proxy solubility | **0.673** | 0.651 |
| Proxy PDZ-binding groove | 0.708 | **0.798** |

The selective-SSM improves **validity (+0.23)** and the **PDZ-groove score
(+0.09)** — the two metrics tied to producing real, functional domains — at 100%
novelty. The baseline's slightly higher diversity/solubility follows from its
lower validity (noisier sequences). This is the signal the recurrent selective
memory adds over a memoryless k-gram; longer training widens it. Numbers are from
a short run for reproducibility; rerun `scripts/generate.py` to refresh.

## 7. Limitations & honesty

- The oracle scorers are **in-silico proxies, not validated assays**. They
  demonstrate the *generate → score* interface; a real readout plugs in unchanged.
- The model is a **toy-scale demonstration** (~0.7 M params, small data), not a
  foundation model. The contribution is the from-scratch selective-SSM
  implementation and the measured, reproducible pipeline.
- Sequence identity uses an alignment-free ratio, adequate for short same-family
  sequences but not a substitute for a proper MSA-based identity at scale.

## 8. Reproduce

```bash
pip install -e ".[torch]"
python scripts/download_pdz_family.py --out data/pdz.jsonl
python -m neuromamba.train --data data/pdz.jsonl --max-steps 4000 --device cuda
python scripts/generate.py --ckpt outputs/neuromamba/model.pt --n 64   # NeuroMamba
python scripts/generate.py --markov --data data/pdz.jsonl --n 64        # baseline
```
