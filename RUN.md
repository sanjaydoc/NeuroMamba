# RUN.md — setup & run commands

Verified end-to-end on the RTX 3000 dev machine (Windows). Most of this page is
Windows `cmd`; the **macOS / Linux equivalents** are in the box just below. Two
rules that make everything work on Windows:

1. **Use Python 3.10**, not 3.13/3.14 — PyTorch has no CUDA wheels for those.
2. **Always call pip as `python -m pip ...`**, never bare `pip ...` — some
   Windows (Device Guard / WDAC) policies block the generated `pip.exe` shim, but
   running it through `python.exe` is allowed.

Every new terminal, reactivate the venv first:

```bat
.venv\Scripts\activate.bat
```

### macOS / Linux equivalents

```bash
# setup (Python 3.10–3.12)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"                 # Mac/CPU; CUDA box: pip install torch --index-url https://download.pytorch.org/whl/cu121

# run (forward slashes; --device mps on Apple Silicon, cuda on Linux+NVIDIA, else cpu)
python scripts/download_pdz_family.py --out data/pdz.jsonl
python -m neuromamba.train --data data/pdz.jsonl --max-steps 4000 --batch-size 64 --device auto
python scripts/generate.py --ckpt outputs/neuromamba/model.pt --n 64
```

The only cross-platform differences are venv activation (`source .venv/bin/activate`
vs `.venv\Scripts\activate.bat`) and path separators (`/` vs `\`). `--device auto`
picks CUDA → MPS → CPU automatically.

---

## 0. Clone (one-time)

```bat
cd "C:\Users\<you>\Desktop\All Apps"
git clone https://github.com/sanjaydoc/NeuroMamba.git
cd NeuroMamba
```

## 1. Virtual environment (Python 3.10)

```bat
py -0                                  :: list installed Pythons; need a 3.10
py -3.10 -m venv .venv
.venv\Scripts\activate.bat
python --version                       :: must print 3.10.x
```

If you don't have 3.10: `winget install Python.Python.3.10`
(PowerShell instead of cmd? activate with `.venv\Scripts\Activate.ps1`.)

## 2. Install (GPU)

```bat
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e ".[dev]"
```

`cu121` is correct for the RTX 3000 (Turing). No GPU? Use
`python -m pip install -e ".[torch]"` for a CPU build instead — everything still
runs, just slower.

## 3. Verify

```bat
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -m pytest -q
```

Expect `...+cu121 CUDA: True` and all tests passing.

## 4. Data → Train → Generate

```bat
:: real PDZ-domain sequences (UniProt); synthetic fallback if offline
python scripts\download_pdz_family.py --out data\pdz.jsonl

:: train on the GPU. --batch-size 64 fits 6 GB comfortably with the parallel
:: scan + gradient checkpointing (on by default). Raise it until VRAM is ~80%;
:: lower it if you hit "CUDA out of memory".
python -m neuromamba.train --data data\pdz.jsonl --max-steps 4000 --batch-size 64 --device cuda

:: generate + score novel sequences
python scripts\generate.py --ckpt outputs\neuromamba\model.pt --n 64
```

> **VRAM note.** The parallel scan is fast but activation-heavy (it holds
> ~log2(L) copies of the state). Gradient checkpointing is **on by default**,
> which recomputes each block in the backward pass and bounds peak VRAM to a
> single block — that's what lets batch 64 fit 6 GB. If you still hit OOM, lower
> `--batch-size` (32, 16) and/or set, before the command:
> `set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### Train in batches (resume any time)

Stop training with `Ctrl-C` whenever; continue from the last checkpoint with a
higher `--max-steps` and `--resume`:

```bat
python -m neuromamba.train --data data\pdz.jsonl --max-steps 8000 --device cuda --resume
```

### Torch-free baseline (no GPU / no checkpoint needed)

```bat
python scripts\generate.py --markov --data data\pdz.jsonl --n 64
```

## Full command / flag reference

### `download_pdz_family.py` — build the dataset
```bat
python scripts\download_pdz_family.py --out data\pdz.jsonl --count 400
python scripts\download_pdz_family.py --synthetic              :: skip the network
python scripts\download_pdz_family.py --min-len 40 --max-len 140
```

### `neuromamba.train` — train / resume
```bat
python -m neuromamba.train --data data\pdz.jsonl --max-steps 4000 --batch-size 64 --device cuda
```
| Flag | Default | Meaning |
|---|---|---|
| `--data` | (synthetic) | JSONL of `{"sequence": ...}`; omit or `--synthetic` for synthetic data |
| `--max-steps` | 2000 | optimizer-step budget (early stopping may end sooner) |
| `--batch-size` | 16 | bigger = more VRAM + higher GPU use; drop on OOM |
| `--lr` | 3e-4 | learning rate |
| `--device` | auto | `auto` / `cuda` / `mps` / `cpu` |
| `--resume` | off | continue from `last.pt` (step + optimizer restored) |
| `--val-frac` | 0.1 | held-out validation fraction |
| `--eval-every` | 100 | steps between validation checks |
| `--patience` | 8 | early-stop after N evals with no val improvement |
| `--no-early-stop` | off | disable early stopping (train the full `--max-steps`) |
| `--d-model` / `--n-layers` / `--d-state` | 128 / 6 / 16 | model size (scale up for more GPU work / capacity) |
| `--save-every` | 200 | checkpoint cadence when there's no validation set |
| `--no-amp` | off | disable fp16 mixed precision |
| `--seed` | 42 | RNG seed |

### `generate.py` — sample + score
```bat
python scripts\generate.py --ckpt outputs\neuromamba\model.pt --n 64
python scripts\generate.py --ckpt outputs\neuromamba\model.pt --n 64 --temperature 1.2 --top-p 0.95
python scripts\generate.py --ckpt outputs\neuromamba\model.pt --n 64 --top-k 10
python scripts\generate.py --markov --data data\pdz.jsonl --n 64 --markov-order 3
```
| Flag | Default | Meaning |
|---|---|---|
| `--ckpt` | outputs\neuromamba\model.pt | checkpoint to sample from |
| `--n` | 64 | number of sequences |
| `--max-len` | 130 | max residues per sequence |
| `--temperature` | 1.0 | >1 = more diverse/novel, <1 = more conservative |
| `--top-p` | 0.9 | nucleus sampling threshold |
| `--top-k` | 0 | restrict to k most-likely residues (0 = off) |
| `--data` | — | training JSONL, for the novelty comparison |
| `--markov` / `--markov-order` | off / 3 | torch-free baseline generator |

Outputs land in `outputs\generated\`: `designs.jsonl` (sequences + proxy scores)
and `metrics.json` (validity / novelty / diversity + means).

## Checkpoints: `model.pt` vs `last.pt`

Training writes two files in `outputs\neuromamba\`:

* **`model.pt`** — the **best** model by validation loss. This is what
  `generate.py` loads. It is *not* necessarily the last step — it's the one that
  generalized best.
* **`last.pt`** — the **latest** step (weights + optimizer + step). `--resume`
  continues from here.

## Avoiding overfitting (validation + early stopping)

On a small dataset this model will memorize if trained too long (you'll see the
loss dive toward ~0.1 and generated sequences become near-copies of the training
set — `novel_fraction` near 0). To prevent that, training holds out a validation
split, reports `val_loss` / `val_ppl` every `--eval-every` steps, saves the
best-val model to `model.pt`, and **early-stops** when val loss stops improving:

```bat
:: let early stopping find the sweet spot (big budget, it stops on its own)
python -m neuromamba.train --data data\pdz.jsonl --max-steps 8000 --batch-size 64 --patience 8 --device cuda

:: if generations still look too close to training data, sample hotter:
python scripts\generate.py --ckpt outputs\neuromamba\model.pt --n 64 --temperature 1.4 --top-p 0.98
```

## Making the GPU work harder (higher utilization / faster)

Low GPU utilization (e.g. ~20%) means the GPU is idle waiting between small
operations. Levers, most effective first:

1. **The parallel scan** (already in the model) replaces the old per-timestep
   loop with a work-efficient prefix scan, so the GPU runs big batched ops
   instead of thousands of tiny sequential ones. This is the main fix — measured
   ~2× faster even on CPU, more on GPU. Make sure you're on the latest code:
   `git pull origin main`.

2. **Bigger batch** — larger batches = larger kernels = higher utilization.
   Push it until VRAM is ~80% full; back off if you hit `CUDA out of memory`
   (gradient checkpointing is on by default, so batch 64 fits 6 GB):
   ```bat
   python -m neuromamba.train --data data\pdz.jsonl --max-steps 4000 --batch-size 64 --device cuda
   ```

3. **Bigger model** — more width/depth = more work per step (also raises
   capacity). Scale with the CLI knobs:
   ```bat
   python -m neuromamba.train --data data\pdz.jsonl --max-steps 4000 ^
       --batch-size 96 --d-model 256 --n-layers 8 --d-state 16 --device cuda
   ```

Watch VRAM + utilization live in another terminal:
```bat
nvidia-smi -l 1
```

Note: a 0.7 M-param model is *small* — even fully optimized it won't saturate a
modern GPU, and that's fine. What matters is wall-clock per step, which the
parallel scan + a larger batch cut substantially. If you want to genuinely load
the GPU, raise `--d-model`/`--n-layers` and `--batch-size` together.

## 5. Pull future updates

From inside the `NeuroMamba` folder:

```bat
git pull origin main
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No matching distribution found for torch` | venv is Python 3.13/3.14 | Rebuild the venv with `py -3.10 -m venv .venv` |
| `pip.exe ... blocked by your organization's Device Guard policy` | policy blocks the pip shim | Use `python -m pip ...` instead of `pip ...` |
| `CUDA: False` | CPU torch installed, or driver mismatch | Reinstall torch from the `cu121` index in a 3.10 venv |
