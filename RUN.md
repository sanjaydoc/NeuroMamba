# RUN.md — setup & run commands (Windows, verified)

Verified end-to-end on the RTX 3000 dev machine (Windows). Two rules that make
everything work on this setup:

1. **Use Python 3.10**, not 3.13/3.14 — PyTorch has no CUDA wheels for those.
2. **Always call pip as `python -m pip ...`**, never bare `pip ...` — some
   Windows (Device Guard / WDAC) policies block the generated `pip.exe` shim, but
   running it through `python.exe` is allowed.

Every new terminal, reactivate the venv first:

```bat
.venv\Scripts\activate.bat
```

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

:: train on the GPU (small model, fits 6 GB VRAM)
python -m neuromamba.train --data data\pdz.jsonl --max-steps 4000 --batch-size 128 --device cuda

:: generate + score novel sequences
python scripts\generate.py --ckpt outputs\neuromamba\model.pt --n 64
```

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

## Making the GPU work harder (higher utilization / faster)

Low GPU utilization (e.g. ~20%) means the GPU is idle waiting between small
operations. Levers, most effective first:

1. **The parallel scan** (already in the model) replaces the old per-timestep
   loop with a work-efficient prefix scan, so the GPU runs big batched ops
   instead of thousands of tiny sequential ones. This is the main fix — measured
   ~2× faster even on CPU, more on GPU. Make sure you're on the latest code:
   `git pull origin main`.

2. **Bigger batch** — larger batches = larger kernels = higher utilization.
   Push it until VRAM is ~80% full; back off if you hit `CUDA out of memory`:
   ```bat
   python -m neuromamba.train --data data\pdz.jsonl --max-steps 4000 --batch-size 128 --device cuda
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
