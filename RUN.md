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
