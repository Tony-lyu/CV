# Week 1 Starter — CV PEFT Hybridization (LoRA + NormTune)

This repo bootstraps **Week 1** of your CV project exploring whether hybridizing PEFT methods (LoRA + normalization‑only tuning) yields additive gains under constrained compute.

## TL;DR — Weekly Plan
1) **Set up environment** (Conda or venv) and run a quick smoke‑test training on CIFAR‑10 using a ViT backbone with:
   - **LoRA** (linear layers adapted)
   - **NormTune** (train only LayerNorm γ/β)
   - **Hybrid** (LoRA + NormTune)
2) **Record metrics**: Top‑1 accuracy, **params‑tuned %**, wall‑clock/epoch, and **ECE** (calibration).  
3) **Log results** to `reports/results_week1.csv` and jot brief notes in `reports/week1_kickoff.md`.

> Hardware: Should run on CPU/MPS/ CUDA. CIFAR‑10 keeps runs short on a Mac. Expect ~3–10 min/epoch on CPU; much faster with MPS/CUDA.

## Install
```bash
# Option A: Conda (recommended)
conda env create -f env.yml
conda activate cv-peft

# Option B: venv + pip
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart (single run)
```bash
# LoRA on CIFAR‑10
python -m src.train --config src/config/cifar10_lora.yaml

# NormTune
python -m src.train --config src/config/cifar10_norm.yaml

# Hybrid
python -m src.train --config src/config/cifar10_hybrid.yaml
```

## Reproduce Week 1 table
```bash
bash scripts/run_experiment.sh
```

This will append rows to `reports/results_week1.csv` like:

| ts | method | dataset | backbone | tuned_params_% | acc@1 | ece | train_time_s | epochs |
|----|--------|---------|----------|----------------|-------|-----|--------------|--------|

## Datasets
- CIFAR‑10 will **auto‑download** to `~/.torch`. No extra steps.

## Design Overview
- **Backbone**: `timm` ViT (`vit_base_patch16_224`, ImageNet‑1k pretrained).
- **LoRA**: Replace `nn.Linear` in attention/MLP blocks with `LoRALinear` and freeze base weights.
- **NormTune**: Freeze all weights **except** `LayerNorm.weight`/`.bias`.
- **Hybrid**: LoRA + NormTune; classifier head stays trainable by default (configurable).

## Metrics
- **Top‑1** on CIFAR‑10 test set.
- **ECE** (expected calibration error, 15 bins).
- **Params‑tuned %** (trainable/total params × 100).
- **Time/epoch** for budget awareness.

## Files
- `src/train.py` — Training loop driven by YAML config.
- `src/vit_model.py` — Backbone factory + freezing logic.
- `src/lora.py` — Minimal, framework‑free LoRA for `nn.Linear`.
- `src/datasets.py` — CIFAR‑10 loaders & transforms.
- `src/utils.py` — Metrics (accuracy/ECE), counters, helpers.
- `src/config/*.yaml` — Ready configs for LoRA / Norm / Hybrid.
- `reports/week1_kickoff.md` — Your notes & takeaway.
- `reports/results_week1.csv` — Append‑only results log.
- `scripts/run_experiment.sh` — Reproduce Week 1 table.

## Next (Week 2 preview)
- Expand to **ImageNet‑100** or a VTAB subset.
- Add ablations (LoRA rank/dropout/targets), stability across seeds.
- Pull or re‑implement the exact **EFFT** variant from the paper repo and slot it into the same API alongside NormTune.

---

*Created: 2025-08-25*
