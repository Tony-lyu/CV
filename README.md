# Week 1 — CV PEFT Hybridization (LoRA + NormTune)

## TL;DR — Weekly Plan
1) **Set up environment** (Conda or venv) and run a quick smoke‑test training on CIFAR‑10 using a ViT backbone with:
   - **LoRA** (linear layers adapted)
   - **NormTune** (train only LayerNorm γ/β)
   - **Hybrid** (LoRA + NormTune)
2) **Record metrics**: Top‑1 accuracy, **params‑tuned %**, wall‑clock/epoch, and **ECE** (calibration).  
3) **Log results** to `reports/results_week1.csv` and jot brief notes in `reports/week1_kickoff.md`.

> Hardware: run on Mac Intel i7 CPU.

## environment
```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
# LoRA 
python -m src.train --config src/config/cifar10_lora.yaml

# NormTune
python -m src.train --config src/config/cifar10_norm.yaml

# Hybrid
python -m src.train --config src/config/cifar10_hybrid.yaml
```

This will append rows to `reports/results_week1.csv` like:

| ts | method | dataset | backbone | tuned_params_% | acc@1 | ece | train_time_s | epochs |
|----|--------|---------|----------|----------------|-------|-----|--------------|--------|

## Datasets
- CIFAR‑10 will **auto‑download** to `~/.torch`. No extra steps.

## Design Overview
- **Backbone**: `timm` ViT (`vit_tiny_patch16_224`, ImageNet‑1k pretrained).
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
- `reports/week1_kickoff.md` — notes & takeaway.
- `reports/results_week1.csv` — Append‑only results log.
- `scripts/run_experiment.sh` — Reproduce Week 1 table.


*Created: 2025-08-25*
