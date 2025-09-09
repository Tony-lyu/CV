# Week 3: Scaling Experiments (ViT-Small)

## Why this folder is empty
Week 3 evaluates **scaling behavior** of PEFT methods (LoRA / Norm / Hybrid) on **larger backbones**.  
There are **no new algorithmic changes**—I reuse Week 2 training code and only change run arguments (backbone, epochs, etc.).  
Keeping `src/week3/` minimal avoids code drift and makes it clear that any deltas in results come from **model size**, not code edits.

- Code used: `src/week2/train.py` (imported as a module)
- This folder only contains **docs/instructions** for reproducibility.

---

## Goals (Week 3)
1. **Scale up** from ViT-Tiny → **ViT-Small**.
2. Check if Week 1/2 trends hold at larger scale:
   - Does **NormTune** remain parameter-efficient?
   - Does **Hybrid** show **additivity** or **interference**?
   - How do **ECE** and **training time/epoch** evolve with model size?
3. Produce **plots**:
   - Accuracy vs Epochs (for ViT-Small)
   - **% Parameters Tuned** vs **Accuracy@1** across Tiny/Small/(Base)

---

## Datasets
- Primary: **CIFAR-100** (this week’s main comparison)
- Optional cross-check: **CIFAR-10** (only if time permits)

---

## Reproducible Run Commands

> We continue to call the Week 2 trainer to avoid code duplication.

### ViT-Small
**Hybrid**
```bash
python -m src.week2.train --dataset cifar100 --backbone vit_small_patch16_224 --method hybrid --epochs 10 --batch-size 128 --num-workers 4 --lr 0.001 --lora-lr 0.0005 --lora-r 8 --lora-alpha 16 --lora-drop 0.0 --out-csv reports/results_week3.csv

```
**LoRA**
```bash
python -m src.week2.train --dataset cifar100 --backbone vit_small_patch16_224 --method lora --epochs 10 --batch-size 128 --num-workers 4 --lr 0.001 --lora-r 8 --lora-alpha 16 --lora-drop 0.0 --out-csv reports/results_week3.csv
```

**Norm**
```bash
python -m src.week2.train --dataset cifar100 --backbone vit_small_patch16_224 --method norm --epochs 10 --batch-size 128 --num-workers 4 --lr 0.001 --out-csv reports/results_week3.csv
```