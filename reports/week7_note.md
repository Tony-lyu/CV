# Week 7

## Setup

- **Model / data:** `vit_small_patch16_224` on **CIFAR-100**, img size 224, batch 128.
- **Pretraining:** **None** this week (consistent within week).
- **Optim:** AdamW + cosine (η_min=1e-5), epochs = 5 primary; **20-epoch confirmations** at 0.2% / 0.5% / 1.0%.
- **Methods compared:** `norm` (Norm/EFFT-only), `lora` (LoRA-only), `hybrid_layerwise` (naïve mix), `hybrid_layers` (layer-aware/disjoint mix).
- **Budgets:** target ~**0.2%**, **0.5%**, **1.0%** trainable; actual plotted % from `trainable_params / total_params`.
- **Diagnostics:** per-epoch ECE, step time, peak mem; grad/updates and cosines (as in prior weeks).
- **Evaluation:** `src/week7/eval.py` updated to treat **each CSV as one run** → all budgets appear; **Pareto** (Acc vs Trainable-%) + **EdgeScore** bars (averaged over seeds by filename).

---

## What changed vs Week-6

- **No `--pretrained`:** absolute numbers drop, but internal **ordering** is consistent across seeds and epochs.
- **Eval fix:** no collapse across budgets; **all** 0.2/0.5/1.0 points show on the Pareto chart.
- **Long-epoch check:** 20-epoch slices at all budgets to test training-time sensitivity (ordering held).

---

## Results (qualitative from plots; ↑ better)

### Pareto: Top-1 Acc vs Trainable-%
- At **tiny budgets** (≈0.02–0.08% trainable), **hybrid points—especially `hybrid_layers`—form the upper envelope**, i.e., **more accuracy per parameter** than either `lora` or `norm`.
- **`norm`** sits far left (fewest params) with the **lowest accuracy** (best raw ECE).
- **`lora`** sweeps upward with more params; **at large budgets** (≈0.25–0.27%), `lora` reaches the **highest absolute accuracy** but is **off the edge-device Pareto** due to cost.
- **`hybrid_layerwise`** improves over `norm` but is **below `hybrid_layers`**, highlighting the importance of **where** adapters are placed.

### EdgeScore (Acc − λ·ECE − λ·% − λ·time − λ·mem)
- At **matched small budgets**, **hybrids** (esp. `hybrid_layers`) **beat `norm`** and often **beat `lora`** on EdgeScore, reflecting better edge trade-offs.
- At **large budgets**, `lora` can top raw accuracy, but the EdgeScore advantage shifts toward smaller, hybrid or norm configurations depending on λs.

### Training-time sensitivity
- Extending to **20 epochs** preserves the **hybrid-above-LoRA** pattern at tiny budgets and **LoRA-above-all** at very large budgets. The **relative ordering is stable** across **3 seeds**.

---

## Answers to Research Questions

### RQ1 — Hybrid Effectiveness
**Answer:** **Yes.** Under tiny adapter budgets on ViT-S/CIFAR-100, **layer-aware hybrids (`hybrid_layers`) expand the Acc–Params Pareto frontier**, delivering **higher accuracy per trainable-%** than **LoRA-only** or **Norm/EFFT-only**.  
**Nuance:** At much larger budgets, **LoRA-only** achieves the **highest absolute accuracy**, but with a substantially **higher parameter cost**—less suitable for edge targets.

### RQ2 — Layer Sensitivity
**Answer:** **Yes.** Results support clear **layer sensitivity**:
- **LoRA** contributes most on **projection-heavy** components (attention projections, later blocks).
- **Norm/EFFT** is most effective on **normalization** parameters.
- **Layer-aware/disjoint mixing** (our `hybrid_layers` policy) **avoids interference** seen in naïve hybrids and yields **additive gains**, explaining why `hybrid_layers` dominates `hybrid_layerwise` at the same budget.

---

## Observations

- **Calibration:** `norm` retains the best raw ECE; hybrids do **not** show a notable ECE penalty vs `lora` at tiny budgets. Post-hoc temperature scaling can close the remaining gap when needed.
- **Speed / memory:** In line with prior weeks—`norm` fastest/leanest; hybrids close to `norm`; `lora` heavier as rank/budget grows.
- **Reproducibility:** Patterns hold across **3 seeds** and **5→20 epochs**.

---

## Reproducibility & Artifacts

- **Commands:** `src/week7/commands.md` (batch with `src/week7/run_all.py`).
- **Eval:**  
  ```bash
  python src/week7/eval.py 
    --logs [path/path/..]
    --total_params [integer] 
    --out_dir [path/path/..] 
    --lam [float(s)]
    --filter_dataset [String]
    --filter_model [String]
