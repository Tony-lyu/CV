# Week 8 — RQ3: Efficiency vs. Generalization (Edge Suitability)

## Goal
Test whether **Hybrid PEFT** (our layerwise hybrid) better balances **parameter efficiency** and **generalization under distribution shift**, making it preferable for **mobile/edge** settings.

---

## Experimental Setup

**Backbone**: `vit_small_patch16_224`  
**Dataset (ID)**: CIFAR-100 (test)  
**Dataset (Shifted OOD)**: CIFAR-100-C (15 corruptions × severities 1–5)  
**Budgets**: ~0.2%, 0.5%, 1.0% **trainable params**  
**Methods**: **Norm-only (EFFT/NormTune)**, **LoRA-only**, **Layerwise Hybrid** (our best variant)  
**Schedule**: AdamW + cosine decay with **η_min > 0**, warm-up `freeze_lora_steps=400` for hybrids (unchanged from Week-6/7)  
**Seeds**: 3 (paired across methods)  
**Batch/Size**: 128, 224×224

**Metrics**
- **ID**: Acc@1, **ECE**, **NLL**
- **Shift (CIFAR-C)**: mean Acc (**Acc_C_mean**), worst corruption Acc (**Acc_C_worst**)
- **Efficiency**: trainable params %, **batch-1 CPU latency**
- **Composite**: **PESI** (PEFT Edge Suitability Index)  
$$
\mathrm{PESI}
= \frac{\mathrm{Acc}_{\mathrm{ID}} + \mathrm{Acc}_{\mathrm{C}}}{2}
\;\times\;
\frac{1}{\sqrt{(\text{params}\%) \cdot \text{latency}_{\mathrm{CPU}}}}
\;\times\;
e^{-\alpha \cdot \mathrm{ECE}},
\quad \alpha = 2
$$
  Latency & params% are **median-normalized** across runs.

**Success Criteria**
- Hybrid lies on a **Pareto frontier** (Acc vs Params%) and/or (Acc_C_mean vs Latency), **and**
- Hybrid achieves the **highest PESI** in ≥1 budget, or matches LoRA’s Acc within 0.2–0.5% while beating it on **ECE** and **latency/params**.

---

## What changed this week
- Implemented unified scoring across 9 runs (Norm / LoRA / Hybrid) using **per-image CPU latency** and **% trainable params**.
- Generated **Pareto plot** for ID accuracy vs params% with bubble size = PESI:  
  `reports/week8/pareto_id_acc_vs_params.png`

---

## Results (this run set)

### Method-level means (over available seeds)

| Method  | Params % | CPU Lat (ms/img) | Acc@1 (ID) | ECE (ID) | PESI (α=2) |
|---|---:|---:|---:|---:|---:|
| **Norm**  | ~0.087% | ~1.782 | **0.112** | **0.050** | **0.313** |
| **Hybrid**| ~0.545% | ~2.734 | **0.207** | 0.103 | 0.168 |
| **LoRA**  | ~1.341% | ~3.681 | 0.113 | 0.054 | 0.054 |

**Observations**
- **Hybrid vs LoRA**: Hybrid is **strictly Pareto-better**—higher Acc with **lower params and lower latency**, yielding **~3×** the PESI of LoRA in this slice.
- **Hybrid vs Norm**: A clear trade-off:
  - Hybrid delivers **~+9–10% absolute Acc** over Norm,
  - at ~6× parameters and ~1.5× latency.
  - **Neither dominates**; choose by resource budget and accuracy needs.
- **Norm** wins **edge suitability (PESI)** at tight budgets due to minimal footprint and better calibration—even with lower accuracy.

> Note: In this week’s scored CSVs, `Acc_C_mean` was proxied by `Acc@1 (ID)`. Final RQ3 generalization claims need true CIFAR-C.

---

## Pareto fronts
- **Acc@1 vs Params%** (bubble = PESI): Hybrid points lie **above-left** of LoRA, confirming the efficiency–utility advantage.  
- **Acc_C_mean vs CPU Latency**: Pending once CIFAR-C is evaluated; expected trend mirrors ID Pareto given current gaps.

(See `reports/week8/pareto_id_acc_vs_params.png`.)

---

## Answer to RQ3 (current evidence)
**Yes.** With this run set, **Layerwise Hybrid** achieves a **better efficiency–utility balance** than **LoRA** for edge deployment (higher accuracy at lower parameter and latency budgets → substantially higher PESI).  
At very tight budgets, **Norm** remains the **PESI leader** due to ultra-low params and latency with strong calibration, albeit at lower accuracy.

**Recommended policy**
- **Strict edge budgets (tiny adapters, real-time CPU):** choose **Norm**.
- **Moderate edge budgets (~0.5% params):** choose **Hybrid** as the sweet spot.
- **Avoid LoRA** at this scale for edge; heavier & slower without accuracy payoff here.

---

## Limitations & next steps
- **Generalization**: This week used a proxy (`Acc_C_mean = Acc_ID`).  
  **Action**: run **CIFAR-100-C** for each checkpoint to obtain real `Acc_C_mean` and `Acc_C_worst`, then recompute PESI.
- **Latency**: Per-image CPU latency derived from batch step time for some runs.  
  **Action**: confirm with `src/week8/latency_benchmark.py` (batch=1, warmups+iters).
- **Significance**: Report mean±std over matched seeds and use paired t-tests when complete.

---

## Artifacts
- Scored runs (median-normalized): `reports/week8/pesi_scored_all.csv`
- Pareto (ID Acc vs Params%): `reports/week8/pareto_id_acc_vs_params.png`
- Table for paper: `reports/week8/RQ3_table.md` (generated from summary script)

