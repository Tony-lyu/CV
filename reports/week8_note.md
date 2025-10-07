# Week 7 — RQ3: Efficiency vs. Generalization (Edge Suitability)

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
- **Shift (CIFAR-C)**: mean Acc across corruptions (**Acc_C_mean**), **worst corruption** Acc (**Acc_C_worst**)
- **Efficiency**: trainable params %, **batch-1 CPU latency**, (GPU latency optional)
- **Composite**: **PESI** (PEFT Edge Suitability Index)  
  \[
  \text{PESI} = \underbrace{\frac{\text{Acc}_\text{ID} + \text{Acc}_\text{C}}{2}}_{\text{utility}}
  \times \underbrace{\frac{1}{\sqrt{\text{params\%} \cdot \text{latency}_\text{CPU}}}}_{\text{efficiency}}
  \times \underbrace{\exp(-\alpha \cdot \text{ECE})}_{\text{calibration}},\ \alpha=2
  \]
  Latency and params% are normalized by the median across runs.

**Success Criteria**
- Hybrid sits on the **Pareto frontier** in (Acc vs Params%) and/or (Acc_C_mean vs Latency), **and**
- Hybrid achieves the **highest PESI** in ≥1 budget, or **matches** LoRA’s Acc within 0.2–0.5% while beating it on **ECE** and **latency/params**.

---

## How to Run

1. **(Optional) Get CIFAR-100-C**
   - Place under `./CIFAR-100-C` or set `CIFAR_C_DIR=/path/to/CIFAR-100-C`

2. **Train the matrix**
   - Edit `scripts/rq3_matrix.sh`:
     - Point the **train** block to your `train.py`
     - Ensure each run saves a **whole-model** checkpoint at `checkpoints/m=<method>_b=<budget>_seed=<seed>.pt`
   - Then:
     ```bash
     bash scripts/rq3_matrix.sh
     ```

3. **Outputs**
   - `runs_rq3/eval_id_c.csv` — ID & CIFAR-C results (+ any tags you added)
   - `runs_rq3/latency.csv` — CPU/GPU latencies
   - `runs_rq3/merged.csv` — joined metrics
   - `runs_rq3/scored_pesi.csv` — metrics + **PESI**

**CSV columns (merged & scored)**
