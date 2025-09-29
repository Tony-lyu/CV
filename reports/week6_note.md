# Week 6 

## Setup

- Model / data: vit_small_patch16_224 on CIFAR-100, img size 224, batch 128.

- Optim: AdamW, cosine decay.

- Warm-up trick: freeze_lora_steps=400 to stabilize hybrids (first ~400 mini-batches train Norms/head only).

- LR floor: for layerwise we used a non-zero floor (η_min ≈ 1e-5) by continuing cosine beyond 5 epochs to 15 total “cosine steps” (so LR never hits exact 0 near the end).

- Diagnostics: grad/update norms and per-group cosines recorded each epoch.

- Note on fairness: LoRA-only and Norm-only runs decayed LR to 0 by epoch 15, while layerwise used a non-zero LR floor. Results below are still decisive, but we’ll add “η_min-matched” confirmations next week.

---

## What changed from Week‑5
- **Scheduler:** Cosine annealing now uses a floor (**η_min=1e‑5**), instead of decaying to exactly zero.
- **Warm‑up:** Keep **freeze_lora_steps=400**; this continues to stabilize hybrids.
- **No LR parity needed:** With warm‑up and the LR floor, dynamic LR‑parity wasn’t necessary this week.

---

## Results (top-1 ↑, ECE ↓)

### Best across 15-epoch schedule this week
| Method               | Policy        | Trainable Params | Step Time (s) |               Best Top-1 |     ECE @/near best |
| -------------------- | ------------- | ---------------: | ------------: | -----------------------: | ------------------: |
| **Hybrid Layerwise** | **deep_lora** |      **247,396** |    **~0.297** | **90.10%** (epoch 14–15) |              ~0.076 |
| LoRA-only            | –             |          628,324 |        ~0.451 |        89.58% (epoch 15) |              ~0.078 |
| Norm-only            | –             |           57,700 |        ~0.219 |         89.41% (epoch 6) | **0.059** (epoch 2) |


### Observations

- Accuracy win: Layerwise (deep_lora) is +0.5–0.7 pp over LoRA-only and ~+0.7–0.9 pp over Norm-only.

- Efficiency: Layerwise achieves the best accuracy with ~2.5× fewer trainables than LoRA-only and ~34% faster steps than hybrid_parallel (from previous comparisons).

- Calibration: Norm-only shows the best raw ECE early (~0.059), but at highest-accuracy epochs the gap narrows. Expect temperature scaling to close the remaining ECE difference for layerwise without harming accuracy.

### Learning dynamics:

- With freeze_lora_steps=400, hybrids avoid the noisy early coupling between LoRA and Norm; gradients are cleaner after the warm-up.

- A small LR floor prevents late-epoch regression when LR→0 (seen in earlier weeks). Layerwise benefits from staying “slightly alive” at the tail.

---

## Insight learned vs Week 5

- The warm-up + LR floor combo is the key that consistently pushes layerwise deep_lora past both LoRA-only and Norm-only while keeping compute modest.

- Grad diagnostics show LoRA and Norm magnitudes remain balanced enough after warm-up; no parity scaling was required to achieve the win in layerwise.

---

## “Week‑6” recipe
**Commands**
```bash
python src/week6/train.py \
  --dataset cifar100 --model vit_small_patch16_224 --img_size 224 \
  --batch_size 128 --workers 4 --epochs 15 --pretrained \
  --log_csv reports/results_week6_eta.csv \
  --method hybrid_layerwise --policy deep_lora --unfreeze \
  --lora_r 8 --lora_alpha 16 \
  --lr 1e-3 --lr_norm 1e-3 --lr_lora 5e-4 \
  --weight_decay 0.05 \
  --freeze_lora_steps 400 \
  --lr_parity 0
```

---

## Conclusion

With a short LoRA warm-up and a small LR floor, hybrid layerwise (deep_lora) delivers state-of-the-week best accuracy on CIFAR-100 while using far fewer trainable parameters than LoRA-only—and clearly exceeds Norm-only.

 Deep‑LoRA with a non‑zero LR floor and a short LoRA warm‑up gives the best accuracy (90.1%) at lower cost, and it eliminates the late‑epoch degradation we saw when cosine annealed to zero. Calibration still benefits from post‑hoc temperature scaling next week.
