# Week 4 Note 

## Summary
Goal: explain the earlier “hybrid collapse to norm” and test whether a layerwise split (Norm early, LoRA late) fixes it—while reproducing Week-3’s parallel hybrid as a baseline.  

### Bottom line

- There is no gradient-direction collapse. The earlier collapse was due to placement + step sizes, not destructive alignment.

- Layerwise deep-LoRA (LoRA only in later ViT blocks, Norm earlier) reaches the same accuracy as parallel hybrid at much lower cost.

- On this CIFAR-100 + ViT-Small transfer, Norm-only already nearly solves the task, slightly edging LoRA/Hybrid while being by far the cheapest—explaining why Hybrid often looked “Norm-like.”

---

### Experiment Setup
- Backbone: vit_small_patch16_224 (timm), pretrained, classifier unfrozen (Week-3 parity).

- Data & schedule:
   - hybrid_parallel uses Week-3 augs(Resize→RandomCrop(pad=4)→Flip) + Cosine LR.
   - hybrid_layerwise (hybrid_layers) uses standard augs (+Normalize) + Cosine LR.
- Optimizer & WD: AdamW; Head/Other WD=0.05, Norm & LoRA WD=0.0 (Week-3 parity for hybrid).

- Diagnostics: per-epoch grad/update norms for LoRA vs Norm + cosine similarity.

- Eval stability: logits offloaded to CPU; eval batch = 128 to avoid VRAM spikes

---


### Consolidated Results (ViT-Small, CIFAR-100, 10 epochs)
| Method                 | Policy         |  Trainables |         Step time |        Peak VRAM |           Acc\@1 (best) |      ECE (≈final) |
| ---------------------- | -------------- | ----------: | ----------------: | ---------------: | ----------------------: | ----------------: |
| **Hybrid (parallel)**  | –              | **647,524** |      **\~0.73 s** | **\~7.5–8.0 GB** |               **89.5%** |           \~0.074 |
| **Hybrid (layerwise)** | **deep\_lora** | **247,396** |      **\~0.57 s** |     **\~6.1 GB** |               **89.6%** |           \~0.075 |
| **LoRA-only**          | –              | **628,324** | **\~0.72–0.81 s** |     **\~8.0 GB** |               **89.3%** |           \~0.077 |
| **Norm-only**          | –              |  **57,700** |      **\~0.50 s** |     **\~5.2 GB** | **89.8%** *(best/≈tie)* | **\~0.058–0.062** |

### Takeaways

- Accuracy: deep_lora ≈ parallel ≈ LoRA-only, with Norm-only slightly best this week.

- Efficiency: deep_lora uses ~62% fewer trainables and ~22% less step time than parallel; Norm-only is smallest & fastest.

- Calibration: Norm-only is best calibrated; LoRA-only worst; layerwise/parallel in-between.

### Why “collapse” happened before (and not now)
#### Not gradient conflict
- Measured cosine(grad_LoRA, grad_Norm) ≈ 0 (or undefined when one side absent) → no systematic destructive alignment.

#### It was placement + step sizes
- Parallel hybrid updates every block with both LoRA and Norm; with earlier settings LoRA tended to dominate the step while the head/setup limited additivity, so the overall behavior looked “Norm-like.”

- Layerwise deep_lora puts LoRA in later blocks and Norm earlier, yielding balanced update norms (LoRA/Norm ≈ 1–1.6× in your logs) and stable convergence at far lower cost.

#### Policy ablations confirm placement
- deep_lora > odd_lora ≫ even_lora (even-LoRA notably worse).
⇒ LoRA belongs later, Norm earlier in ViT depth.
---

## Reconciling Week-3 (LoRA > Norm) vs Week-4 (tie / Norm-slight-win)

### This is recipe-sensitive, not contradictory.
- Week-3 LoRA used an effectively larger LoRA step and different augs, giving it a small edge (~90.8% vs ~89.3%).

- Under Week-4’s recipe (WD layout, augs, LR splits), Norm-only ties/edges Hybrid/LoRA and deep_lora matches parallel at lower cost.

### Conclusion
- the earlier “collapse” wasn’t inherent—where we update and how hard we push it determines additivity.

---

## Findings
- **Mechanism:** No gradient-direction collapse; outcomes are driven by placement and step sizes. 
- **Best trade-off:** Layerwise deep-LoRA achieves parallel accuracy withfar fewer params, lower VRAM and faster steps.  

---


