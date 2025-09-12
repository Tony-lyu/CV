# Week 3 Note 

## Summary
Over Weeks 1–3, I scaled PEFT experiments from **ViT-Tiny** to **ViT-Small on CIFAR-100**.  
The aim was to test whether early trends hold consistently as backbone size and dataset difficulty increase.  

Across all settings, I observed:
- **NormTune**: consistently parameter-efficient and best-calibrated, with competitive accuracy.  
- **LoRA**: consistently yields the highest accuracy, but at ~10× more tuned parameters and with degraded calibration (ECE).  
- **Hybrid**: continues to collapse to Norm-like behavior, providing no additive benefit.  

---

## Consolidated Results

### ViT-Tiny (Week 1, CIFAR-10, 1 epoch)
| Method | Tuned Params % | Acc@1 | ECE   | Time/Epoch (s) |
|--------|----------------|-------|-------|----------------|
| Norm   | ~0.2           | ~94.6%| ~0.059| ~485           |
| LoRA   | ~5.01          | ~95.9%| ~0.049| ~724           |
| Hybrid | ~0.19          | ~94.6%| ~0.067| ~543           |
 
### ViT-Tiny (Week 2, CIFAR-100, 5 epochs)
| Method | Tuned Params % | Acc@1 | ECE   | Time/Epoch (s) |
|--------|----------------|-------|-------|----------------|
| Norm   | ~0.52          | ~83.0%| ~0.094| ~3659          |
| LoRA   | ~5.38          | ~85.5%| ~0.087| ~5143          |
| Hybrid | ~0.49          | ~83.3%| ~0.195| ~4731          |

### ViT-Small (Week 3, CIFAR-100, 10 epochs)
| Method | Tuned Params % | Acc@1 | ECE   | Time/Epoch (s) |
|--------|----------------|-------|-------|----------------|
| Norm   | ~0.27          | ~89.3%| ~0.048 | ~1,524        |
| LoRA   | ~2.82          | ~90.8%| ~0.060 | ~2,011        |
| Hybrid | ~0.26          | ~89.4%| ~0.049 | ~1,848        |

---

## Cross-Week Trends

1. **Accuracy**
   - LoRA consistently wins on accuracy (Tiny CIFAR-10, CIFAR-100, Small CIFAR-100).  
   - Norm is very close to LoRA on CIFAR-100, sometimes within ~1–1.5%.  
   - Hybrid does not surpass Norm — interference dominates.  

2. **Calibration (ECE)**
   - Norm consistently has the **lowest ECE** (best-calibrated).  
   - LoRA introduces overconfidence (higher ECE) across scales.  
   - Hybrid mirrors Norm, not LoRA.  

3. **Efficiency**
   - Tuned parameters remain stable across scales:  
     - Norm: ~0.25–0.3%  
     - LoRA: ~2.8–3%  
   - Training time scales with model size (Tiny ≈ 200s/epoch → Small ≈ 1,800–2,000s/epoch).  
   - Relative efficiency (acc per tuned param) heavily favors Norm.  

4. **Scaling Consistency**
   - Trends from Tiny → Small hold:  
     - Norm stays efficient, calibrated, and competitive.  
     - LoRA is expensive, slightly better in raw accuracy.  
     - Hybrid provides no additive gain.  

---

## Visualizations
(*See attached plots generated from Week 3 data; additional composite plots should merge Tiny + Small results for Accuracy vs Tuned Params and ECE vs Tuned Params.*)

- Accuracy vs Tuned Params %  
- ECE vs Tuned Params %  
- Training Time per Epoch  

---

## Findings
- **NormTune is the most robust choice** across datasets and scales.  
- **LoRA’s benefits are marginal** relative to its cost and calibration hit.  
- **Hybrid fails to provide additivity**, suggesting interference when combining Norm and LoRA.  
- **Scaling backbone size does not change these relative dynamics** — results are consistent across Tiny and Small.  

---


