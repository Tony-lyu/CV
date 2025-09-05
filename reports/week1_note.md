# Week 1 Notes

## Related Research Questions 
1. **Additivity**: Does LoRA+NormTune outperform each alone at matched or lower **params‑tuned %**?
2. **Efficiency**: Is **acc/params‑tuned** and **acc/time** better for hybrid?
3. **Calibration**: Does hybrid improve **ECE** vs. single methods?

## Week 1 Deliverables
- [x] Environment created and smoke tests run.
- [x] Three vit_tiny runs (LoRA / Norm / Hybrid) completed.
- [x] `reports/results_week1.csv` populated.
- [x] This note updated with quick observations/next steps.

## Early Observations 
### 1-epoch on 50 batches
- **NormTune wins early:**
With only 0.2% parameters tuned, it achieved the highest accuracy (82.7%) and the lowest calibration error (0.18).
Suggests normalization parameters alone carry strong adaptation capacity on CIFAR-10.
- **LoRA is competitive:**
At ~5% params, LoRA reached 79.0% accuracy — slightly worse than NormTune despite tuning ~25× more parameters.
Still a decent baseline, but efficiency is lower.
- **Hybrid collapsed:**
Hybrid (LoRA+NormTune) underperformed badly (57.2% acc), even though it tuned the same tiny fraction of weights.
Likely causes: interference between updates in the very short 1-epoch run, or learning-rate mismatch. Needs further debugging.
- **Training dynamics:**
All runs were limited to 1 epoch for speed on CPU. With longer training (5–10 epochs), LoRA and Hybrid might converge further.
Early results show NormTune is surprisingly strong and stable even under minimal training.

### 5-epoch on 50 batches
- **LoRA converges strongly:** At 5 epochs, LoRA achieved the best accuracy (92.0%) and low ECE (~0.087).
NormTune remains competitive: Reached 91.6% with just 0.2% of parameters tuned — confirming its remarkable efficiency.
- **Hybrid still underperforms:** Accuracy improved vs. 1-epoch (from 57% → 83%), but it failed to match either LoRA or NormTune alone, and calibration remained poor. Suggests optimization conflict or mis-allocation of learning rates.
- **No additive gains :** Hybridization, instead of yielding complementary improvements, produced interference under this setup.

### 1-epoch full vit_tiny
- **LoRA vs. Norm:** LoRA and Norm both achieve > 94% accuracy with ViT-Tiny, but Norm does so with ~0.2% parameters.
- **Hybrid catches up:** On full dataset, Hybrid recovers to Norm-like performance, but does not outperform either method. No additive benefit yet. 

## Summary
- Norm is the most parameter-efficient.
- LoRA tields slightly stronger calibration.
- Hybrid currently adds training overhead with no gain, performs extremely bad in few-shot situations. 