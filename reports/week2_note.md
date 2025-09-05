# Week 2 Notes

## Related Research Questions 
1. **Additivity**: Does LoRA+NormTune outperform each alone at matched or lower **params‑tuned %**?
2. **Efficiency**: Is **acc/params‑tuned** and **acc/time** better for hybrid?
3. **Calibration**: Does hybrid improve **ECE** vs. single methods?

## Week 1 Deliverables
- [x] Extend dataset coverage to CIFAR-100.
- [x] Run full NormTune and LoRA 10-epoch baseline on CIFAR-100 (ViT-Tiny).
- [x] Run Conduct Hybrid LR sweep to diagnose intererence.
- [x] Identify best Hybrid LR config and run 10-epoch (norm=1e3,lora=5e-4).
- [x] Populate`reports/results_week2.csv`.

## Early Observations 
### NormTune (CIFAR-100, ViT-Tiny, 10 epochs)
- 82.99% accuracy at 0.52% tuned params.
- Calibration is good (ECE=0.054).
- Strong confirmation that NormTune is extremely parameter-efficient and generalizes beyond CIFAR-10.

### LoRA (CIFAR-100, ViT-Tiny, 10 epochs)
- 85.53% acc @ 5.38% tuned params.
- Best accuracy overall, but at ~10× higher parameter budget than Norm/Hybrid.
- Calibration weaker (ECE=0.081) → more overconfident predictions.
- Training cost highest (~51k s on CPU run).

### Hybrid (CIFAR-100, ViT-Tiny, tuned LR, 10 epochs)
- 83.33% accuracy at 0.50% tuned params.
- Matches NormTune on parameter budget, slightly higher accuracy (+0.3%).
- Calibration is strong (ECE=0.055, between Norm and LoRA).
- LR sweep showed that imbalanced learning rates (norm slightly higher than lora) can avoid/reduce interference.


## Summary
- **NormTune:** Most parameter-efficient,consistently stable, strong calibration.
- **LoRA tields:** Best raw accuracy, but less efficient and weaker calibration.
- **Hybrid:** Shows additive potential after tuning. Beating NormTune in accuracy while retaining its efficiency. Calibration is competitive, runtime is manageable. 
- **Week-1's concern about Hybrid collapse is resolved: tuning learning-rate reduce/avoid interference issue.**