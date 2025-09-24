# Week 5 — Selective Freezing for Hybrid

## SummarySummary 

Added batch-wise selective freezing to hybrid adapter training and got two concrete wins over Week-4:

- Accuracy win: warming up LoRA for 200 batches improved CIFAR-100 Acc@1 from 90.04 → 90.21 (+0.17 pp) and ECE ↓ ~6% at the same cost.

- Efficiency + calibration win: alternating (LoRA-first, every 50 batches) delivered ~19% higher throughput, ~13% lower peak memory, and better ECE with only ~0.1 pp accuracy trade-off.
---

## Setup

**Model/Data**: ViT-Small/16 on CIFAR-100, img_size=224, batch_size=128, epochs=5 unless noted.

**Method**: hybrid_layerwise with deep_lora policy; head unfreezed (--unfreeze).

### New schedules:

- Warm-up LoRA: --freeze_lora_steps 200 (train Norm+head first 200 batches, LoRA frozen), then train both.

- Alternation (LoRA-first): --alt_freeze_every K (K∈{25,50,100}); alternate which group is trainable every K batches, starting by freezing LoRA (so Norm goes first).

- Metrics: Acc@1, ECE (15 bins), per-step wall clock (avg_step_time_s), and peak CUDA memory (peak_mem_bytes).

## Headline results
| Schedule                          | Acc\@1 (final) | Δ Acc vs. baseline |        ECE |                 Δ ECE | Step time (s) | Throughput (img/s) |  Peak mem (bytes) |      Δ mem |
| --------------------------------- | -------------: | -----------------: | ---------: | --------------------: | ------------: | -----------------: | ----------------: | ---------: |
| **Baseline (Week-4 recipe)**      |     **0.9004** |                  — |     0.0637 |                     — |        0.5698 |              224.7 |     6,145,093,632 |          — |
| **Warm-up LoRA (200)**            |     **0.9021** |        **+0.0017** | **0.0600** | **−0.0037 (\~−5.8%)** |        0.5714 |              224.0 |     6,145,093,632 |       \~0% |
| **Alternation K=50 (LoRA-first)** |         0.8993 |            −0.0011 | **0.0592** | **−0.0045 (\~−7.0%)** |    **0.4781** | **267.7 (+19.1%)** | **5,355,933,696** | **−12.8%** |

### Notes:

**Throughput** = 128 / step_time.

**Baseline row from**: acc1=0.9004, ece=0.0636929,

**avg_step_time_s**=0.5698237, 

**peak_mem_bytes**=6145093632.

**Warm-up LoRA best row**: acc1=0.9021, ece=0.0600139, avg_step_time_s=0.5714201.

**Alternation K=50 (LoRA-first) exemplar row**: acc1=0.8993, ece=0.0592258, avg_step_time_s=0.4781239, peak_mem_bytes=5355933696.

## Secondary findings & ablations

- **K sweep (LoRA-first)**: K∈{25,50,100} kept the efficiency gains; accuracy hovered ~0.897–0.899 (slightly below baseline), ECE improved across K.

- **Warm-up LoRA + Alternation**: freeze_lora_steps=200 then K=25 retained efficiency and the ECE gains; accuracy ≈ baseline −0.03–0.1 pp (small dip).

- **10-epoch alternation trials (K=25/50, and a LoRA LR bump to 7e-4)**: did not recover accuracy above the 5-epoch baseline; ECE sometimes regressed slightly.

- **Warm-up Norm (200) and symmetric 100/100 warm-ups**: neutral or slightly worse in both Acc and ECE; we drop them.

## Interpretation

### Early interference matters: 
letting Norm adapt first (LoRA frozen) for ~200 batches stabilizes training and yields a small but consistent accuracy + calibration improvement.

### Compute-efficient training:
turning off one branch half the time (alternation) cuts compute/memory while improving calibration—useful when wall-clock or memory are bottlenecks.

**LoRA-first** is better than Norm-first for alternation in my setting (Norm-first hurt accuracy noticeably).

More epochs didn’t fix the alternation acc gap here. Likely, the alternating constraint is a mild bias; you bank efficiency and ECE but concede ~0.1–0.3 pp Acc.

## Caveats / tooling notes

GradMonitor cosine shows 0/NaN. Likely per-layer vector pairing leads to samesize=0 often. Should compute a global cosine each step by concatenating all LoRA grads vs all Norm grads, then average by epoch. This will let me show reduced interference when schedules help ECE.

## Conclusion
Accuracy track: With a simple LoRA warm-up (first 200 batches), Acc@1 improved from 90.04% → 90.21% (+0.17 pp) and ECE improved from 0.0637 → 0.0600 at identical compute.

Efficiency track: Alternating selective freezing (LoRA-first, K=50) improved throughput by ~19% and reduced peak memory by ~13% while improving ECE (to 0.0592) and keeping accuracy within 0.1 pp of baseline.
