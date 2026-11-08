# INT4 collapse diagnosis: handoff summary (Amanda)

Activation quantizer: amanda, 8-bit. All numbers on the full 10,000-image test set.

## Checkpoint comparison (FP32 activations)
- Shared: FP32 89.82 | W8 89.87 | W4 17.58
- Hub: FP32 94.05 | W8 94.02 | W4 92.84
- Hub at A8: FP32 / A8 93.95, W8 / A8 93.93, W4 / A8 92.80

## Where it breaks
- Block-1 depthwise, isolated W4 output SQNR: shared 8.79 dB vs hub 23.38 dB (W4 weight SQNR 21.39 vs 20.26 dB)
- Logit SQNR at uniform W4: shared 2.68 dB vs hub 15.19 dB
- Spearman(isolated W4 SQNR, midterm sensitivity drop): -0.70
- W4 only in blocks 0-3: 18.00% | W4 only in blocks 4-19: 70.10%

## Fixes at A8
- W8/A8 89.67% | uniform W4/A8 17.62%
- + MSE clip 20.93% | + bias correction 77.06%
- Chosen on validation (smallest within 1 pt of W8/A8): Top-k + bias correction Top-30 -> test 89.12% (validation 89.50%) at 1.671 MB deployed (W8: 2.339 MB)

## Pareto frontier at A8 (deployed MB, accuracy)
- Uniform W4 + bias correction: 1.238 MB, 77.06%
- Whole blocks + bias correction Block 1 only: 1.238 MB, 80.89%
- Whole blocks + bias correction Blocks 0-1: 1.239 MB, 84.85%
- Whole blocks + bias correction Blocks 0-2: 1.241 MB, 86.16%
- Whole blocks + bias correction Blocks 0-3: 1.245 MB, 86.72%
- Whole blocks + bias correction Blocks 0-4: 1.250 MB, 87.73%
- Whole blocks + bias correction Blocks 0-5: 1.257 MB, 87.84%
- Whole blocks + bias correction Blocks 0-7: 1.274 MB, 88.78%
- Top-k + bias correction Top-30: 1.671 MB, 89.12%
- Top-k Top-34: 1.773 MB, 89.13%
- Top-k Top-40: 1.788 MB, 89.18%
- Top-k Top-42: 1.803 MB, 89.28%
- Top-k Top-43: 1.880 MB, 89.47%
- Top-k Top-45: 1.984 MB, 89.51%
- Top-k Top-46: 2.012 MB, 89.55%
- Uniform W8: 2.339 MB, 89.67%

## Retrained variants
- Shared checkpoint (64x64) (64x64, torchvision strides): FP32 89.82 | W4 17.58 | W4/A8 17.62

## Evidence table
- **Weight outliers set too coarse a 4-bit step** (MSE-optimal clipping (C2.5)): Uniform W4 17.58% -> 20.86% with MSE clip (+3.28 pts). Suggested: weakened.
- **Damage is a systematic per-channel offset (smooth upsampled input)** (Bias correction (C2.5)): Uniform W4 17.58% -> 77.22% with bias correction (+59.64 pts). Suggested: supported.
- **Early layers alone cause the collapse** (W4 in one region only (C2.3)): W4 only in blocks 0-3: 18.00%; W4 only in blocks 4-19: 70.10%. Suggested: supported.
- **Error compounds through the early blocks** (Isolated vs cumulative SQNR (C1.4, C2.4)): Mean compounding in blocks 1-3: ours 5.61 dB, hub 3.88 dB. Suggested: compare: similar values mean compounding is not specific to our model.
- **ReLU6 saturation in early layers** (Fraction of inputs at the cap of 6 (C3.1)): Blocks 0-3 mean: ours 1.436%, hub 0.000%. Suggested: check magnitude: matters most for A8 (see C3.2).
- **Collapse is reproducible with another training seed** (Retrain variant res64_seed1 (C3.4)): not run yet. Suggested: untested (run retrain_variants.py).
- **Input resolution (64x64 upsampled vs native 32x32)** (Retrain variant res32 (C3.4)): not run yet. Suggested: untested (run retrain_variants.py).
- **Resolution + hub-like strides** (Retrain variant res32_cifar_strides (C3.4)): not run yet. Suggested: untested (run retrain_variants.py).
- **Whole-block protection beats layer-wise top-k** (Matched-storage comparison (C2.2 without, C3.2 with bias correction)): Without bias correction: blocks win 3 of 8 pairs (range -9.49 to +4.60 pts). With bias correction at A8: blocks win 6 of 7 near-equal-storage pairs (mean +3.31 pts). Suggested: mixed: depends on bias correction and budget.

## Files (results/)
- a8_results.csv
- amanda_handoff.md
- amanda_neurosim_configs.json
- block_protection_results.csv
- block_vs_topk_matched.csv
- checkpoint_comparison.csv
- clipping_bias_correction_results.csv
- error_growth_by_block.csv
- evidence_table.csv
- fig_c2_block_vs_topk.png
- fig_c2_error_by_block.png
- fig_c3_pareto_a8.png
- fig_early_weight_distributions.png
- fig_error_vs_sensitivity.png
- fig_layer_error_by_depth.png
- layer_quantization_error.csv
- layer_weight_stats.csv
- pareto_frontier_a8.csv
- relu6_saturation.csv
- retrain_comparison.csv
- slide_1_where_it_breaks.png
- slide_2_fixes_and_cost.png
- target_selection_validation.csv
- topk_curve.csv
- w4_region_results.csv
