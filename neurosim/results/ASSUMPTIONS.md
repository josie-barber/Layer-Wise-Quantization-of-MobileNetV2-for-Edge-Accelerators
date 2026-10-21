# MobileNetV2 on DNN+NeuroSim V1.4: assumptions

- **FP mode, not WAGE.** WAGE expects networks trained within ±1; our FP32-trained model clipped 17–81% of conv outputs and scored 1/8. FP mode quantizes by each tensor's own range.
- **Conventional mapping.** NeuroSim's default novel mapping needs a 3×3 layer with ≥128 rows; the depth-1 file has none, so it aborts. Both bounds use conventional mapping for a like-for-like comparison.
- **Depthwise bounds.** *Dense* stores each depthwise conv as a block-diagonal dense conv: overcounts cells (~19×), area, energy and operations. *Depth-1* stores the real 3×3×C weights but feeds one 9-input window to all C columns: undercounts input reads, buffer/interconnect traffic and read cycles (~C× per depthwise layer).
- **NeuroSim code fixes.** `Chip.cpp`: novel-mapping tiles overran arrays (crash) when channels×bits isn't a multiple of 512; VGG8 results unchanged. `Param.cpp`: opt-in `NEUROSIM_CONVENTIONAL_MAPPING=1`; default unchanged.
- **ReLU6 saturation.** NeuroSim's input encoder assumes [−1, 1); activations up to 6 become all-ones, inflating input activity and energy.
- **Accuracy** (≈89% CIFAR-10) comes from the team's PyTorch evaluation, not from NeuroSim's hardware simulation.
