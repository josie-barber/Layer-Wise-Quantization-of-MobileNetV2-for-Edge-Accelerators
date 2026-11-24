# NeuroSim A8 runs (Dana, Nov 10)
- Same settings as midterm (FP mode, conventional mapping, batch 8, 1 batch, subArray 128, ADC 5-bit, cellBit 1), but activations fixed at 8 bits.
- Checkpoint SHA-256 matches Lilly's (2c7af937...a32ae433). W8A8 reproduces the midterm INT8 numbers exactly.
- Replaces the midterm INT4 rows, which were W4A4. 4-bit energy saving is now ~1.8x (was 3.3-3.7x).
- "unscaled": inputs use NeuroSim's default [-1,1) encoding, so values >=1 saturate. Will rerun after rescaling with Lilly's calibration ranges.
- Logs include per-layer latency/energy, used for mixed-precision split-and-sum.
