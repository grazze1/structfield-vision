# 08 Performance reference

CPU and CUDA references record 8/16/32 orientations at 32 and 64 pixel square
inputs, median latency samples, and CUDA peak allocated memory. They identify
hardware/software and explicitly make no performance commitment.

Tracked records are `benchmarks/performance_cpu_reference.json` and
`benchmarks/performance_cuda_reference.json`. The RTX 4050 reference observed
peak allocated memory from 858,112 bytes (8 orientations, 32x32) to 12,068,352
bytes (32 orientations, 64x64). These measurements are environment records,
not cross-machine targets or speed claims.
