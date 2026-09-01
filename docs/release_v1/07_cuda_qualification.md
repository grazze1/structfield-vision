# 07 CUDA qualification

Qualification uses the locked `pylock.cu126.toml` environment on the local RTX
4050 Laptop GPU. It covers batch/RGB, 8/16/32 orientations, uncertainty,
adapters, corruption, equivariance backward, all three application
forward/backward paths, checkpoint load, benchmark, CLI and peak memory.

Final qualification result: **PASS**. The fresh environment used Python
3.11.15, PyTorch 2.13.0+cu126 and CUDA 12.6 on an NVIDIA GeForce RTX 4050
Laptop GPU (6141 MiB, driver 580.173.02). A real CUDA tensor probe succeeded,
then the public GPU-enabled suite reported 900 passed with no skips.
