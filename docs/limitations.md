# Limitations

StructField-Vision 1.0.0 is an engineering release, not a clinical device or a
claim of state-of-the-art task performance. The analytic estimator can be
sensitive to boundaries, low contrast, nonstationary texture, and structures
outside its fixed frequency/scale candidates.

The uncertainty implementation is a transparent heuristic baseline; coherence,
uncertainty, calibration, and learned reliability are different concepts. The
reference applications are experimental and ship without trained weights.

The deterministic benchmark is synthetic. Its release baseline detects software
regressions but does not establish generalization to real imagery. Performance
JSON files describe specific hardware and are not performance promises.

Supported release scope is Linux x86_64, Python 3.11, CPU and NVIDIA CUDA 12.6.
Windows, macOS, ROCm, external data, long training, and model downloads are not
qualified by this release.
