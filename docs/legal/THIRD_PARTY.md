# Third-party components

Core runtime dependencies are NumPy, PyTorch, and Pillow. Matplotlib is an
optional visualization dependency. Exact CPU and CUDA 12.6 dependency graphs,
hashes, SPDX/license expressions, and CycloneDX 1.6 components are tracked in
`pylock.*.toml` and `sbom/`.

BM3D and escnn are research-only requirements in separate files. They are not
distribution extras and are not relicensed. Any `UNKNOWN` license classification
is a release-blocking error.
