# Analytic structural-field estimator

## Public API

```python
from structfield import AnalyticStructuralFieldEstimator

estimator = AnalyticStructuralFieldEstimator(num_orientations=16)
field = estimator(image)
field, diagnostics = estimator.estimate_with_diagnostics(image)
```

`image` must be a finite floating-point tensor shaped `[B,1,H,W]` or `[B,3,H,W]`. RGB channel values are combined as `0.2126 R + 0.7152 G + 0.0722 B`; gamma decoding is intentionally outside this estimator. The result preserves input dtype and device. Float16 and bfloat16 inputs are evaluated internally in float32 to avoid epsilon underflow and are cast back through differentiable operations.

## Analytic pipeline

1. Optional weak Gaussian denoising.
2. Local mean/variance normalization with a bounded standardized output.
3. Multi-scale structure-tensor eigensystem, stable coherence, scale response, and scale posterior.
4. Multi-frequency complex Gabor evidence over uniform axial tangent bins.
5. A normalized orientation-frequency posterior with orientation and frequency marginals.
6. Frequency posterior mean/variance, normalized orientation entropy, and axial circular top-k modes.
7. Optional multi-scale Hessian line diagnostics.

The public `StructuralField` contains task-neutral estimates. Raw and joint analytic evidence stays in `StructuralFieldDiagnostics`. The estimator writes the parameter-free five-factor `heuristic-v0.2` result into `structural_uncertainty`; its definition and limitations are frozen in `STRUCTURAL_UNCERTAINTY.md`.

## Coordinate correction

V0.2 angles increase counter-clockwise on the displayed image. With row coordinates increasing downward, the tangent vector is `(cos(theta), -sin(theta))`. V0.1 used the opposite numeric sign, so parity evaluation maps old angles with `(-theta) mod pi` before comparison. The estimator does not preserve the old sign error for the sake of raw numeric parity.

## Numerical corrections

- The tensor and Hessian eigengaps use a zero-preserving stabilized square root. Adding epsilon inside the old square root created coherence `1000.000061` on a flat image.
- Coherence is constrained to `[0,1]`.
- Local variance at or below `normalization_eps^2` is treated as non-informative and normalized to exact zero. This prevents GPU convolution roundoff on constant images from being amplified into spurious directional evidence.
- Zero Gabor evidence produces uniform orientation/frequency posteriors instead of artificial positive evidence.
- Frequency variance is evaluated as the posterior expectation of squared deviations.
- Small images fall back from reflect to replicate padding when the filter radius requires it.

## Algorithm provenance

The code is an independent project implementation of standard analytic ideas. It does not copy an external implementation. Algorithm-level references are:

- J. Bigün, G. H. Granlund, and J. Wiklund, “Multidimensional orientation estimation with applications to texture analysis and optical flow,” 1991, [DOI 10.1109/34.85668](https://doi.org/10.1109/34.85668).
- D. Gabor, “Theory of communication. Part 1: The analysis of information,” 1946, [DOI 10.1049/ji-3-2.1946.0074](https://doi.org/10.1049/ji-3-2.1946.0074).
- A. F. Frangi et al., “Multiscale vessel enhancement filtering,” 1998, [DOI 10.1007/BFb0056195](https://doi.org/10.1007/BFb0056195).
- [Recommendation ITU-R BT.709-6](https://www.itu.int/rec/R-REC-BT.709/en) for the fixed RGB coefficient vector.

These references describe mathematical methods or a signal convention; they are not sources of copied code. Final whole-repository licensing remains subject to the provenance audit.
