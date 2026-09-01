# Structural-field conventions

## Image coordinates

- Tensors use `[B,C,H,W]`.
- `x` increases to the right and indexes width.
- `y` increases downward and indexes height.
- Angles describe local line tangents, not gradient normals.

## Axial orientation

Orientations are unoriented lines:

```text
theta ∈ [0, pi)
theta ≡ theta + pi
```

Angles use radians. Axial distance is:

```text
min(|Δtheta| mod pi, pi - (|Δtheta| mod pi))
```

Doubled-angle vectors use `(cos(2 theta), sin(2 theta))`, so the vectors for `0` and `pi` are identical.

## Orientation bins

The default discretization has uniform centers `theta_k = k*pi/K` for `k=0,...,K-1`. V0.2 tests 8, 16, and 32 bins, while the schema accepts any integer `K >= 2` with valid increasing centers.

Top-k suppression uses the same axial distance across the wraparound boundary. A mode near `pi` is therefore close to a mode near zero.

## Frequency and scale

- Spatial frequency is measured in cycles/pixel.
- Frequency variance is measured in `(cycles/pixel)^2`.
- Scale is measured in pixels.
- Candidate values must be finite and positive.

## Rotation

Positive rotation is counter-clockwise in displayed image coordinates. A geometric transform must rotate both spatial maps and axial orientation values, wrap angles modulo `pi`, and provide a valid mask for padding/interpolation regions. These transform operations are implemented in Stage 7; this document freezes their convention in Stage 2.

Equivalently, the displayed-image tangent vector for angle `theta` is `(cos(theta), -sin(theta))` because tensor row coordinates increase downward. This differs in sign from the historical V0.1 implementation, whose numeric angle increased clockwise on screen. Legacy comparisons therefore use `theta_v02 = (-theta_v01) mod pi` and reflect orientation channels accordingly.

## RGB input

The schema itself stores field tensors and does not convert images. The Stage 3 estimator accepts grayscale or RGB BCHW tensors and applies the fixed BT.709 coefficient vector `(0.2126, 0.7152, 0.0722)` to RGB channel values. It does not perform gamma decoding; this choice is recorded in estimator metadata.
