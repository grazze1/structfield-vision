# Structural-field conventions

`StructuralField` stores one authoritative orientation probability tensor with
shape `[B,K,H,W]`. `topk_orientations()` derives separated modes dynamically.
Scalar maps use `[B,1,H,W]`; scale probability, when present, uses `[B,S,H,W]`.

- Orientation is an unoriented axis with period π and range `[0,π)`.
- Positive rotation is counter-clockwise in displayed coordinates where x is
  right and tensor y is down.
- Frequency is non-negative cycles/pixel.
- Scale is non-negative pixels.
- Probability channels sum to one in strict mode.

Coherence describes local directional concentration. Structural uncertainty is
a five-signal, parameter-free baseline heuristic. Reliability is a distinct
downstream prediction and is never treated as a synonym for either quantity.
