# Getting started

Install the core distribution with `pip install structfield-vision==1.0.0`, or
install a local checkout with `pip install .`. Matplotlib is only required for
visual export and lives in the `visualization` extra.

```python
import torch
from structfield import AnalyticStructuralFieldEstimator

image = torch.rand(1, 1, 64, 64)
field = AnalyticStructuralFieldEstimator()(image)
top1 = field.topk_orientations(k=1)
```

External tensors are strictly checked by default. `ValidationMode.trusted`
retains type, shape, dtype, device, and schema checks while omitting
synchronizing value checks; use it only inside a controlled tensor pipeline.
