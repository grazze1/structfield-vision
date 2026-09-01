"""Export a reproducible synthetic image and orientation-probability gallery."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from structfield import AnalyticStructuralFieldEstimator
from structfield.benchmarks import StructFieldBenchConfig, generate_sample

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path("assets/gallery/synthetic_gallery.png"))
args = parser.parse_args()
output = args.output
output.parent.mkdir(parents=True, exist_ok=True)
families = ("straight", "curved", "crossing")
config = StructFieldBenchConfig(height=48, width=48, num_orientations=16)
estimator = AnalyticStructuralFieldEstimator(num_orientations=16)
figure, axes = plt.subplots(2, len(families), figsize=(7.2, 4.8), constrained_layout=True)
for column, family in enumerate(families):
    sample = generate_sample(family, split="validation", index=0, config=config)
    field = estimator(sample.image.unsqueeze(0))
    axes[0, column].imshow(sample.image[0], cmap="gray", vmin=0, vmax=1)
    axes[0, column].set_title(family)
    axes[1, column].imshow(field.orientation_prob[0].amax(0), cmap="magma", vmin=0, vmax=1)
    for row in range(2):
        axes[row, column].axis("off")
figure.savefig(output, dpi=160)
plt.close(figure)
print(output)
