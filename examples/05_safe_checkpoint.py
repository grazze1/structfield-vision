"""Round-trip an untrained reference application through the safe format."""

import tempfile
from pathlib import Path

from structfield import AnalyticEstimatorConfig, AnalyticStructuralFieldEstimator
from structfield.checkpoint import load_application_checkpoint, save_application_checkpoint
from structfield.experimental import applications

model = applications.CurvilinearSegmenter(num_orientations=8, width=4)
estimator = AnalyticStructuralFieldEstimator(AnalyticEstimatorConfig(num_orientations=8))
with tempfile.TemporaryDirectory() as temporary:
    destination = Path(temporary) / "checkpoint"
    save_application_checkpoint(
        destination,
        model,
        estimator=estimator,
        preprocessing={"range": [0.0, 1.0]},
        training_provenance={"status": "untrained", "datasets": []},
    )
    loaded = load_application_checkpoint(destination)
    print(type(loaded.model).__name__)
