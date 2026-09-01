"""V0.3 application boundaries with explicit StructuralField contracts."""

from .curvilinear_segmentation import (
    CurvilinearSegmenter,
    curvilinear_segmentation_loss,
    soft_dice_loss,
)
from .fringe_restoration import (
    FringeRestorationLoss,
    FringeRestorationLossConfig,
    FringeRestorationLossOutput,
    FringeRestorer,
)
from .texture_restoration import (
    TextureRestorationLoss,
    TextureRestorationLossConfig,
    TextureRestorer,
    texture_restoration_loss,
)

APPLICATION_SCHEMA_VERSION = "structfield-applications-v0.3"

__all__ = [
    "APPLICATION_SCHEMA_VERSION",
    "CurvilinearSegmenter",
    "FringeRestorationLoss",
    "FringeRestorationLossConfig",
    "FringeRestorationLossOutput",
    "FringeRestorer",
    "TextureRestorationLoss",
    "TextureRestorationLossConfig",
    "TextureRestorer",
    "curvilinear_segmentation_loss",
    "soft_dice_loss",
    "texture_restoration_loss",
]
