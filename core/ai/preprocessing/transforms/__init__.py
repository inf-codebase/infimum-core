"""Transform implementations."""

from .image import ResizeTransform, NormalizeTransform, AugmentTransform
from .text import TokenizeTransform, CleanTransform, NormalizeTextTransform
from .audio import ResampleTransform, NormalizeAudioTransform

from ...base.preprocessing.factory import TransformFactory

# Register transforms
TransformFactory.register("resize", ResizeTransform)
TransformFactory.register("normalize", NormalizeTransform)
TransformFactory.register("augment", AugmentTransform)
TransformFactory.register("tokenize", TokenizeTransform)
TransformFactory.register("clean", CleanTransform)
TransformFactory.register("normalize_text", NormalizeTextTransform)
TransformFactory.register("resample", ResampleTransform)
TransformFactory.register("normalize_audio", NormalizeAudioTransform)

__all__ = [
    "ResizeTransform",
    "NormalizeTransform",
    "AugmentTransform",
    "TokenizeTransform",
    "CleanTransform",
    "NormalizeTextTransform",
    "ResampleTransform",
    "NormalizeAudioTransform",
]
