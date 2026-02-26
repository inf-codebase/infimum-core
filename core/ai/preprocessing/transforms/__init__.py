"""Transform implementations."""

from .image import ResizeTransform, NormalizeTransform, AugmentTransform
from .text import TokenizeTransform, CleanTransform, NormalizeTextTransform
from .audio import ResampleTransform, NormalizeAudioTransform

from ...base.preprocessing.registration import register_transform
from ...base.preprocessing.registry import TransformMetadata

# Register transforms (unified: updates both Factory and Registry)
register_transform(
    "resize",
    ResizeTransform,
    TransformMetadata(
        transform_name="resize",
        data_type="image",
        description="Resize images to target dimensions",
    ),
)
register_transform(
    "normalize",
    NormalizeTransform,
    TransformMetadata(
        transform_name="normalize",
        data_type="image",
        description="Normalize image pixel values",
    ),
)
register_transform(
    "augment",
    AugmentTransform,
    TransformMetadata(
        transform_name="augment",
        data_type="image",
        description="Augment images with random transformations",
    ),
)
register_transform(
    "tokenize",
    TokenizeTransform,
    TransformMetadata(
        transform_name="tokenize",
        data_type="text",
        description="Tokenize text into tokens",
    ),
)
register_transform(
    "clean",
    CleanTransform,
    TransformMetadata(
        transform_name="clean",
        data_type="text",
        description="Clean text by removing noise",
    ),
)
register_transform(
    "normalize_text",
    NormalizeTextTransform,
    TransformMetadata(
        transform_name="normalize_text",
        data_type="text",
        description="Normalize text casing and whitespace",
    ),
)
register_transform(
    "resample",
    ResampleTransform,
    TransformMetadata(
        transform_name="resample",
        data_type="audio",
        description="Resample audio to target sample rate",
    ),
)
register_transform(
    "normalize_audio",
    NormalizeAudioTransform,
    TransformMetadata(
        transform_name="normalize_audio",
        data_type="audio",
        description="Normalize audio amplitude",
    ),
)

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
