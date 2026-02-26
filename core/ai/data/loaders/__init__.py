"""Data loader implementations."""

from .image import ImageLoader
from .text import TextLoader
from .audio import AudioLoader
from .video import VideoLoader
from .multimodal import MultimodalLoader

from ...base.data.registration import register_loader
from ...base.data.registry import LoaderMetadata

# Register loaders (unified: updates both Factory and Registry)
register_loader(
    "image",
    ImageLoader,
    LoaderMetadata(
        data_type="image",
        loader_name="image",
        supported_formats={"jpg", "jpeg", "png", "bmp", "tiff", "webp"},
        description="Loads image files",
    ),
)
register_loader(
    "text",
    TextLoader,
    LoaderMetadata(
        data_type="text",
        loader_name="text",
        supported_formats={"txt", "csv", "json", "xml"},
        description="Loads text files",
    ),
)
register_loader(
    "audio",
    AudioLoader,
    LoaderMetadata(
        data_type="audio",
        loader_name="audio",
        supported_formats={"wav", "mp3", "flac", "ogg"},
        description="Loads audio files",
    ),
)
register_loader(
    "video",
    VideoLoader,
    LoaderMetadata(
        data_type="video",
        loader_name="video",
        supported_formats={"mp4", "avi", "mkv", "mov"},
        description="Loads video files",
    ),
)
register_loader(
    "multimodal",
    MultimodalLoader,
    LoaderMetadata(
        data_type="multimodal",
        loader_name="multimodal",
        supported_formats={"jpg", "png", "mp4", "txt", "wav"},
        description="Loads multimodal data",
    ),
)

__all__ = [
    "ImageLoader",
    "TextLoader",
    "AudioLoader",
    "VideoLoader",
    "MultimodalLoader",
]
