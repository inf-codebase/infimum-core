"""Data loader implementations."""

from .image import ImageLoader
from .text import TextLoader
from .audio import AudioLoader
from .video import VideoLoader
from .multimodal import MultimodalLoader

from ...base.data.factory import LoaderFactory

# Register loaders
LoaderFactory.register("image", ImageLoader)
LoaderFactory.register("text", TextLoader)
LoaderFactory.register("audio", AudioLoader)
LoaderFactory.register("video", VideoLoader)
LoaderFactory.register("multimodal", MultimodalLoader)

__all__ = [
    "ImageLoader",
    "TextLoader",
    "AudioLoader",
    "VideoLoader",
    "MultimodalLoader",
]
