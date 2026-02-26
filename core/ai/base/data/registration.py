"""
Unified registration for data loaders.

Provides a single entry point that registers a loader in both the
LoaderFactory (for creation by name) and the LoaderRegistry (for
metadata search/discovery), keeping them in sync.

Recommended usage::

    from core.ai.base.data.registration import register_loader, unregister_loader
    from core.ai.base.data.registry import LoaderMetadata

    register_loader(
        loader_name="image",
        loader_class=ImageLoader,
        metadata=LoaderMetadata(
            data_type="image",
            loader_name="image",
            supported_formats={"jpg", "png", "bmp"},
            description="Loads image files",
        ),
    )
"""

from typing import Type

from .base import BaseLoader
from .factory import LoaderFactory
from .registry import LoaderRegistry, LoaderMetadata


def register_loader(
    loader_name: str,
    loader_class: Type[BaseLoader],
    metadata: LoaderMetadata,
) -> None:
    """Register a loader for both creation and discovery.

    This is the **recommended** way to register a new data loader.
    It atomically updates both:

    * ``LoaderFactory`` – so the loader can be created by name via
      ``LoaderFactory.create(loader_name)``.
    * ``LoaderRegistry`` – so the loader can be discovered/searched via
      ``LoaderRegistry.search(data_type=..., format=...)``.

    Args:
        loader_name: Unique loader name (used as key in both systems).
        loader_class: The concrete ``BaseLoader`` subclass.
        metadata: A ``LoaderMetadata`` instance describing the loader.
    """
    LoaderFactory._registry[loader_name] = loader_class
    LoaderRegistry._loaders[loader_name] = metadata


def unregister_loader(loader_name: str) -> None:
    """Remove a loader from both Factory and Registry.

    Args:
        loader_name: The loader name to remove.
    """
    LoaderFactory.unregister(loader_name)
    # LoaderRegistry doesn't have an unregister method, so we remove directly
    if loader_name in LoaderRegistry._loaders:
        del LoaderRegistry._loaders[loader_name]
