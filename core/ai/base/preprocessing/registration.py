"""
Unified registration for transforms.

Provides a single entry point that registers a transform in both the
TransformFactory (for creation by name) and the TransformRegistry (for
metadata search/discovery), keeping them in sync.

Recommended usage::

    from core.ai.base.preprocessing.registration import register_transform
    from core.ai.base.preprocessing.registry import TransformMetadata

    register_transform(
        transform_name="resize",
        transform_class=ResizeTransform,
        metadata=TransformMetadata(
            transform_name="resize",
            data_type="image",
            description="Resize images to target dimensions",
        ),
    )
"""

from typing import Type

from .base import BaseTransform
from .factory import TransformFactory
from .registry import TransformRegistry, TransformMetadata


def register_transform(
    transform_name: str,
    transform_class: Type[BaseTransform],
    metadata: TransformMetadata,
) -> None:
    """Register a transform for both creation and discovery.

    This is the **recommended** way to register a new transform.
    It atomically updates both:

    * ``TransformFactory`` – so the transform can be created by name via
      ``TransformFactory.create(transform_name)``.
    * ``TransformRegistry`` – so the transform can be discovered/searched via
      ``TransformRegistry.search(data_type=...)``.

    Args:
        transform_name: Unique transform name (used as key in both systems).
        transform_class: The concrete ``BaseTransform`` subclass.
        metadata: A ``TransformMetadata`` instance describing the transform.
    """
    TransformFactory._registry[transform_name] = transform_class
    TransformRegistry._transforms[transform_name] = metadata


def unregister_transform(transform_name: str) -> None:
    """Remove a transform from both Factory and Registry.

    Args:
        transform_name: The transform name to remove.
    """
    TransformFactory.unregister(transform_name)
    if transform_name in TransformRegistry._transforms:
        del TransformRegistry._transforms[transform_name]
