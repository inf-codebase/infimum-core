"""
Image transform implementations.

Chain of Responsibility: Each transform is a link in the processing chain.
"""

from typing import Tuple, Optional
from PIL import Image
import numpy as np
from ...base.preprocessing.base import BaseTransform
from ...base.data.item import DataItem


class ResizeTransform(BaseTransform):
    """Resize image transform."""

    def __init__(self, size: Tuple[int, int], resample: int = Image.BICUBIC):
        """
        Initialize resize transform.

        Args:
            size: Target size (width, height)
            resample: Resampling method
        """
        self.size = size
        self.resample = resample

    def transform(self, data: DataItem) -> DataItem:
        """
        Resize image.

        Args:
            data: Input data item with image

        Returns:
            Transformed data item
        """
        if data.data_type != "image":
            raise ValueError(
                f"ResizeTransform expects image data, got {data.data_type}"
            )

        image = data.data
        if not isinstance(image, Image.Image):
            raise ValueError("Data must be a PIL Image")

        resized = image.resize(self.size, self.resample)
        data.data = resized
        data.set("resized_size", self.size)
        return data


class NormalizeTransform(BaseTransform):
    """Normalize image transform."""

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Initialize normalize transform.

        Args:
            mean: Mean values for normalization
            std: Standard deviation values for normalization
        """
        self.mean = mean
        self.std = std

    def transform(self, data: DataItem) -> DataItem:
        """
        Normalize image.

        Args:
            data: Input data item with image

        Returns:
            Transformed data item
        """
        if data.data_type != "image":
            raise ValueError(
                f"NormalizeTransform expects image data, got {data.data_type}"
            )

        # Convert PIL to numpy
        image = np.array(data.data).astype(np.float32) / 255.0

        # Normalize
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - self.mean[i]) / self.std[i]

        data.data = image
        data.set("normalized", True)
        return data


class AugmentTransform(BaseTransform):
    """Image augmentation transform."""

    def __init__(self, flip_prob: float = 0.5, rotate_range: int = 10):
        """
        Initialize augment transform.

        Args:
            flip_prob: Probability of horizontal flip
            rotate_range: Rotation range in degrees
        """
        self.flip_prob = flip_prob
        self.rotate_range = rotate_range

    def transform(self, data: DataItem) -> DataItem:
        """
        Augment image.

        Args:
            data: Input data item with image

        Returns:
            Transformed data item
        """
        if data.data_type != "image":
            raise ValueError(
                f"AugmentTransform expects image data, got {data.data_type}"
            )

        import random

        image = data.data

        # Random horizontal flip
        if random.random() < self.flip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # Random rotation
        if self.rotate_range > 0:
            angle = random.uniform(-self.rotate_range, self.rotate_range)
            image = image.rotate(angle, resample=Image.BICUBIC, expand=False)

        data.data = image
        data.set("augmented", True)
        return data
