"""
Image loader implementation.

Strategy pattern: Implements BaseLoader for image data.
"""

from typing import Union
from pathlib import Path
from PIL import Image
import numpy as np
from ...core.data.base import BaseLoader
from ...core.data.item import DataItem


class ImageLoader(BaseLoader):
    """
    Image data loader.
    
    Loads images from files, PIL Images, or numpy arrays.
    """
    
    def _load(self, source: Union[str, Path, Image.Image, np.ndarray]) -> DataItem:
        """
        Load image data.
        
        Args:
            source: Image source (path, PIL Image, or numpy array)
            
        Returns:
            DataItem: Loaded image data
        """
        if isinstance(source, (str, Path)):
            # Load from file
            image = Image.open(source).convert('RGB')
            return DataItem(
                data=image,
                data_type="image",
                source=str(source),
                metadata={
                    "format": image.format,
                    "size": image.size,
                    "mode": image.mode,
                }
            )
        elif isinstance(source, Image.Image):
            # Already a PIL Image
            if source.mode != 'RGB':
                source = source.convert('RGB')
            return DataItem(
                data=source,
                data_type="image",
                metadata={
                    "format": source.format,
                    "size": source.size,
                    "mode": source.mode,
                }
            )
        elif isinstance(source, np.ndarray):
            # Numpy array
            if len(source.shape) == 3 and source.shape[2] == 3:
                # BGR to RGB if needed
                image = Image.fromarray(source[..., ::-1] if source.dtype == np.uint8 else source)
            else:
                image = Image.fromarray(source)
            return DataItem(
                data=image.convert('RGB'),
                data_type="image",
                metadata={
                    "shape": source.shape,
                    "dtype": str(source.dtype),
                }
            )
        else:
            raise ValueError(f"Unsupported image source type: {type(source)}")
