"""
Multimodal loader implementation.

Strategy pattern: Implements BaseLoader for multimodal data (image + text, etc.).
"""

from typing import Union, Dict, Any
from pathlib import Path
from ...core.data.base import BaseLoader
from ...core.data.item import DataItem
from .image import ImageLoader
from .text import TextLoader
from .audio import AudioLoader


class MultimodalLoader(BaseLoader):
    """
    Multimodal data loader.
    
    Loads and combines multiple data types (image + text, etc.).
    """
    
    def __init__(self):
        """Initialize multimodal loader with sub-loaders."""
        super().__init__()
        self._image_loader = ImageLoader()
        self._text_loader = TextLoader()
        self._audio_loader = AudioLoader()
    
    def _load(self, source: Union[str, Path, Dict[str, Any]]) -> DataItem:
        """
        Load multimodal data.
        
        Args:
            source: Multimodal source (dict with keys like 'image', 'text', 'audio')
            
        Returns:
            DataItem: Loaded multimodal data
        """
        if isinstance(source, dict):
            # Load each modality
            modalities = {}
            metadata = {}
            
            if 'image' in source:
                image_item = self._image_loader.load(source['image'])
                modalities['image'] = image_item.data
                metadata['image'] = image_item.metadata
            
            if 'text' in source:
                text_item = self._text_loader.load(source['text'])
                modalities['text'] = text_item.data
                metadata['text'] = text_item.metadata
            
            if 'audio' in source:
                audio_item = self._audio_loader.load(source['audio'])
                modalities['audio'] = audio_item.data
                metadata['audio'] = audio_item.metadata
            
            return DataItem(
                data=modalities,
                data_type="multimodal",
                metadata=metadata
            )
        else:
            raise ValueError(f"Multimodal source must be a dict, got: {type(source)}")
