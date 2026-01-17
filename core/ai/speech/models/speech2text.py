"""
Facade pattern for speech-to-text.

Provides simple high-level API hiding complexity of providers and preprocessing.
"""

from typing import Union, Optional
from pathlib import Path
from ...base.providers.factory import ProviderFactory
from ...base.providers.config import ModelConfigBuilder
from ...base.data.factory import LoaderFactory
from ...base.preprocessing.factory import TransformFactory


class Speech2Text:
    """
    Facade for speech-to-text system.
    
    Hides complexity of providers, data loaders, and preprocessing.
    """
    
    def __init__(
        self,
        provider: str = "whisper",
        model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize speech-to-text facade.
        
        Args:
            provider: Provider name (whisper, huggingface, etc.)
            model_path: Path to model (optional)
            device: Device to use (optional)
        """
        # Build configuration
        builder = ModelConfigBuilder()
        builder.with_model_type("speech")
        builder.with_provider(provider)
        
        if model_path:
            builder.with_model_path(model_path)
        if device:
            builder.with_device(device)
        
        config = builder.build()
        
        # Create provider (hidden complexity)
        self._provider = ProviderFactory.create("speech", provider, config)
        
        # Create data loader (hidden complexity)
        self._audio_loader = LoaderFactory.create("audio")
        
        # Create preprocessing pipeline (hidden complexity)
        self._pipeline = TransformFactory.create_pipeline(["audio_normalize"])
    
    def transcribe(
        self,
        audio_source: Union[str, Path, bytes],
        language: Optional[str] = None
    ) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio_source: Audio file path, bytes, or stream
            language: Optional language code
            
        Returns:
            Transcribed text
        """
        # Load audio (hidden complexity)
        audio_item = self._audio_loader.load(audio_source)
        
        # Preprocess (hidden complexity)
        audio_item = self._pipeline.apply(audio_item)
        
        # Transcribe using provider (hidden complexity)
        handle = self._provider.get_model(self._provider.config)
        # Provider-specific transcription would be called here
        # This is a placeholder for the actual implementation
        
        return "Transcribed text"  # Placeholder
    
    def transcribe_batch(
        self,
        audio_sources: list[Union[str, Path, bytes]],
        language: Optional[str] = None
    ) -> list[str]:
        """
        Transcribe multiple audio files.
        
        Args:
            audio_sources: List of audio sources
            language: Optional language code
            
        Returns:
            List of transcribed texts
        """
        return [self.transcribe(source, language) for source in audio_sources]
