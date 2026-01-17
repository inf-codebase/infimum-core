"""
Facade pattern for text-to-speech.

Provides simple high-level API hiding complexity of providers and preprocessing.
"""

from typing import Union, Optional
from pathlib import Path
from ...base.providers.factory import ProviderFactory
from ...base.providers.config import ModelConfigBuilder


class Text2Speech:
    """
    Facade for text-to-speech system.
    
    Hides complexity of providers and audio generation.
    """
    
    def __init__(
        self,
        provider: str = "coqui",
        model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize text-to-speech facade.
        
        Args:
            provider: Provider name (coqui, huggingface, etc.)
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
    
    def synthesize(
        self,
        text: str,
        output_path: Optional[Union[str, Path]] = None,
        speaker_id: Optional[int] = None
    ) -> bytes:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            output_path: Optional path to save audio
            speaker_id: Optional speaker ID for multi-speaker models
            
        Returns:
            Audio data as bytes
        """
        # Get model handle
        handle = self._provider.get_model(self._provider.config)
        
        # Synthesize using provider (hidden complexity)
        # This is a placeholder for the actual implementation
        audio_data = b""  # Placeholder
        
        # Save if path provided
        if output_path:
            with open(output_path, "wb") as f:
                f.write(audio_data)
        
        return audio_data
