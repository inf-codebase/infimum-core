"""
Facade pattern for speech-to-text.

Provides simple high-level API hiding complexity of providers and preprocessing.
"""

from typing import Union, Optional
from pathlib import Path
from ...base.providers.factory import ProviderFactory
from ...base.providers.config import ModelConfigBuilder

# Optional factories - may not be implemented yet
try:
    from ...base.data.factory import LoaderFactory
except ImportError:
    LoaderFactory = None

try:
    from ...base.preprocessing.factory import TransformFactory
except ImportError:
    TransformFactory = None


class Speech2Text:
    """
    Facade for speech-to-text system.
    
    Hides complexity of providers, data loaders, and preprocessing.
    """
    
    def __init__(
        self,
        provider: str = "whisper",
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize speech-to-text facade.
        
        Args:
            provider: Provider name (whisper, huggingface, etc.)
            model_path: Path to model file (optional, for custom models)
            model_name: Model name (optional, for Whisper: tiny, base, small, medium, large)
            device: Device to use (optional, e.g., "cpu", "cuda")
        """
        # Build configuration
        builder = ModelConfigBuilder()
        builder.with_model_type("speech")
        builder.with_provider(provider)
        
        # For Whisper, model_name can be used instead of model_path
        # Since Whisper's load_model() accepts both names and paths, we can use model_name as model_path
        if model_name:
            builder.with_model_name(model_name)
            # Set model_path to model_name for validation (Whisper accepts both)
            builder.with_model_path(model_name)
        elif model_path:
            builder.with_model_path(model_path)
        else:
            # Default to "base" for Whisper if nothing specified
            if provider == "whisper":
                default_model = "base"
                builder.with_model_name(default_model)
                builder.with_model_path(default_model)
            else:
                builder.with_model_path("")  # Will be validated by provider
        
        if device:
            builder.with_device(device)
        
        config = builder.build()
        
        # Create provider (hidden complexity)
        self._provider = ProviderFactory.create("speech", provider, config)
        
        # Create data loader (hidden complexity) - optional, may not be implemented yet
        if LoaderFactory:
            try:
                self._audio_loader = LoaderFactory.create("audio")
            except (AttributeError, ValueError, NotImplementedError):
                self._audio_loader = None
        else:
            self._audio_loader = None
        
        # Create preprocessing pipeline (hidden complexity) - optional, may not be implemented yet
        if TransformFactory:
            try:
                self._pipeline = TransformFactory.create_pipeline(["audio_normalize"])
            except (AttributeError, ValueError, NotImplementedError):
                self._pipeline = None
        else:
            self._pipeline = None
    
    def transcribe(
        self,
        audio_source: Union[str, Path, bytes],
        language: Optional[str] = None
    ) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio_source: Audio file path, bytes, or stream
            language: Optional language code (e.g., "en", "es")
            
        Returns:
            Transcribed text
        """
        # Convert Path to string if needed
        if isinstance(audio_source, Path):
            audio_path = str(audio_source)
        elif isinstance(audio_source, bytes):
            # For bytes, we'd need to save to temp file or handle differently
            # For now, assume it's a file path string
            raise ValueError("Bytes input not yet supported. Please provide a file path.")
        else:
            audio_path = audio_source
        
        # Load audio (hidden complexity) - optional preprocessing
        if self._audio_loader:
            audio_item = self._audio_loader.load(audio_source)
        
        # Preprocess (hidden complexity) - optional preprocessing
        if self._pipeline:
            audio_item = self._pipeline.apply(audio_item)
        
        # Get model handle
        handle = self._provider.get_model(self._provider.config)
        
        # Transcribe using provider
        # Check if provider has transcribe method (Whisper-specific)
        if hasattr(self._provider, 'transcribe'):
            return self._provider.transcribe(handle, audio_path, language=language)
        else:
            # Fallback: use model directly if it's a Whisper model
            model = handle.model
            if hasattr(model, 'transcribe'):
                result = model.transcribe(audio_path, language=language)
                return result.get("text", "")
            else:
                raise NotImplementedError(
                    f"Provider {self._provider.__class__.__name__} does not support transcription"
                )
    
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
