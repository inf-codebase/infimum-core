"""
Whisper provider for speech-to-text.

Uses OpenAI's Whisper library for transcription.
"""

from typing import Optional
from ...base.providers.base import BaseProvider, ModelHandle
from ...base.providers.config import ModelConfig
from ....engine.package_utils import ensure_package_installed


class WhisperProvider(BaseProvider):
    """
    Whisper provider for speech-to-text transcription.
    
    Supports Whisper model names: tiny, base, small, medium, large
    or custom model paths.
    """
    
    # Valid Whisper model names
    VALID_MODEL_NAMES = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}
    
    @classmethod
    def _ensure_whisper_installed(cls) -> None:
        """
        Ensure Whisper library is installed, auto-install if missing.
        
        Uses the common package installation utility from core.engine.
        
        Raises:
            ImportError: If installation fails or import still fails after installation
        """
        ensure_package_installed(
            package_name="whisper",
            install_name="openai-whisper",
            prefer_uv=True
        )
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize Whisper provider.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self._model = None
    
    def _validate_config(self, config: ModelConfig) -> None:
        """
        Validate Whisper configuration.
        
        For Whisper, either model_path or model_name must be provided.
        If model_name is provided, it should be a valid Whisper model name.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        # For Whisper, model_name can be used instead of model_path
        # But model_path is required by base validation, so we check if we have either
        model_identifier = config.model_name or config.model_path
        
        if not model_identifier:
            raise ValueError(
                "Either model_path or model_name is required for Whisper. "
                "Use model_name for standard models (tiny, base, small, medium, large) "
                "or model_path for custom model files."
            )
        
        # Validate model_name if provided (and it's a standard model name)
        if config.model_name and config.model_name not in self.VALID_MODEL_NAMES:
            # If it's not a standard name, it might be a custom path - that's OK
            # Only validate if it looks like a model name (no path separators)
            if "/" not in config.model_name and "\\" not in config.model_name:
                raise ValueError(
                    f"Invalid Whisper model name: {config.model_name}. "
                    f"Valid names: {', '.join(sorted(self.VALID_MODEL_NAMES))}"
                )
    
    def load_model(self, config: ModelConfig) -> ModelHandle:
        """
        Load Whisper model.
        
        Args:
            config: Model configuration
            
        Returns:
            ModelHandle: Handle to loaded Whisper model
        """
        # Ensure Whisper is installed (will auto-install if needed)
        self._ensure_whisper_installed()
        
        import whisper
        
        # Determine model identifier
        # Use model_name if provided (e.g., "small"), otherwise use model_path
        model_identifier = config.model_name if config.model_name else config.model_path
        
        # Load the model
        model = whisper.load_model(model_identifier, device=config.device)
        
        # Store model reference for cleanup
        self._model = model
        
        # Create handle
        handle = ModelHandle(
            model=model,
            config=config,
            metadata={
                "model_name": model_identifier,
                "device": config.device or "cpu",
                "provider": "whisper"
            }
        )
        
        return handle
    
    def unload_model(self, handle: ModelHandle) -> None:
        """
        Unload Whisper model.
        
        Args:
            handle: Model handle to unload
        """
        # Whisper models are Python objects, Python GC will handle cleanup
        # But we can explicitly clear the reference
        if hasattr(handle.model, 'cpu'):
            # Move to CPU if on GPU to free GPU memory
            try:
                handle.model.cpu()
            except Exception:
                pass
        
        self._model = None
    
    def transcribe(
        self,
        handle: ModelHandle,
        audio_path: str,
        language: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Transcribe audio using Whisper model.
        
        Args:
            handle: Model handle
            audio_path: Path to audio file
            language: Optional language code (e.g., "en", "es")
            **kwargs: Additional Whisper transcribe parameters
            
        Returns:
            Transcribed text
        """
        model = handle.model
        
        # Prepare transcribe options
        transcribe_options = {}
        if language:
            transcribe_options["language"] = language
        
        # Add any extra parameters from config
        if handle.config.extra_params:
            transcribe_options.update(handle.config.extra_params)
        
        # Override with kwargs
        transcribe_options.update(kwargs)
        
        # Transcribe
        result = model.transcribe(audio_path, **transcribe_options)
        
        # Return text
        return result.get("text", "")
