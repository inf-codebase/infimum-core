"""
MedASR provider for speech-to-text.

Uses Google Health AI's MedASR model for medical transcription.
"""

from typing import Optional
import numpy as np
from ...base.providers import (
    BaseProvider,
    ModelConfig,
    ModelHandle,
    ProviderMetadata,
    ProviderRegistry,
)
from .medasr_repository import MedASRRepository
from .transcription_entity import TranscriptionEntity


class MedASRProvider(BaseProvider):
    """
    MedASR provider for speech-to-text transcription.
    
    Supports HuggingFace model identifiers (e.g., "google/medasr").
    Wraps MedASRRepository to provide a provider-based interface.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize MedASR provider.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self._repository = None
    
    def _validate_config(self, config: ModelConfig) -> None:
        """
        Validate MedASR configuration.
        
        For MedASR, model_name (HuggingFace identifier) is required.
        model_path can also be used but model_name is preferred.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        # MedASR uses HuggingFace model identifiers (e.g., "google/medasr")
        # Prefer model_name, but accept model_path as fallback
        model_identifier = config.model_name or config.model_path
        
        if not model_identifier:
            raise ValueError(
                "Either model_name or model_path is required for MedASR. "
                "Use model_name for HuggingFace model identifiers (e.g., 'google/medasr') "
                "or model_path for custom model paths."
            )
    
    def load_model(self, config: ModelConfig) -> ModelHandle:
        """
        Load MedASR model by initializing MedASRRepository.
        
        Args:
            config: Model configuration
            
        Returns:
            ModelHandle: Handle to loaded MedASR repository
        """
        # Determine model identifier
        # Prefer model_name (HuggingFace identifier), fallback to model_path
        model_name = config.model_name if config.model_name else config.model_path
        
        # Get device from config or use default
        device = config.device
        
        # Get token from extra_params if provided
        token = config.extra_params.get("token") if config.extra_params else None
        
        # Initialize MedASRRepository (lazy initialization happens in transcribe)
        repository = MedASRRepository(
            model_name=model_name,
            device=device,
            token=token
        )
        
        # Store repository reference
        self._repository = repository
        
        # Create handle with repository
        handle = ModelHandle(
            model=repository,  # Store repository as the model
            config=config,
            metadata={
                "model_name": model_name,
                "device": device or "cpu",
                "provider": "medasr"
            }
        )
        
        return handle
    
    def unload_model(self, handle: ModelHandle) -> None:
        """
        Unload MedASR model.
        
        Args:
            handle: Model handle to unload
        """
        # MedASRRepository doesn't have explicit cleanup, but we can clear references
        repository = handle.model
        if hasattr(repository, 'model') and repository.model is not None:
            # Move model to CPU if on GPU to free GPU memory
            try:
                if hasattr(repository.model, 'cpu'):
                    repository.model.cpu()
            except Exception:
                pass
        
        self._repository = None
    
    def transcribe(
        self,
        handle: ModelHandle,
        audio_array: np.ndarray,
        sample_rate: int = 16000,
        return_confidences: bool = True,
        return_timestamps: bool = False,
        **kwargs,
    ) -> TranscriptionEntity:
        """
        Transcribe audio using MedASR model.

        Args:
            handle: Model handle containing MedASRRepository
            audio_array: Audio array (mono, 16kHz)
            sample_rate: Sample rate of audio (default: 16000)
            return_confidences: Whether to return confidence scores
            return_timestamps: Whether to return timestamps
            **kwargs: Additional parameters (ignored for MedASR)

        Returns:
            TranscriptionEntity with transcript and confidence scores
        """
        repository = handle.model

        # Use repository's transcribe method
        return repository.transcribe(
            audio_array=audio_array,
            sample_rate=sample_rate,
            return_confidences=return_confidences,
            return_timestamps=return_timestamps,
        )


# Register MedASR provider in the unified registry
ProviderRegistry.register(
    model_type="speech",
    provider_name="medasr",
    provider_class=MedASRProvider,
    metadata=ProviderMetadata(
        model_type="speech",
        provider_name="medasr",
        capabilities={"speech_to_text", "transcription", "medical"},
        description=(
            "Google Health MedASR provider for medical speech-to-text transcription"
        ),
        version="1.0.0",
    ),
)
