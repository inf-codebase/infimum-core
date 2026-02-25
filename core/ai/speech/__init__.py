"""Speech module for speech-to-text and text-to-speech."""

from .providers.whisper_provider import WhisperProvider
from .providers.medasr_provider import MedASRProvider
from ..base.providers import ProviderFactory

# Register Whisper provider
ProviderFactory.register("speech", "whisper", WhisperProvider)

# Register MedASR provider
ProviderFactory.register("speech", "medasr", MedASRProvider)

__all__ = [
    "WhisperProvider",
    "MedASRProvider",
]
