"""Speech module for speech-to-text and text-to-speech."""

from .providers import WhisperProvider
from ..base.providers import ProviderFactory

# Register Whisper provider
ProviderFactory.register("speech", "whisper", WhisperProvider)

__all__ = ["WhisperProvider"]
