"""Speech module for speech-to-text and text-to-speech."""

from ..providers.speech.whisper_provider import WhisperProvider
from ..providers.speech.medasr_provider import MedASRProvider
from ..base.providers.registration import register_provider
from ..base.providers.registry import ProviderMetadata

# Register providers (unified: updates both Factory and Registry)
register_provider(
    "speech",
    "whisper",
    WhisperProvider,
    ProviderMetadata(
        model_type="speech",
        provider_name="whisper",
        capabilities={"speech_to_text", "transcription"},
        description="OpenAI Whisper speech-to-text provider",
    ),
)
register_provider(
    "speech",
    "medasr",
    MedASRProvider,
    ProviderMetadata(
        model_type="speech",
        provider_name="medasr",
        capabilities={"speech_to_text", "medical_transcription"},
        description="MedASR medical speech recognition provider",
    ),
)

__all__ = [
    "WhisperProvider",
    "MedASRProvider",
]
