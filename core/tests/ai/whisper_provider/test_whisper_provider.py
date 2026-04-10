import pytest
from unittest.mock import patch, MagicMock
from core.ai.base.providers.config import ModelConfig
from core.ai.speech.providers.whisper_provider import WhisperProvider

def test_whisper_provider_openai_api():
    # Test initialization with OpenAI API
    config = ModelConfig(
        model_type="speech",
        provider="whisper",
        model_name="whisper-1",
        extra_params={
            "api_provider": "openai",
            "api_key": "fake-openai-key"
        }
    )
    provider = WhisperProvider(config)
    
    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        
        handle = provider.load_model(config)
        assert handle.metadata.get("provider") == "api_openai"
        assert handle.model == mock_client

        # Test transcription behavior
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {"text": "hello test", "segments": []}
        mock_client.audio.transcriptions.create.return_value = mock_response

        with patch("builtins.open", MagicMock()):
            result = provider.transcribe(handle, "dummy.wav", language="en", response_format="verbose_json")
            
        assert result == {"text": "hello test", "segments": []}
        mock_client.audio.transcriptions.create.assert_called_once()
        _, kwargs = mock_client.audio.transcriptions.create.call_args
        assert kwargs["model"] == "whisper-1"
        assert kwargs["language"] == "en"
        assert kwargs["response_format"] == "verbose_json"

def test_whisper_provider_google_api():
    # Test initialization with Google Gemini API
    config = ModelConfig(
        model_type="speech",
        provider="whisper",
        model_name="gemini-2.5-flash",
        extra_params={
            "api_provider": "google",
            "api_key": "fake-gemini-key"
        }
    )
    provider = WhisperProvider(config)
    
    with patch("google.genai.Client") as mock_google_cls:
        mock_client = MagicMock()
        mock_google_cls.return_value = mock_client
        
        handle = provider.load_model(config)
        assert handle.metadata.get("provider") == "api_google"

        # Mock generating content
        mock_response = MagicMock()
        mock_response.text = '{"text": "gemini test", "segments": []}'
        mock_client.models.generate_content.return_value = mock_response
        
        # Test transcription
        result = provider.transcribe(handle, "dummy.wav", prompt="Return JSON")
        
        assert isinstance(result, dict)
        assert result["text"] == "gemini test"
        mock_client.files.upload.assert_called_once()
        mock_client.models.generate_content.assert_called_once()
