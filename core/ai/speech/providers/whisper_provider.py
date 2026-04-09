"""
Whisper provider for speech-to-text.

Acts as a generic loader that supports:
1. OpenAI's Whisper library (Local CPU/GPU inference)
2. OpenAI API / proxy endpoints
3. Google GenAI (Gemini)
"""

import json
import logging
import os
from typing import Optional, Dict, Any, Union
from ...base.providers import (
    BaseProvider,
    ModelConfig,
    ModelHandle,
    ProviderMetadata,
    ProviderRegistry,
)

logger = logging.getLogger(__name__)

class WhisperProvider(BaseProvider):
    """
    Whisper provider for speech-to-text transcription.
    
    Supports local Whisper models or Cloud API Models (OpenAI/Google).
    """
    
    # Valid Whisper model names for local inference
    VALID_MODEL_NAMES = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}
    
    @classmethod
    def _ensure_whisper_installed(cls) -> None:
        from core.engine.package_utils import ensure_package_installed
        ensure_package_installed(
            package_name="whisper",
            install_name="openai-whisper",
            prefer_uv=True
        )
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(config)
        self._model = None
    
    def _validate_config(self, config: ModelConfig) -> None:
        api_provider = config.extra_params.get("api_provider") or os.getenv("AI_PROVIDER")
        if api_provider:
            # If using API, we defer validation of model_name as any string can be passed to API
            return
            
        model_identifier = config.model_name or config.model_path
        if not model_identifier:
            raise ValueError(
                "Either model_path or model_name is required for local Whisper. "
                "Or set api_provider in extra_params for API-based transcription."
            )
        
        if config.model_name and config.model_name not in self.VALID_MODEL_NAMES:
            if "/" not in config.model_name and "\\" not in config.model_name:
                raise ValueError(
                    f"Invalid local Whisper model name: {config.model_name}. "
                    f"Valid names: {', '.join(sorted(self.VALID_MODEL_NAMES))}"
                )
    
    def load_model(self, config: ModelConfig) -> ModelHandle:
        api_provider = config.extra_params.get("api_provider") or os.getenv("AI_PROVIDER")
        
        if api_provider:
            # Setup API Clients
            api_key = config.extra_params.get("api_key")
            base_url = config.extra_params.get("base_url") or os.getenv("OPENAI_BASE_URL")
            model_name = config.model_name or config.extra_params.get("model", "whisper-1")
            
            client = None
            if api_provider == "google":
                from google import genai
                api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("API key is required for Google provider (GEMINI_API_KEY)")
                client = genai.Client(api_key=api_key)
            elif api_provider in ("openai", "local"):
                from openai import OpenAI
                api_key = api_key or os.getenv("OPENAI_API_KEY")
                client_kwargs = {}
                if api_key:
                    client_kwargs["api_key"] = api_key
                else:
                    client_kwargs["api_key"] = "dummy-key-for-local"
                    
                if (api_provider == "local" or not api_key) and base_url:
                    client_kwargs["base_url"] = base_url
                client = OpenAI(**client_kwargs)
            else:
                raise ValueError(f"Unsupported API provider: {api_provider}")
                
            return ModelHandle(
                model=client,
                config=config,
                metadata={
                    "model_name": model_name,
                    "device": "api",
                    "provider": f"api_{api_provider}"
                }
            )

        # Ensure Whisper is installed (will auto-install if needed)
        self._ensure_whisper_installed()
        import whisper
        
        model_identifier = config.model_name if config.model_name else config.model_path
        model = whisper.load_model(model_identifier, device=config.device)
        self._model = model
        
        return ModelHandle(
            model=model,
            config=config,
            metadata={
                "model_name": model_identifier,
                "device": config.device or "cpu",
                "provider": "whisper_local"
            }
        )
    
    def unload_model(self, handle: ModelHandle) -> None:
        provider_type = handle.metadata.get("provider", "")
        if provider_type.startswith("api_"):
            return
            
        if hasattr(handle.model, 'cpu'):
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
        **kwargs,
    ) -> Union[str, Dict[str, Any]]:
        """
        Transcribe audio using the configured model/API.
        
        Returns:
            String for basic local transcription (backward compat) 
            or Dict if API/return_dict is used.
        """
        provider_type = handle.metadata.get("provider", "")
        model_name = handle.metadata.get("model_name")
        api_provider = handle.config.extra_params.get("api_provider")
        
        if provider_type == "api_google" or api_provider == "google":
            client = handle.model
            from google.genai import types
            
            logger.info(f"Uploading {audio_path} to Gemini...")
            audio_file = client.files.upload(file=str(audio_path))
            
            default_prompt = (
                "Transcribe the following audio exactly. "
                "Return the result in JSON format with 'text' and 'segments' fields. "
                "Each segment should contain 'start', 'end', and 'text'."
            )
            prompt = kwargs.get("prompt", default_prompt)
            
            response = client.models.generate_content(
                model=model_name,
                contents=[audio_file, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type=kwargs.get("response_mime_type", "application/json"),
                    temperature=kwargs.get("temperature", 0.0)
                )
            )
            try:
                mime_type = kwargs.get("response_mime_type", "application/json")
                if mime_type == "application/json":
                    return json.loads(response.text)
                return response.text
            except json.JSONDecodeError:
                return response.text

        elif provider_type in ("api_openai", "api_local") or api_provider in ("openai", "local"):
            client = handle.model
            with open(audio_path, "rb") as audio_file:
                transcribe_options = {"model": model_name}
                if language:
                    transcribe_options["language"] = language
                # Remove non-api params
                if "return_dict" in kwargs:
                    kwargs.pop("return_dict")
                transcribe_options.update(kwargs)
                
                response = client.audio.transcriptions.create(
                    file=audio_file,
                    **transcribe_options
                )
            
            if hasattr(response, "model_dump"):
                return response.model_dump()
            elif isinstance(response, dict):
                return response
            else:
                return getattr(response, "text", str(response))

        else:
            # Local Inference
            model = handle.model
            transcribe_options = {}
            if language:
                transcribe_options["language"] = language
                
            if handle.config.extra_params:
                filtered_params = {k: v for k, v in handle.config.extra_params.items() 
                                 if k not in ["api_provider", "api_key", "base_url"]}
                transcribe_options.update(filtered_params)

            if "prompt" in kwargs:
                transcribe_options["initial_prompt"] = kwargs.pop("prompt")
                
            return_dict = kwargs.pop("return_dict", False)
            transcribe_options.update(kwargs)
            
            result = model.transcribe(audio_path, **transcribe_options)
            
            if return_dict or kwargs.get("verbose") or kwargs.get("word_timestamps"):
                return result
            return result.get("text", "")


# Register Whisper provider in the unified registry
ProviderRegistry.register(
    model_type="speech",
    provider_name="whisper",
    provider_class=WhisperProvider,
    metadata=ProviderMetadata(
        model_type="speech",
        provider_name="whisper",
        capabilities={"speech_to_text", "transcription"},
        description="Whisper ASR provider (supports Local, OpenAI API, and Google GenAI)",
        version="1.1.0",
    ),
)
