"""
Adapter pattern for existing ModelManager.

Adapts ModelManager singleton to BaseProvider interface.
"""

from typing import Optional
from ...base.providers import BaseProvider, ModelConfig, ModelHandle
from ...base.providers.registry import ProviderMetadata
from ...base.providers.registration import register_provider


class ModelManagerProviderAdapter(BaseProvider):
    """
    Adapter for existing ModelManager singleton.

    Adapts ModelManager to BaseProvider interface.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize adapter.

        Args:
            config: Model configuration
        """
        super().__init__(config)
        self._manager = None

    def load_model(self, config: ModelConfig) -> ModelHandle:
        """
        Adapt ModelManager to BaseProvider interface.

        Args:
            config: Model configuration

        Returns:
            ModelHandle: Handle containing model components
        """
        # Import here to avoid circular dependencies
        # Note: This will work once visual_language_model is moved to vlm/models
        try:
            from ...vlm.models.model_manager import (
                ModelManager,
                ModelConfig as VLMConfig,
            )
        except ImportError:
            # Fallback to existing location
            from ...visual_language_model.model_manager import (
                ModelManager,
                ModelConfig as VLMConfig,
            )

        # Get singleton instance
        self._manager = ModelManager.get_instance()

        # Convert our config to VLM config
        vlm_config = VLMConfig(
            model_path=config.model_path,
            model_base=config.model_base,
            device=config.device,
            load_8bit=config.load_8bit,
            load_4bit=config.load_4bit,
        )

        # Load model using manager
        self._manager.load_model(vlm_config)

        # Get model components
        tokenizer, model, processor, context_len = self._manager.get_model(vlm_config)

        # Wrap in ModelHandle
        return ModelHandle(
            model={
                "tokenizer": tokenizer,
                "model": model,
                "processor": processor,
            },
            config=config,
            metadata={
                "context_len": context_len,
                "manager": "ModelManager",
                "capabilities": ["inference", "batch_inference", "caching"],
            },
        )

    def unload_model(self, handle: ModelHandle) -> None:
        """
        Unload model using manager.

        Args:
            handle: Model handle to unload
        """
        if self._manager:
            # Convert config back to VLM config
            from ...vlm.models.model_manager import ModelConfig as VLMConfig

            vlm_config = VLMConfig(
                model_path=handle.config.model_path,
                model_base=handle.config.model_base,
                device=handle.config.device,
                load_8bit=handle.config.load_8bit,
                load_4bit=handle.config.load_4bit,
            )

            self._manager.unload_model(vlm_config)


# Register adapter (unified: updates both Factory and Registry)
register_provider(
    "vlm",
    "model_manager",
    ModelManagerProviderAdapter,
    ProviderMetadata(
        model_type="vlm",
        provider_name="model_manager",
        capabilities={"inference", "batch_inference", "caching", "singleton"},
        description="Adapter for existing ModelManager singleton",
        version="1.0.0",
    ),
)
