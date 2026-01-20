"""
Adapter pattern for existing LLaVA builder.

Adapts load_pretrained_model to BaseProvider interface.
"""

from typing import Optional, Tuple, Any
from ...base.providers import BaseProvider, ModelConfig, ModelHandle
from ...base.providers import ProviderRegistry, ProviderMetadata


class LLaVAProviderAdapter(BaseProvider):
    """
    Adapter for existing LLaVA builder.
    
    Adapts load_pretrained_model to BaseProvider interface.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize adapter.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
    
    def load_model(self, config: ModelConfig) -> ModelHandle:
        """
        Adapt load_pretrained_model to BaseProvider interface.
        
        Args:
            config: Model configuration
            
        Returns:
            ModelHandle: Handle containing model components
        """
        # Import here to avoid circular dependencies
        # Note: This will work once visual_language_model is moved to vlm/models
        try:
            from ...vlm.models.builder import load_pretrained_model
        except ImportError:
            # Fallback to existing location
            from ...visual_language_model.builder import load_pretrained_model
        
        # Extract parameters from config
        model_name = config.model_name or config.model_path.split('/')[-1]
        
        # Use existing builder
        tokenizer, model, processor, context_len = load_pretrained_model(
            model_path=config.model_path,
            model_base=config.model_base,
            model_name=model_name,
            device=config.device or "auto",
            load_8bit=config.load_8bit,
            load_4bit=config.load_4bit,
            **config.extra_params
        )
        
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
                "model_name": model_name,
                "capabilities": ["inference", "batch_inference", "multimodal"]
            }
        )
    
    def unload_model(self, handle: ModelHandle) -> None:
        """
        Cleanup model.
        
        Args:
            handle: Model handle to unload
        """
        model = handle.get("model")
        if model and hasattr(model, 'cpu'):
            model.cpu()
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


# Register adapter
ProviderRegistry.register(
    "vlm-llava",
    ProviderMetadata(
        model_type="vlm",
        provider_name="llava",
        capabilities={"inference", "batch_inference", "multimodal"},
        description="Adapter for existing LLaVA model builder",
        version="1.0.0"
    )
)
