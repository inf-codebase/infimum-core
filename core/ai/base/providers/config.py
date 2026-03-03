"""
Builder pattern for model configurations.

Allows step-by-step construction of complex model configurations.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class ModelType(str, Enum):
    """Model types."""
    LLM = "llm"
    VLM = "vlm"
    SPEECH = "speech"


@dataclass
class ModelConfig:
    """Model configuration.
    """
    model_type: str
    provider: str
    model_path: Optional[str] = None
    model_base: Optional[str] = None
    model_name: Optional[str] = None
    device: Optional[str] = None
    load_8bit: bool = False
    load_4bit: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate configuration."""
        if not self.model_type:
            raise ValueError("model_type is required")
        if not self.provider:
            raise ValueError("provider is required")
        if not self.model_path and not self.model_name:
            raise ValueError(
                "At least one of model_path or model_name must be provided"
            )

    def __hash__(self) -> int:
        """Make config hashable."""
        return hash((
            self.model_type,
            self.provider,
            self.model_path,
            self.model_name,
            self.model_base,
            self.device,
            self.load_8bit,
            self.load_4bit,
        ))


class ModelConfigBuilder:
    """Builder for model configurations."""
    
    def __init__(self):
        """Initialize builder with empty config."""
        self._config = ModelConfig(
            model_type="",
            provider="",
        )
    
    def with_model_type(self, model_type: str) -> 'ModelConfigBuilder':
        """
        Set model type.
        
        Args:
            model_type: Model type (llm, vlm, speech)
            
        Returns:
            Self for chaining
        """
        self._config.model_type = model_type
        return self
    
    def with_provider(self, provider: str) -> 'ModelConfigBuilder':
        """
        Set provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Self for chaining
        """
        self._config.provider = provider
        return self
    
    def with_model_path(self, path: str) -> 'ModelConfigBuilder':
        """
        Set model path.
        
        Args:
            path: Path to model
            
        Returns:
            Self for chaining
        """
        self._config.model_path = path
        return self
    
    def with_model_base(self, base: str) -> 'ModelConfigBuilder':
        """
        Set model base path.
        
        Args:
            base: Base model path
            
        Returns:
            Self for chaining
        """
        self._config.model_base = base
        return self
    
    def with_model_name(self, name: str) -> 'ModelConfigBuilder':
        """
        Set model name.
        
        Args:
            name: Model name
            
        Returns:
            Self for chaining
        """
        self._config.model_name = name
        return self
    
    def with_device(self, device: str) -> 'ModelConfigBuilder':
        """
        Set device.
        
        Args:
            device: Device (cuda, cpu, mps)
            
        Returns:
            Self for chaining
        """
        self._config.device = device
        return self
    
    def with_quantization(self, bits: int) -> 'ModelConfigBuilder':
        """
        Set quantization.
        
        Args:
            bits: Quantization bits (4 or 8)
            
        Returns:
            Self for chaining
        """
        if bits == 8:
            self._config.load_8bit = True
            self._config.load_4bit = False
        elif bits == 4:
            self._config.load_4bit = True
            self._config.load_8bit = False
        else:
            raise ValueError(f"Unsupported quantization bits: {bits}. Use 4 or 8.")
        return self
    
    def with_temperature(self, temp: float) -> 'ModelConfigBuilder':
        """
        Set temperature.
        
        Args:
            temp: Temperature value
            
        Returns:
            Self for chaining
        """
        self._config.temperature = temp
        return self
    
    def with_max_tokens(self, max_tokens: int) -> 'ModelConfigBuilder':
        """
        Set max tokens.
        
        Args:
            max_tokens: Maximum tokens
            
        Returns:
            Self for chaining
        """
        self._config.max_tokens = max_tokens
        return self
    
    def with_extra_param(self, key: str, value: Any) -> 'ModelConfigBuilder':
        """
        Add extra parameter.
        
        Args:
            key: Parameter key
            value: Parameter value
            
        Returns:
            Self for chaining
        """
        self._config.extra_params[key] = value
        return self
    
    def build(self) -> ModelConfig:
        """
        Build final configuration.
        
        Returns:
            ModelConfig: Built configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        self._config.validate()
        return self._config
