"""
DeepSeek-OCR-2 Provider Adapter.

Adapts DeepSeek-OCR-2 model to BaseProvider interface for OCR capabilities.
"""

from typing import Optional

from core.ai.base import ProviderRegistry
from ...base.providers import BaseProvider, ModelConfig, ModelHandle
from ...base.providers.registry import ProviderMetadata


class DeepSeekOCRProviderAdapter(BaseProvider):
    """
    Adapter for DeepSeek-OCR-2 model.

    Capabilities: OCR, document parsing, markdown conversion.
    Uses Visual Causal Flow for accurate text extraction.
    """

    DEFAULT_MODEL_PATH = "deepseek-ai/DeepSeek-OCR-2"

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize adapter.

        Args:
            config: Optional model configuration
        """
        super().__init__(config)
        self._model = None
        self._tokenizer = None

    def load_model(self, config: ModelConfig) -> ModelHandle:
        """
        Load DeepSeek-OCR-2 model.

        Args:
            config: Model configuration

        Returns:
            ModelHandle: Handle containing model components
        """
        import torch
        from transformers import AutoModel, AutoTokenizer

        model_path = config.model_path or self.DEFAULT_MODEL_PATH
        device = config.device or "cuda"

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        # Determine dtype
        dtype = torch.bfloat16
        dtype_param = config.extra_params.get("dtype", "bfloat16")
        if dtype_param == "float16":
            dtype = torch.float16
        elif dtype_param == "float32":
            dtype = torch.float32

        # Determine attention implementation
        attn_impl = config.extra_params.get("attn_implementation", "flash_attention_2")

        # Load model with fallback for Flash Attention
        try:
            self._model = AutoModel.from_pretrained(
                model_path,
                _attn_implementation=attn_impl,
                trust_remote_code=True,
                use_safetensors=True,
            )
        except ImportError:
            if attn_impl == "flash_attention_2":
                print("⚠️ FlashAttention not found, falling back to eager attention")
                self._model = AutoModel.from_pretrained(
                    model_path,
                    _attn_implementation="eager",
                    trust_remote_code=True,
                    use_safetensors=True,
                )
            else:
                raise

        # Move to device and set eval mode
        if device == "cuda" and torch.cuda.is_available():
            self._model = self._model.eval().cuda().to(dtype)
        else:
            self._model = self._model.eval().to(dtype)

        # Return handle
        return ModelHandle(
            model={
                "model": self._model,
                "tokenizer": self._tokenizer,
            },
            config=config,
            metadata={
                "model_name": "DeepSeek-OCR-2",
                "model_path": model_path,
                "device": device,
                "dtype": str(dtype),
                "capabilities": ["ocr", "document_parsing", "markdown_conversion"],
                "default_prompts": {
                    "document": "<image>\n<|grounding|>Convert the document to markdown.",
                    "free_ocr": "<image>\nFree OCR.",
                },
                "inference_params": {
                    "base_size": 1024,
                    "image_size": 768,
                    "crop_mode": True,
                },
            },
        )

    def unload_model(self, handle: ModelHandle) -> None:
        """
        Unload model and free memory.

        Args:
            handle: Model handle to unload
        """
        if self._model:
            self._model.cpu()
            del self._model
            self._model = None

        if self._tokenizer:
            del self._tokenizer
            self._tokenizer = None

        # Clear CUDA cache
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def infer(
        self,
        handle: ModelHandle,
        image_path: str,
        prompt: Optional[str] = None,
        output_path: Optional[str] = None,
        mode: str = "document",
        **kwargs,
    ) -> str:
        """
        Run OCR inference on an image.

        Args:
            handle: Model handle
            image_path: Path to input image
            prompt: Custom prompt (uses default based on mode if not provided)
            output_path: Optional output directory for saving results
            mode: OCR mode - "document" (with layout) or "free" (without layout)
            **kwargs: Additional inference parameters
                - base_size: Base image size (default: 1024)
                - image_size: Crop image size (default: 768)
                - crop_mode: Whether to use crop mode (default: True)
                - save_results: Whether to save results to file (default: False)

        Returns:
            str: OCR result text or markdown
        """
        model = handle.get("model")
        tokenizer = handle.get("tokenizer")

        # Use default prompt if not provided
        if prompt is None:
            default_prompts = handle.metadata.get("default_prompts", {})
            if mode == "free":
                prompt = default_prompts.get("free_ocr", "<image>\nFree OCR.")
            else:
                prompt = default_prompts.get(
                    "document",
                    "<image>\n<|grounding|>Convert the document to markdown.",
                )

        # Get default inference params and merge with kwargs
        default_params = handle.metadata.get("inference_params", {}).copy()
        params = {**default_params, **kwargs}

        # Run inference using temp directory to capture output
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.infer(
                tokenizer,
                prompt=prompt,
                image_file=image_path,
                output_path=tmp_dir,
                base_size=params.get("base_size", 1024),
                image_size=params.get("image_size", 768),
                crop_mode=params.get("crop_mode", True),
                save_results=True,  # Force save to capture output
            )

            # Find generated file (usually matches image stem + .md or similar)
            # We just take the first file found in tmp_dir
            files = list(Path(tmp_dir).glob("*"))
            if files:
                # Read content
                result = files[0].read_text(encoding="utf-8")
                # If optional output_path was provided in kwargs, copy it there?
                # But BaseProvider.infer() is expected to return str.
                # If caller wants to save, they can write the result.

                # Copy to output_path if specified
                if output_path:
                    out_dir = Path(output_path)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    dest_file = out_dir / files[0].name
                    dest_file.write_text(result, encoding="utf-8")

                return result

            return ""


ProviderRegistry.register(
    "vlm",
    "deepseek-ocr",
    DeepSeekOCRProviderAdapter,
    ProviderMetadata(
        model_type="vlm",
        provider_name="deepseek-ocr",
        capabilities={"ocr", "document_parsing", "markdown_conversion", "multimodal"},
        description="DeepSeek-OCR-2: Visual Causal Flow for document OCR and parsing",
        version="2.0.0",
        requirements=["transformers>=4.51.1", "flash-attn>=2.7.3", "torch>=2.6.0"],
    ),
)
