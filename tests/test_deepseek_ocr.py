#!/usr/bin/env python3
"""
Unit tests for DeepSeek-OCR-2 provider.
"""

import sys
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Stub out broken modules that are not needed for this test
sys.modules["core.ai.speech"] = MagicMock()
sys.modules["core.ai.speech.models"] = MagicMock()
sys.modules["core.ai.speech.models.speech2text"] = MagicMock()
sys.modules["core.ai.speech.models.text2speech"] = MagicMock()

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.ai.vlm.providers.deepseek_ocr import DeepSeekOCRProviderAdapter
from core.ai.base.providers import ModelConfig


class TestDeepSeekOCRProviderAdapter:
    """Unit tests for DeepSeekOCRProviderAdapter."""

    @pytest.fixture
    def mock_deps(self):
        """Mock torch and transformers dependencies."""
        with patch("transformers.AutoModel") as mock_model, \
             patch("transformers.AutoTokenizer") as mock_tokenizer, \
             patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.empty_cache"):
            
            # Setup mock model
            model_instance = MagicMock()
            # The code calls AutoModel.from_pretrained
            mock_model.from_pretrained.return_value = model_instance
            model_instance.eval.return_value = model_instance
            model_instance.cuda.return_value = model_instance
            model_instance.to.return_value = model_instance
            
            # Setup mock tokenizer
            tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = tokenizer_instance
            
            yield {
                "model": mock_model,
                "tokenizer": mock_tokenizer,
                "model_instance": model_instance,
                "tokenizer_instance": tokenizer_instance
            }

    def test_load_model(self, mock_deps):
        """Test loading the model with various configurations."""
        adapter = DeepSeekOCRProviderAdapter()
        config = ModelConfig(
            model_type="vlm",
            provider="deepseek-ocr",
            model_path="test-path",
            device="cuda",
            extra_params={"dtype": "float16", "attn_implementation": "eager"}
        )
        
        handle = adapter.load_model(config)
        
        assert handle.metadata["model_path"] == "test-path"
        assert handle.metadata["device"] == "cuda"
        mock_deps["model"].from_pretrained.assert_called_with(
            "test-path",
            _attn_implementation="eager",
            trust_remote_code=True,
            use_safetensors=True
        )
        mock_deps["tokenizer"].from_pretrained.assert_called_with(
            "test-path",
            trust_remote_code=True
        )

    def test_load_model_flash_attn_fallback(self, mock_deps):
        """Test fallback when Flash Attention is not available."""
        # Simulate ImportError on first call for flash_attention_2
        mock_deps["model"].from_pretrained.side_effect = [
            ImportError("FlashAttention not installed"),
            mock_deps["model_instance"]
        ]
        
        adapter = DeepSeekOCRProviderAdapter()
        config = ModelConfig(
            model_type="vlm",
            provider="deepseek-ocr",
            model_path="test-path",
            extra_params={"attn_implementation": "flash_attention_2"}
        )
        
        handle = adapter.load_model(config)
        
        # Should be called twice: once with flash_attention_2, then with eager fallback
        assert mock_deps["model"].from_pretrained.call_count == 2
        mock_deps["model"].from_pretrained.assert_any_call(
            "test-path",
            _attn_implementation="flash_attention_2",
            trust_remote_code=True,
            use_safetensors=True
        )
        mock_deps["model"].from_pretrained.assert_any_call(
            "test-path",
            _attn_implementation="eager",
            trust_remote_code=True,
            use_safetensors=True
        )

    def test_unload_model(self, mock_deps):
        """Test unloading coordinates memory cleanup."""
        adapter = DeepSeekOCRProviderAdapter()
        config = ModelConfig(
            model_type="vlm", 
            provider="deepseek-ocr", 
            model_path="test-path"
        )
        handle = adapter.load_model(config)
        
        adapter.unload_model(handle)
        
        assert adapter._model is None
        assert adapter._tokenizer is None

    def test_infer(self, mock_deps):
        """Test inference flow including path and prompt handling."""
        adapter = DeepSeekOCRProviderAdapter()
        config = ModelConfig(
            model_type="vlm", 
            provider="deepseek-ocr", 
            model_path="test-path"
        )
        handle = adapter.load_model(config)
        
        # Set up mock behavior for model.infer to create a result file
        def mock_infer_side_effect(*args, **kwargs):
            output_path = kwargs.get("output_path")
            if output_path:
                result_file = Path(output_path) / "result.md"
                # Write some dummy text
                result_file.write_text("Mocked OCR Result", encoding="utf-8")
        
        mock_deps["model_instance"].infer.side_effect = mock_infer_side_effect
        
        result = adapter.infer(handle, "dummy.png", mode="document")
        
        assert result == "Mocked OCR Result"
        mock_deps["model_instance"].infer.assert_called_once()
        # Verify prompt inclusion for document mode
        call_args = mock_deps["model_instance"].infer.call_args
        assert "<|grounding|>" in call_args.kwargs["prompt"]


if __name__ == "__main__":
    pytest.main([__file__])
