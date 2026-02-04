#!/usr/bin/env python3
"""
Test script for DeepSeek-OCR-2 provider.

Usage:
    python test_deepseek_ocr.py <image_path>
    
Example:
    python test_deepseek_ocr.py demo.jpeg
"""

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock

# Senior Trick: Stub out broken modules that are not needed for this test
# This prevents ModuleNotFoundError in core.ai.speech due to relative import issues
sys.modules["core.ai.speech"] = MagicMock()
sys.modules["core.ai.speech.models"] = MagicMock()
sys.modules["core.ai.speech.models.speech2text"] = MagicMock()
sys.modules["core.ai.speech.models.text2speech"] = MagicMock()

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.ai.vlm.providers.deepseek_ocr import DeepSeekOCRProviderAdapter
from core.ai.base.providers import ModelConfig, ProviderRegistry, ProviderFactory


def test_deepseek_ocr(image_path: str, mode: str = "document"):
    """
    Test DeepSeek-OCR-2 provider using the Factory Pattern.
    
    This demonstrates the Strategy and Factory patterns:
    1. Manual registration in the Factory.
    2. Listing available strategies.
    3. Dynamic instantiation through the Factory.
    
    Args:
        image_path: Path to the image file
        mode: OCR mode - "document" (default) or "free"
    """

    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return
    # Register provider in factory
    ProviderFactory.register("vlm", "deepseek-ocr", DeepSeekOCRProviderAdapter)
    
    # List available providers for a specific type
    vlm_providers = ProviderFactory.list_providers("vlm")
    print(f"📋 Registered VLM providers: {vlm_providers}")
    
    # Configuration
    config = ModelConfig(
        model_type="vlm",
        provider="deepseek-ocr",
        model_path="deepseek-ai/DeepSeek-OCR-2",
        device="cuda"
    )
    
    # Dynamic instantiation (Factory Pattern)
    provider = ProviderFactory.create("vlm", "deepseek-ocr", config)
    
    try:
        # Load the model strategy
        handle = provider.get_model(config)
        
        # Run OCR inference (The Strategy executes its specific logic)
        result = provider.infer(handle, image_path, mode=mode)
        
        # Output result
        print("OCR RESULT")
        print(result)
        
        # Store (optional)
        # output_file = Path(image_path).stem + "_ocr_result.md"
        # with open(output_file, "w", encoding="utf-8") as f:
        #     f.write(result)
        # print(f"Result saved to: {output_file}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup resource (Strategy-specific cleanup)
        if 'provider' in locals():
            provider.unload_model(handle if 'handle' in locals() else None)


def main():
    """Execution entry point."""
    image_path ="demo.jpeg"
    mode = "document"
    
    test_deepseek_ocr(image_path, mode)


if __name__ == "__main__":
    main()
