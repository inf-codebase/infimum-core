"""VLM module for vision-language models.

Providers:
  - XCLIPProvider   — inference / event detection
  - XCLIPTrainer    — contrastive fine-tuning
  - XCLIPVideoDataset — PyTorch Dataset for fine-tuning
  - XCLIPTrainConfig  — dataclass with training hyper-parameters
"""

from .providers import (
    XCLIPProvider,
    XCLIPTrainer,
    XCLIPVideoDataset,
    XCLIPTrainConfig,
)

__all__ = [
    "XCLIPProvider",
    "XCLIPTrainer",
    "XCLIPVideoDataset",
    "XCLIPTrainConfig",
]
