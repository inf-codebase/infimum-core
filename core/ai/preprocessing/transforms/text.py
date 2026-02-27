"""
Text transform implementations.

Chain of Responsibility: Each transform is a link in the processing chain.
"""

import re
from typing import Optional
from ...base.preprocessing.base import BaseTransform
from ...base.data.item import DataItem


class TokenizeTransform(BaseTransform):
    """Tokenize text transform."""

    def __init__(self, tokenizer=None):
        """
        Initialize tokenize transform.

        Args:
            tokenizer: Optional tokenizer function or object
        """
        self.tokenizer = tokenizer

    def transform(self, data: DataItem) -> DataItem:
        """
        Tokenize text.

        Args:
            data: Input data item with text

        Returns:
            Transformed data item
        """
        if data.data_type != "text":
            raise ValueError(
                f"TokenizeTransform expects text data, got {data.data_type}"
            )

        text = data.data
        if self.tokenizer:
            if callable(self.tokenizer):
                tokens = self.tokenizer(text)
            else:
                tokens = self.tokenizer.tokenize(text)
        else:
            # Simple whitespace tokenization
            tokens = text.split()

        data.data = tokens
        data.set("tokenized", True)
        data.set("token_count", len(tokens))
        return data


class CleanTransform(BaseTransform):
    """Clean text transform."""

    def __init__(self, remove_urls: bool = True, remove_emails: bool = True):
        """
        Initialize clean transform.

        Args:
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses
        """
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails

    def transform(self, data: DataItem) -> DataItem:
        """
        Clean text.

        Args:
            data: Input data item with text

        Returns:
            Transformed data item
        """
        if data.data_type != "text":
            raise ValueError(f"CleanTransform expects text data, got {data.data_type}")

        text = data.data

        if self.remove_urls:
            text = re.sub(r"http\S+|www\S+", "", text)

        if self.remove_emails:
            text = re.sub(r"\S+@\S+", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        data.data = text
        data.set("cleaned", True)
        return data


class NormalizeTextTransform(BaseTransform):
    """Normalize text transform (lowercase, etc.)."""

    def __init__(self, lowercase: bool = True, remove_punctuation: bool = False):
        """
        Initialize normalize text transform.

        Args:
            lowercase: Whether to convert to lowercase
            remove_punctuation: Whether to remove punctuation
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation

    def transform(self, data: DataItem) -> DataItem:
        """
        Normalize text.

        Args:
            data: Input data item with text

        Returns:
            Transformed data item
        """
        if data.data_type != "text":
            raise ValueError(
                f"NormalizeTextTransform expects text data, got {data.data_type}"
            )

        text = data.data

        if self.lowercase:
            text = text.lower()

        if self.remove_punctuation:
            text = re.sub(r"[^\w\s]", "", text)

        data.data = text
        data.set("normalized", True)
        return data
