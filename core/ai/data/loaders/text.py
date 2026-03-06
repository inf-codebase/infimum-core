"""
Text loader implementation.

Strategy pattern: Implements BaseLoader for text data.
"""

from typing import Callable, Optional, Union, List
from pathlib import Path
from ...base.data.base import BaseLoader
from ...base.data.item import DataItem


class TextLoader(BaseLoader):
    """
    Text data loader.

    Loads text from files or strings.
    """

    def _load(self, source: Union[str, Path], data_collator: Optional[Callable] = None, frame_indices: Optional[List[int]] = None) -> DataItem:
        """
        Load text data.

        Args:
            source: Text source (path or string)
            data_collator: Optional data collator function
            frame_indices: Optional list of frame indices to load (for video loading)
        Returns:
            DataItem: Loaded text data
        """
        if isinstance(source, Path):
            source = str(source)

        if isinstance(source, str):
            # Check if it's a file path
            path = Path(source)
            if path.exists() and path.is_file():
                # Load from file
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()

                if data_collator:
                    text = data_collator(text)

                return DataItem(
                    data=text,
                    data_type="text",
                    source=str(path),
                    metadata={
                        "length": len(text),
                        "encoding": "utf-8",
                    },
                )
            else:
                # Treat as text content
                return DataItem(
                    data=source,
                    data_type="text",
                    metadata={
                        "length": len(source),
                    },
                )
        else:
            raise ValueError(f"Unsupported text source type: {type(source)}")
