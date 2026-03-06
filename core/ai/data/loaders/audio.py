"""
Audio loader implementation.

Strategy pattern: Implements BaseLoader for audio data.
"""

from typing import Callable, Optional, Union, List
from pathlib import Path
from ...base.data.base import BaseLoader
from ...base.data.item import DataItem


class AudioLoader(BaseLoader):
    """
    Audio data loader.

    Loads audio from files.
    """

    def _load(self, source: Union[str, Path], data_collator: Optional[Callable] = None, frame_indices: Optional[List[int]] = None) -> DataItem:
        """
        Load audio data.

        Args:
            source: Audio source (file path)
            data_collator: Optional data collator function
            frame_indices: Optional list of frame indices to load (for video loading)
        Returns:
            DataItem: Loaded audio data
        """
        try:
            import librosa
            import soundfile as sf
        except ImportError:
            raise ImportError(
                "Audio loading requires librosa and soundfile. "
                "Install with: pip install librosa soundfile"
            )

        if isinstance(source, Path):
            source = str(source)

        if isinstance(source, str):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Audio file not found: {source}")

            # Load audio using librosa
            audio, sr = librosa.load(str(path), sr=None)

            if data_collator:
                audio = data_collator(audio)

            # Get metadata from soundfile
            info = sf.info(str(path))

            return DataItem(
                data=audio,
                data_type="audio",
                source=str(path),
                metadata={
                    "sample_rate": sr,
                    "duration": len(audio) / sr,
                    "channels": info.channels,
                    "format": info.format,
                    "subtype": info.subtype,
                },
            )
        else:
            raise ValueError(f"Unsupported audio source type: {type(source)}")
