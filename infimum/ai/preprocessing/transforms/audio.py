"""
Audio transform implementations.

Chain of Responsibility: Each transform is a link in the processing chain.
"""

from typing import Optional
import numpy as np
from ...base.preprocessing.base import BaseTransform
from ...base.data.item import DataItem


class ResampleTransform(BaseTransform):
    """Resample audio transform."""

    def __init__(self, target_sr: int = 16000):
        """
        Initialize resample transform.

        Args:
            target_sr: Target sample rate
        """
        self.target_sr = target_sr

    def transform(self, data: DataItem) -> DataItem:
        """
        Resample audio.

        Args:
            data: Input data item with audio

        Returns:
            Transformed data item
        """
        if data.data_type != "audio":
            raise ValueError(
                f"ResampleTransform expects audio data, got {data.data_type}"
            )

        try:
            import librosa
        except ImportError:
            raise ImportError("ResampleTransform requires librosa")

        audio = data.data
        current_sr = data.get("sample_rate")

        if current_sr and current_sr != self.target_sr:
            audio = librosa.resample(
                audio, orig_sr=current_sr, target_sr=self.target_sr
            )
            data.set("sample_rate", self.target_sr)
            data.set("resampled", True)

        data.data = audio
        return data


class NormalizeAudioTransform(BaseTransform):
    """Normalize audio transform."""

    def __init__(self, method: str = "peak"):
        """
        Initialize normalize audio transform.

        Args:
            method: Normalization method ('peak' or 'rms')
        """
        self.method = method

    def transform(self, data: DataItem) -> DataItem:
        """
        Normalize audio.

        Args:
            data: Input data item with audio

        Returns:
            Transformed data item
        """
        if data.data_type != "audio":
            raise ValueError(
                f"NormalizeAudioTransform expects audio data, got {data.data_type}"
            )

        audio = data.data

        if self.method == "peak":
            # Peak normalization
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val
        elif self.method == "rms":
            # RMS normalization
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                audio = audio / rms

        data.data = audio
        data.set("normalized", True)
        data.set("normalization_method", self.method)
        return data
