"""
Audio preprocessing utilities for format conversion.
"""

import io
import librosa
import numpy as np
from typing import BinaryIO, Tuple
import torch


class AudioPreprocessingError(Exception):
    """Exception raised for audio preprocessing errors."""
    pass


async def convert_to_mono_16khz(
    audio_file,
    target_sample_rate: int = 16000
) -> Tuple[np.ndarray, int]:
    """
    Convert audio to mono 16kHz format.
    
    Args:
        audio_file: FastAPI UploadFile or binary file-like object
        target_sample_rate: Target sample rate (default: 16000)
    
    Returns:
        Tuple of (audio_array, sample_rate)
    
    Raises:
        AudioPreprocessingError: If conversion fails
    """
    try:
        # Read audio bytes - use async for FastAPI UploadFile
        if hasattr(audio_file, 'read') and hasattr(audio_file.read, '__call__'):
            try:
                import inspect
                if inspect.iscoroutinefunction(audio_file.read):
                    # FastAPI UploadFile - use async methods
                    await audio_file.seek(0)
                    audio_bytes = await audio_file.read()
                    await audio_file.seek(0)
                else:
                    # Sync file-like object
                    audio_file.seek(0)
                    audio_bytes = audio_file.read()
                    audio_file.seek(0)
            except (AttributeError, TypeError):
                # Fallback: try .file attribute
                if hasattr(audio_file, 'file'):
                    file_obj = audio_file.file
                    file_obj.seek(0)
                    audio_bytes = file_obj.read()
                    file_obj.seek(0)
                else:
                    audio_bytes = audio_file.read()
                    if hasattr(audio_file, 'seek'):
                        audio_file.seek(0)
        else:
            raise AudioPreprocessingError("Invalid audio file object")
        
        if not audio_bytes:
            raise AudioPreprocessingError("Empty audio file")
        
        audio_buffer = io.BytesIO(audio_bytes)
        
        # Load audio and convert to mono, resample to target rate
        y, sr = librosa.load(
            audio_buffer,
            sr=target_sample_rate,
            mono=True,
            res_type='kaiser_best'
        )
        
        return y, sr
        
    except Exception as e:
        raise AudioPreprocessingError(f"Failed to convert audio: {str(e)}")


def normalize_audio(
    audio: np.ndarray,
    target_level: float = 0.95
) -> np.ndarray:
    """
    Normalize audio to target level.
    
    Args:
        audio: Audio array
        target_level: Target peak level (0.0 to 1.0)
    
    Returns:
        Normalized audio array
    """
    if len(audio) == 0:
        return audio
    
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        normalization_factor = target_level / max_val
        audio = audio * normalization_factor
    
    return audio
