"""
Audio validation utilities for format and quality checking.
"""

import io
import wave
import audioop
import librosa
import numpy as np
from typing import Optional, Tuple, BinaryIO
from dataclasses import dataclass
from pydantic import BaseModel


class AudioValidationError(Exception):
    """Exception raised for audio validation errors."""
    pass


@dataclass
class AudioValidationResult:
    """Result of audio validation."""
    is_valid: bool
    sample_rate: int
    channels: int
    duration: float
    errors: list[str]
    warnings: list[str]


class AudioFormatSpec(BaseModel):
    """Audio format specification for validation."""
    required_sample_rate: int = 16000
    required_channels: int = 1  # mono
    min_duration_seconds: float = 0.1
    max_duration_seconds: float = 600.0  # 10 minutes
    min_rms: int = 100  # Minimum RMS to avoid silence
    supported_formats: list[str] = ["audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3"]


async def validate_audio_format(
    audio_file,
    filename: Optional[str] = None,
    format_spec: Optional[AudioFormatSpec] = None
) -> AudioValidationResult:
    """
    Validate audio file format and basic properties.
    
    Args:
        audio_file: FastAPI UploadFile or binary file-like object
        filename: Optional filename for format detection
        format_spec: Optional format specification (uses defaults if None)
    
    Returns:
        AudioValidationResult with validation status and metadata
    
    Raises:
        AudioValidationError: If audio cannot be read or is severely invalid
    """
    if format_spec is None:
        format_spec = AudioFormatSpec()
    
    errors = []
    warnings = []
    
    # Read file content - use async for FastAPI UploadFile
    if hasattr(audio_file, 'read') and hasattr(audio_file.read, '__call__'):
        # Check if it's FastAPI UploadFile (has async methods)
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
            # Fallback: try .file attribute (synchronous)
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
        raise AudioValidationError("Invalid audio file object")
    
    if not audio_bytes:
        raise AudioValidationError("Audio file is empty")
    
    # Try to read as WAV first
    sample_rate = None
    channels = None
    duration = 0.0
    
    try:
        # Try WAV format
        audio_buffer = io.BytesIO(audio_bytes)
        with wave.open(audio_buffer, "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            sampwidth = wf.getsampwidth()
            duration = n_frames / sample_rate if sample_rate > 0 else 0.0
            
            # Check channels
            if channels != format_spec.required_channels:
                errors.append(
                    f"Audio must be mono (1 channel), got {channels} channels"
                )
            
            # Check sample rate
            if sample_rate != format_spec.required_sample_rate:
                errors.append(
                    f"Sample rate must be {format_spec.required_sample_rate}Hz, got {sample_rate}Hz"
                )
            
            # Check duration
            if duration < format_spec.min_duration_seconds:
                errors.append(
                    f"Audio too short: {duration:.2f}s (minimum: {format_spec.min_duration_seconds}s)"
                )
            elif duration > format_spec.max_duration_seconds:
                errors.append(
                    f"Audio too long: {duration:.2f}s (maximum: {format_spec.max_duration_seconds}s)"
                )
            
            # Check for silence
            raw_audio = wf.readframes(n_frames)
            rms = audioop.rms(raw_audio, sampwidth)
            if rms < format_spec.min_rms:
                errors.append("Audio contains only silence or is too quiet")
            
    except wave.Error:
        # Not a WAV file, try librosa for other formats
        try:
            audio_buffer = io.BytesIO(audio_bytes)
            y, sr = librosa.load(audio_buffer, sr=None, mono=False)
            sample_rate = int(sr)
            
            if len(y.shape) == 1:
                channels = 1
            else:
                channels = y.shape[0]
            
            duration = len(y) / sample_rate if sample_rate > 0 else 0.0
            
            # Check channels
            if channels != format_spec.required_channels:
                errors.append(
                    f"Audio must be mono (1 channel), got {channels} channels"
                )
            
            # Check sample rate
            if sample_rate != format_spec.required_sample_rate:
                warnings.append(
                    f"Sample rate is {sample_rate}Hz, will be resampled to {format_spec.required_sample_rate}Hz"
                )
            
            # Check duration
            if duration < format_spec.min_duration_seconds:
                errors.append(
                    f"Audio too short: {duration:.2f}s (minimum: {format_spec.min_duration_seconds}s)"
                )
            elif duration > format_spec.max_duration_seconds:
                errors.append(
                    f"Audio too long: {duration:.2f}s (maximum: {format_spec.max_duration_seconds}s)"
                )
            
            # Check for silence
            if channels == 1:
                rms = np.sqrt(np.mean(y**2)) * 32767  # Convert to int16 scale
            else:
                rms = np.sqrt(np.mean(y[0]**2)) * 32767
            
            if rms < format_spec.min_rms:
                errors.append("Audio contains only silence or is too quiet")
                
        except Exception as e:
            raise AudioValidationError(f"Unable to read audio file: {str(e)}")
    
    is_valid = len(errors) == 0
    
    return AudioValidationResult(
        is_valid=is_valid,
        sample_rate=sample_rate or 0,
        channels=channels or 0,
        duration=duration,
        errors=errors,
        warnings=warnings
    )


async def validate_audio_quality(
    audio_file,
    format_spec: Optional[AudioFormatSpec] = None
) -> Tuple[bool, list[str]]:
    """
    Validate audio quality metrics (SNR, clipping, etc.).
    
    Args:
        audio_file: FastAPI UploadFile or binary file-like object
        format_spec: Optional format specification
    
    Returns:
        Tuple of (is_good_quality, quality_issues)
    """
    if format_spec is None:
        format_spec = AudioFormatSpec()
    
    quality_issues = []
    
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
            raise AudioValidationError("Invalid audio file object")
        
        audio_buffer = io.BytesIO(audio_bytes)
        
        # Load audio
        y, sr = librosa.load(audio_buffer, sr=16000, mono=True)
        
        # Only flag severe quality issues that would prevent transcription
        # The MedASR notebook doesn't do quality validation - it processes directly
        # So we should be lenient and only block on critical issues
        
        signal_power = np.mean(y**2)
        
        # Only block if audio is completely silent or extremely weak
        # This is a critical issue that would prevent transcription
        if signal_power < 0.0001:  # Very low threshold - only catch truly silent audio
            quality_issues.append("Audio signal is too weak or silent")
        
        # Check for severe clipping (only flag if really bad)
        max_amplitude = np.max(np.abs(y))
        if max_amplitude > 0.99:  # Only flag severe clipping
            quality_issues.append("Audio is severely clipped")
        
        # Note: We removed the "excessive noise" check because:
        # 1. It's too simplistic and flags normal speech
        # 2. MedASR model can handle various noise levels
        # 3. The notebook doesn't validate noise levels
        
    except Exception as e:
        # Only add to quality issues if it's a critical error
        # Don't block on analysis failures - let the model try
        quality_issues.append(f"Unable to analyze audio quality: {str(e)}")
    
    # Only block if there are critical quality issues
    # Most quality issues should be warnings, not blockers
    is_good_quality = len(quality_issues) == 0
    return is_good_quality, quality_issues
