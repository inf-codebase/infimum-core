"""
Transcription entity for storing transcription results.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class WordConfidence(BaseModel):
    """Word-level confidence score."""
    word: str = Field(description="Transcribed word")
    confidence: float = Field(description="Confidence score (0.0 to 1.0)", ge=0.0, le=1.0)
    start_time: Optional[float] = Field(default=None, description="Start time in seconds")
    end_time: Optional[float] = Field(default=None, description="End time in seconds")


class TranscriptionEntity(BaseModel):
    """Entity representing a transcription result."""
    id: Optional[str] = Field(default=None, description="Unique identifier")
    transcript: str = Field(description="Full transcript text")
    overall_confidence: float = Field(description="Overall confidence score", ge=0.0, le=1.0)
    word_confidences: List[WordConfidence] = Field(
        default_factory=list,
        description="Word-level confidence scores"
    )
    audio_duration: Optional[float] = Field(default=None, description="Audio duration in seconds")
    sample_rate: int = Field(default=16000, description="Audio sample rate")
    model_name: str = Field(default="google/medasr", description="Model used for transcription")
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    retry_count: int = Field(default=0, description="Number of retries attempted")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
