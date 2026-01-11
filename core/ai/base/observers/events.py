"""Event types and definitions for the observer pattern."""

from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, Optional
from datetime import datetime


class EventType(str, Enum):
    """Event types for model operations."""
    # Model loading events
    MODEL_LOADING_STARTED = "model_loading_started"
    MODEL_LOADING_COMPLETED = "model_loading_completed"
    MODEL_LOADING_FAILED = "model_loading_failed"
    MODEL_UNLOADING_STARTED = "model_unloading_started"
    MODEL_UNLOADING_COMPLETED = "model_unloading_completed"
    
    # Inference events
    INFERENCE_STARTED = "inference_started"
    INFERENCE_COMPLETED = "inference_completed"
    INFERENCE_FAILED = "inference_failed"
    
    # Training events
    TRAINING_STARTED = "training_started"
    TRAINING_EPOCH_COMPLETED = "training_epoch_completed"
    TRAINING_COMPLETED = "training_completed"
    TRAINING_FAILED = "training_failed"
    
    # Data loading events
    DATA_LOADING_STARTED = "data_loading_started"
    DATA_LOADING_COMPLETED = "data_loading_completed"
    DATA_LOADING_FAILED = "data_loading_failed"
    
    # Preprocessing events
    PREPROCESSING_STARTED = "preprocessing_started"
    PREPROCESSING_COMPLETED = "preprocessing_completed"
    PREPROCESSING_FAILED = "preprocessing_failed"


@dataclass
class Event:
    """Event object for observer pattern."""
    type: EventType
    data: Dict[str, Any]
    timestamp: datetime = None
    source: Optional[str] = None
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "source": self.source,
        }
