"""
Observer pattern implementation.

Provides Observer interface and Observable base class for event-driven architecture.

```
class MyService(Observable):
    def do_something(self):
        self.notify(Event(
            type=EventType.INFERENCE_STARTED,
            data={"model_id": "my-model"},
            source="MyService",
        ))
        # ... do work ...
        self.notify(Event(
            type=EventType.INFERENCE_COMPLETED,
            data={"model_id": "my-model", "duration_ms": 42},
            source="MyService",
        ))
        

class LoggingObserver(Observer):
    def on_event(self, event: Event) -> None:
        if event.type == EventType.INFERENCE_COMPLETED:
            print(f"Inference done: {event.data}")
        elif event.type == EventType.INFERENCE_FAILED:
            print(f"Inference failed: {event.data}")
            
 
service = MyService()
observer = LoggingObserver()

service.attach(observer)
service.do_something()   # observer.on_event() is called for each Event
service.detach(observer)           

```

"""

from abc import ABC, abstractmethod
from typing import List
from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    from datetime import UTC, datetime, timezone
except ImportError:
    # Python 3.10 compatibility: UTC is not defined in datetime module
    from datetime import datetime, timezone
    UTC = timezone.utc

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
            self.timestamp = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "source": self.source,
        }


class Observer(ABC):
    """Observer interface for receiving events."""
    
    @abstractmethod
    def on_event(self, event: Event) -> None:
        """
        Handle an event.
        
        Args:
            event: The event to handle
        """
        pass


class Observable:
    """Subject that notifies observers of events."""
    
    def __init__(self):
        """Initialize observable with empty observer list."""
        self._observers: List[Observer] = []
    
    def attach(self, observer: Observer) -> None:
        """
        Attach an observer to receive events.
        
        Args:
            observer: The observer to attach
        """
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach(self, observer: Observer) -> None:
        """
        Detach an observer.
        
        Args:
            observer: The observer to detach
        """
        if observer in self._observers:
            self._observers.remove(observer)
    
    def notify(self, event: Event) -> None:
        """
        Notify all observers of an event.
        
        Args:
            event: The event to notify observers about
        """
        for observer in self._observers:
            try:
                observer.on_event(event)
            except Exception as e:
                # Log error but don't break notification chain
                import logging
                logging.getLogger(__name__).error(
                    f"Error in observer {observer.__class__.__name__}: {e}",
                    exc_info=True
                )
