"""
Observer pattern implementation.

Provides Observer interface and Observable base class for event-driven architecture.
"""

from abc import ABC, abstractmethod
from typing import List
from .events import Event


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
