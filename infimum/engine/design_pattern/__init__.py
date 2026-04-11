"""
Design pattern implementations for the engine.
"""

from .singleton import singleton
from .observer import Observer, Observable, Event, EventType

__all__ = [
    "singleton",
    "Observer",
    "Observable",
    "Event",
    "EventType",
]