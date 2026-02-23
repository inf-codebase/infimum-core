"""
Unit tests for observer system.

Tests Observer pattern.
"""

import unittest
from unittest.mock import Mock
from datetime import datetime

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from core.ai.base.observers.base import Observer, Observable
from core.ai.base.observers.events import Event, EventType


class TestObserver(unittest.TestCase):
    """Test Observer interface."""
    
    def test_observer_abstract(self):
        """Test that Observer cannot be instantiated."""
        with self.assertRaises(TypeError):
            Observer()
    
    def test_observer_implementation(self):
        """Test implementing Observer."""
        class TestObserver(Observer):
            def __init__(self):
                self.events = []
            
            def on_event(self, event):
                self.events.append(event)
        
        observer = TestObserver()
        event = Event(
            type=EventType.MODEL_LOADING_STARTED,
            data={"test": "data"}
        )
        
        observer.on_event(event)
        self.assertEqual(len(observer.events), 1)
        self.assertEqual(observer.events[0].type, EventType.MODEL_LOADING_STARTED)


class TestObservable(unittest.TestCase):
    """Test Observable."""
    
    def test_attach_observer(self):
        """Test attaching observers."""
        class TestObserver(Observer):
            def on_event(self, event):
                pass
        
        observable = Observable()
        observer1 = TestObserver()
        observer2 = TestObserver()
        
        observable.attach(observer1)
        observable.attach(observer2)
        
        self.assertEqual(len(observable._observers), 2)
    
    def test_attach_duplicate_observer(self):
        """Test that duplicate observers are not added."""
        class TestObserver(Observer):
            def on_event(self, event):
                pass
        
        observable = Observable()
        observer = TestObserver()
        
        observable.attach(observer)
        observable.attach(observer)  # Try to attach again
        
        self.assertEqual(len(observable._observers), 1)
    
    def test_detach_observer(self):
        """Test detaching observers."""
        class TestObserver(Observer):
            def on_event(self, event):
                pass
        
        observable = Observable()
        observer = TestObserver()
        
        observable.attach(observer)
        self.assertEqual(len(observable._observers), 1)
        
        observable.detach(observer)
        self.assertEqual(len(observable._observers), 0)
    
    def test_notify_observers(self):
        """Test notifying observers."""
        class TestObserver(Observer):
            def __init__(self):
                self.events = []
            
            def on_event(self, event):
                self.events.append(event)
        
        observable = Observable()
        observer1 = TestObserver()
        observer2 = TestObserver()
        
        observable.attach(observer1)
        observable.attach(observer2)
        
        event = Event(
            type=EventType.MODEL_LOADING_STARTED,
            data={"test": "data"}
        )
        observable.notify(event)
        
        self.assertEqual(len(observer1.events), 1)
        self.assertEqual(len(observer2.events), 1)
        self.assertEqual(observer1.events[0], event)
        self.assertEqual(observer2.events[0], event)
    
    def test_notify_with_error(self):
        """Test that errors in observers don't break notification chain."""
        class FailingObserver(Observer):
            def on_event(self, event):
                raise Exception("Observer error")
        
        class WorkingObserver(Observer):
            def __init__(self):
                self.events = []
            
            def on_event(self, event):
                self.events.append(event)
        
        observable = Observable()
        failing = FailingObserver()
        working = WorkingObserver()
        
        observable.attach(failing)
        observable.attach(working)
        
        event = Event(
            type=EventType.MODEL_LOADING_STARTED,
            data={"test": "data"}
        )
        
        # Should not raise, and working observer should still receive event
        observable.notify(event)
        
        self.assertEqual(len(working.events), 1)


class TestEvent(unittest.TestCase):
    """Test Event."""
    
    def test_event_creation(self):
        """Test creating an Event."""
        event = Event(
            type=EventType.MODEL_LOADING_STARTED,
            data={"key": "value"}
        )
        
        self.assertEqual(event.type, EventType.MODEL_LOADING_STARTED)
        self.assertEqual(event.data["key"], "value")
        self.assertIsNotNone(event.timestamp)
        self.assertIsInstance(event.timestamp, datetime)
    
    def test_event_with_timestamp(self):
        """Test creating Event with custom timestamp."""
        custom_time = datetime(2024, 1, 1, 12, 0, 0)
        event = Event(
            type=EventType.MODEL_LOADING_COMPLETED,
            data={},
            timestamp=custom_time
        )
        
        self.assertEqual(event.timestamp, custom_time)
    
    def test_event_to_dict(self):
        """Test converting Event to dictionary."""
        event = Event(
            type=EventType.MODEL_LOADING_STARTED,
            data={"test": "data"},
            source="TestSource"
        )
        
        result = event.to_dict()
        self.assertEqual(result["type"], "model_loading_started")
        self.assertEqual(result["data"]["test"], "data")
        self.assertEqual(result["source"], "TestSource")
        self.assertIn("timestamp", result)


class TestEventType(unittest.TestCase):
    """Test EventType enum."""
    
    def test_event_type_values(self):
        """Test EventType enum values."""
        self.assertEqual(EventType.MODEL_LOADING_STARTED.value, "model_loading_started")
        self.assertEqual(EventType.MODEL_LOADING_COMPLETED.value, "model_loading_completed")
        self.assertEqual(EventType.DATA_LOADING_STARTED.value, "data_loading_started")
    
    def test_event_type_string_enum(self):
        """Test that EventType is a string enum."""
        # Should be comparable to strings
        self.assertEqual(EventType.MODEL_LOADING_STARTED, "model_loading_started")
        self.assertNotEqual(EventType.MODEL_LOADING_STARTED, "other")


if __name__ == '__main__':
    unittest.main()
