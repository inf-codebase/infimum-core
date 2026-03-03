from core.base.registry import EntityRegistry
from core.database import DatabaseManager
from core.engine.context import with_database
from core.engine.design_pattern import Observer, Event, EventType
from core.engine.design_pattern.singleton import Singleton

class Engine:
    @staticmethod
    def initialize(entities_packages=['entities']):
        for entities_package in entities_packages:
            EntityRegistry.discover_entities(entities_package)

    @staticmethod
    def with_database(db_name_or_prefix: str, async_mode: bool = False) -> DatabaseManager:
        return with_database(db_name_or_prefix, async_mode)
    
    
__all__ = [
    "Engine",
    "with_database",
    "Observer",
    "Event",
    "EventType",
    "Singleton",
]