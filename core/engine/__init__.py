from core.base.registry import EntityRegistry

class Engine:
    @staticmethod
    def initialize(entities_packages=['entities']):
        for entities_package in entities_packages:
            EntityRegistry.discover_entities(entities_package)

__all__ = [
    "Engine",
]