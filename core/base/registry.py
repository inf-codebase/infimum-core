from typing import Dict, List, Type, Set
from core.base.entity import BaseEntity
import importlib
import pkgutil
import inspect
import os
from loguru import logger
from collections import defaultdict

class EntityRegistry:
    """Registry for entity classes to enable automatic table creation"""
    
    _entities: Dict[str, Type[BaseEntity]] = {}
    _dependencies: Dict[str, Set[str]] = defaultdict(set)
    
    @classmethod
    def register(cls, entity_class: Type[BaseEntity]) -> None:
        """Register an entity class"""
        if not issubclass(entity_class, BaseEntity) or entity_class is BaseEntity:
            return
            
        cls._entities[entity_class.__name__] = entity_class
        # Analyze and store dependencies when registering
        cls._analyze_dependencies(entity_class)
        logger.debug(f"Registered entity: {entity_class.__name__}")
    
    @classmethod
    def _analyze_dependencies(cls, entity_class: Type[BaseEntity]) -> None:
        """Analyze entity class to identify its dependencies for table creation order"""
        # Only look for foreign keys, not relationships (to avoid circular dependencies)
        for attr_name, attr_value in entity_class.__dict__.items():
            # Check for SQLAlchemy Column with ForeignKey
            if hasattr(attr_value, 'type') and hasattr(attr_value, 'foreign_keys'):
                # Handle both resolved and unresolved foreign keys
                if attr_value.foreign_keys:
                    for fk in attr_value.foreign_keys:
                        if hasattr(fk, '_colspec') and isinstance(fk._colspec, str):
                            # Handle string-based foreign keys like "user.id"
                            table_name = fk._colspec.split('.')[0]
                            # Find the actual entity class by table name
                            target_entity = None
                            for name, entity_cls in cls._entities.items():
                                if hasattr(entity_cls, '__tablename__') and entity_cls.__tablename__ == table_name:
                                    target_entity = name
                                    break
                            
                            if target_entity:
                                cls._dependencies[entity_class.__name__].add(target_entity)
                                logger.debug(f"Found string FK dependency: {entity_class.__name__} -> {target_entity}")
                            else:
                                # Fallback: convert table name to entity name
                                target_entity = table_name.replace('_', ' ').title().replace(' ', '')
                                cls._dependencies[entity_class.__name__].add(target_entity)
                                logger.debug(f"Found string FK dependency (fallback): {entity_class.__name__} -> {target_entity}")
                        elif hasattr(fk, 'column') and hasattr(fk.column, 'table'):
                            table_name = fk.column.table.name
                            # Convert table name to entity name (capitalize first letter)
                            target_entity = table_name.replace('_', ' ').title().replace(' ', '')
                            cls._dependencies[entity_class.__name__].add(target_entity)
                            logger.debug(f"Found resolved FK dependency: {entity_class.__name__} -> {target_entity}")
        
        logger.debug(f"FK Dependencies for {entity_class.__name__}: {cls._dependencies[entity_class.__name__]}")
    
    @classmethod
    def discover_entities(cls, package_path):
        if not package_path:
            logger.warning("No package path provided for entity discovery, using default 'entities'")
            package_path = 'entities'
        
        try:
            # Import the package
            package = importlib.import_module(package_path)
            
            # Check if package has a __file__ attribute
            if not hasattr(package, '__file__') or package.__file__ is None:
                logger.warning(f"Package {package_path} doesn't have a valid file path")
                return
                
            package_dir = os.path.dirname(package.__file__)
            
            # Walk through all modules in the package
            for _, module_name, is_pkg in pkgutil.iter_modules([package_dir]):
                # Import the module
                full_module_name = f"{package_path}.{module_name}"
                module = importlib.import_module(full_module_name)
                
                # Register all entity classes in the module
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and issubclass(obj, BaseEntity) and obj is not BaseEntity:
                        cls.register(obj)
                
                # If it's a package, recursively discover entities
                if is_pkg:
                    cls.discover_entities(full_module_name)
                    
            logger.info(f"Discovered {len(cls._entities)} entities from {package_path}")
            
        except ImportError as e:
            logger.warning(f"Could not import package {package_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Error discovering entities: {str(e)}")
    
    @classmethod
    def _topological_sort(cls) -> List[Type[BaseEntity]]:
        """Sort entities based on their dependencies using topological sort"""
        logger.info(f"All detected dependencies: {dict(cls._dependencies)}")
        
        # Create a graph of dependencies
        graph = {name: set() for name in cls._entities.keys()}
        for entity, deps in cls._dependencies.items():
            for dep in deps:
                if dep in graph:  # Only consider registered entities
                    graph[entity].add(dep)
        
        logger.info(f"Dependency graph: {dict(graph)}")
        
        # Perform topological sort
        visited = set()
        temp_mark = set()
        ordered = []
        
        def visit(node):
            if node in temp_mark:
                logger.warning(f"Circular dependency detected involving {node}")
                return
            if node not in visited:
                temp_mark.add(node)
                for dep in graph[node]:
                    visit(dep)
                temp_mark.remove(node)
                visited.add(node)
                ordered.append(node)
        
        for node in graph:
            if node not in visited:
                visit(node)
        
        # The post-order traversal already gives us the correct order
        # (dependencies are added before entities that depend on them)
        
        # Convert entity names back to entity classes
        return [cls._entities[name] for name in ordered]
    
    @classmethod
    def get_all_entities(cls) -> List[Type[BaseEntity]]:
        """Get all registered entities in dependency order"""
        # Re-analyze dependencies now that all entities are registered
        cls._analyze_all_dependencies()
        ordered_entities = cls._topological_sort()
        logger.info(f"Entity creation order: {[e.__name__ for e in ordered_entities]}")
        return ordered_entities
    
    @classmethod
    def _analyze_all_dependencies(cls) -> None:
        """Re-analyze dependencies for all entities after registration is complete"""
        cls._dependencies.clear()
        for entity_name, entity_class in cls._entities.items():
            cls._analyze_dependencies(entity_class)
    
    @classmethod
    def get_entity(cls, name: str) -> Type[BaseEntity]:
        """Get a specific entity by name"""
        return cls._entities.get(name)
    
    @classmethod
    def create_tables(cls, db_manager, drop_first=False) -> bool:
        """Create tables for all registered entities"""
        try:
            # Get entities in dependency order
            ordered_entities = cls.get_all_entities()
            
            if hasattr(db_manager, 'create_tables'):
                return db_manager.create_tables(ordered_entities, drop_first)
            elif hasattr(db_manager, 'engine'):
                # Direct table creation if create_tables method doesn't exist
                if drop_first:
                    BaseEntity.metadata.drop_all(db_manager.engine)
                BaseEntity.metadata.create_all(db_manager.engine)
                return True
            else:
                logger.error("Database manager doesn't support table creation")
                return False
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}")
            return False 