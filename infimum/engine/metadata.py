from typing import Type, get_type_hints, Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, create_model
from infimum.base.entity import BaseEntity
from sqlalchemy.orm import Mapped
from datetime import datetime

def create_dynamic_class(class_name: str, attributes: dict = {}, base_classes: tuple = (object,)) -> type:
    """
    Create a class dynamically with the given name and attributes
    
    Args:
        class_name: Name of the class to create
        attributes: Dictionary of attributes and methods
        base_classes: Tuple of base classes to inherit from
        
    Returns:
        The newly created class
    """
    # Create the class using type()
    new_class = type(class_name, base_classes, attributes)
    return new_class

def create_dto_class(entity_class: Type[BaseEntity], class_name: str = None) -> Type[BaseModel]:
    """
    Create a DTO (Data Transfer Object) class from an entity class.
    The DTO will include all fields from the entity except relationships.
    
    Args:
        entity_class: The entity class to create a DTO from
        class_name: Optional name for the DTO class. If not provided, will use entity name + 'DTO'
        
    Returns:
        A new DTO class inheriting from Pydantic BaseModel
    """
    if class_name is None:
        class_name = f"{entity_class.__name__}DTO"
    
    # Get all type hints from the entity class
    type_hints = get_type_hints(entity_class, include_extras=True)
    
    # Also inspect class attributes for SQLAlchemy columns (which may not have type hints)
    from sqlalchemy import Column
    from sqlalchemy.orm import RelationshipProperty
    
    # Create field definitions dictionary
    field_definitions: Dict[str, Any] = {}
    
    # Add fields to the DTO, excluding relationships
    ignore_fields = ['id', 'created_at', 'updated_at', '__name__', '__tablename__', '__origin__']
    
    # First, process type hints
    for field_name, field_type in type_hints.items():
        # Skip relationship fields and SQLAlchemy specific fields
        if field_name.endswith('_id') or field_name in ignore_fields:
            continue
            
        # Extract the actual type from Mapped if it's a Mapped type
        if hasattr(field_type, '__origin__') and field_type.__origin__ is Mapped:
            actual_type = field_type.__args__[0]
        else:
            actual_type = field_type
            
        # Add the field definition
        field_definitions[field_name] = (actual_type, Field(default=None))
    
    # Then, process SQLAlchemy table columns (for columns defined as class attributes)
    # This handles the case where columns are defined as class attributes without type hints
    if hasattr(entity_class, '__table__') and entity_class.__table__ is not None:
        for column in entity_class.__table__.columns:
            attr_name = column.name
            # Skip if already in field_definitions or should be ignored
            if attr_name in field_definitions or attr_name in ignore_fields or attr_name.endswith('_id'):
                continue
            
            # Check if there's a relationship property with the same name (skip relationships)
            if hasattr(entity_class, attr_name):
                attr = getattr(entity_class, attr_name)
                if isinstance(attr, RelationshipProperty):
                    continue
            
            # Try to infer type from column type
            col_type = column.type
            python_type = str  # Default to str
            if hasattr(col_type, 'python_type'):
                python_type = col_type.python_type
            elif hasattr(col_type, '__class__'):
                # Try to map common SQLAlchemy types
                type_name = col_type.__class__.__name__
                if 'Integer' in type_name:
                    python_type = int
                elif 'Float' in type_name or 'Numeric' in type_name:
                    python_type = float
                elif 'Boolean' in type_name:
                    python_type = bool
                elif 'DateTime' in type_name or 'Date' in type_name:
                    python_type = datetime
                else:
                    python_type = str
            
            field_definitions[attr_name] = (python_type, Field(default=None))
    
    # Create the model using Pydantic's create_model
    return create_model(
        class_name,
        __config__=ConfigDict(arbitrary_types_allowed=True),
        **field_definitions
    )

def create_dtos_for_entities(entities: List[Type[BaseEntity]]) -> dict:
    """
    Create DTOs for multiple entity classes
    
    Args:
        entities: List of entity classes to create DTOs for
        
    Returns:
        Dictionary mapping entity class names to their corresponding DTO classes
    """
    dtos = {}
    for entity in entities:
        dto_name = f"{entity.__name__}DTO"
        dtos[dto_name] = create_dto_class(entity, dto_name)
    return dtos
