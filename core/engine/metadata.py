from typing import Type, get_type_hints, Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, create_model
from core.base.entity import BaseEntity
from sqlalchemy.orm import Mapped

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
    type_hints = get_type_hints(entity_class)
    
    # Create field definitions dictionary
    field_definitions: Dict[str, Any] = {}
    
    # Add fields to the DTO, excluding relationships
    ignore_fields = ['id', 'created_at', 'updated_at', '__name__', '__tablename__', '__origin__']
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
