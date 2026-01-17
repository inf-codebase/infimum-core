"""Unit tests for core.engine.metadata module."""
import pytest
from unittest.mock import Mock, patch
from core.engine.metadata import (
    create_dynamic_class,
    create_dto_class,
    create_dtos_for_entities
)
from core.base.entity import BaseEntity
from sqlalchemy import Column, String, Integer, ForeignKey
from pydantic import BaseModel


class TestCreateDynamicClass:
    """Test cases for create_dynamic_class function."""

    def test_create_simple_class(self):
        """Test creating a simple dynamic class."""
        attributes = {"name": "Test", "value": 42}
        DynamicClass = create_dynamic_class("TestClass", attributes)
        
        instance = DynamicClass()
        assert instance.name == "Test"
        assert instance.value == 42

    def test_create_class_with_methods(self):
        """Test creating a class with methods."""
        def greet(self):
            return f"Hello, {self.name}"
        
        attributes = {"name": "World", "greet": greet}
        DynamicClass = create_dynamic_class("Greeter", attributes)
        
        instance = DynamicClass()
        assert instance.greet() == "Hello, World"

    def test_create_class_with_base_classes(self):
        """Test creating a class with base classes."""
        class Base:
            def base_method(self):
                return "base"
        
        attributes = {"value": 10}
        DynamicClass = create_dynamic_class("Derived", attributes, (Base,))
        
        instance = DynamicClass()
        assert instance.value == 10
        assert instance.base_method() == "base"


class TestCreateDtoClass:
    """Test cases for create_dto_class function."""

    def test_create_dto_from_entity(self):
        """Test creating DTO from entity class."""
        class UserEntity(BaseEntity):
            name = Column(String(100))
            age = Column(Integer)
        
        UserDTO = create_dto_class(UserEntity)
        
        assert issubclass(UserDTO, BaseModel)
        dto = UserDTO(name="Test", age=25)
        assert dto.name == "Test"
        assert dto.age == 25

    def test_create_dto_with_custom_name(self):
        """Test creating DTO with custom name."""
        class UserEntity(BaseEntity):
            name = Column(String(100))
        
        CustomDTO = create_dto_class(UserEntity, "CustomUserDTO")
        
        assert CustomDTO.__name__ == "CustomUserDTO"

    def test_create_dto_excludes_relationships(self):
        """Test that DTO excludes relationship fields."""
        class UserEntity(BaseEntity):
            name = Column(String(100))
            user_id = Column(Integer, ForeignKey('other.id'))
        
        UserDTO = create_dto_class(UserEntity)
        
        # user_id should be excluded
        dto = UserDTO(name="Test")
        assert dto.name == "Test"
        # user_id should not be in the DTO
        assert not hasattr(dto, 'user_id') or dto.user_id is None


class TestCreateDtosForEntities:
    """Test cases for create_dtos_for_entities function."""

    def test_create_dtos_for_multiple_entities(self):
        """Test creating DTOs for multiple entities."""
        class UserEntity(BaseEntity):
            name = Column(String(100))
        
        class ProductEntity(BaseEntity):
            title = Column(String(200))
        
        dtos = create_dtos_for_entities([UserEntity, ProductEntity])
        
        assert "UserEntityDTO" in dtos
        assert "ProductEntityDTO" in dtos
        assert issubclass(dtos["UserEntityDTO"], BaseModel)
        assert issubclass(dtos["ProductEntityDTO"], BaseModel)
