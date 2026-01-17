"""Unit tests for core.base.entity module."""
import pytest
from datetime import datetime
from core.base.entity import BaseEntity, Document
from sqlalchemy import Column, String, Integer
from pydantic import ValidationError


class TestBaseEntity:
    """Test cases for BaseEntity class."""

    def test_tablename_generation(self):
        """Test automatic table name generation from class name."""
        class TestEntity(BaseEntity):
            name = Column(String(100))
        
        assert TestEntity.__tablename__() == "testentity"

    def test_entity_initialization_with_kwargs(self):
        """Test entity initialization with keyword arguments."""
        class TestEntity(BaseEntity):
            name = Column(String(100))
            age = Column(Integer)
        
        entity = TestEntity(name="Test", age=25)
        assert entity.name == "Test"
        assert entity.age == 25

    def test_entity_has_default_columns(self):
        """Test that BaseEntity has id, created_at, updated_at columns."""
        class TestEntity(BaseEntity):
            pass
        
        assert hasattr(TestEntity, 'id')
        assert hasattr(TestEntity, 'created_at')
        assert hasattr(TestEntity, 'updated_at')

    def test_entity_ignores_unknown_attributes(self):
        """Test that unknown attributes are ignored during initialization."""
        class TestEntity(BaseEntity):
            name = Column(String(100))
        
        entity = TestEntity(name="Test", unknown_attr="ignored")
        assert not hasattr(entity, 'unknown_attr')


class TestDocument:
    """Test cases for Document class."""

    def test_document_initialization(self):
        """Test Document initialization with optional fields."""
        doc = Document()
        assert doc.id is None
        assert doc.created_at is None
        assert doc.updated_at is None

    def test_document_with_values(self):
        """Test Document initialization with values."""
        now = datetime.utcnow()
        doc = Document(id="123", created_at=now, updated_at=now)
        assert doc.id == "123"
        assert doc.created_at == now
        assert doc.updated_at == now

    def test_document_to_dict(self):
        """Test Document to_dict method."""
        doc = Document(id="123", created_at=datetime.utcnow())
        result = doc.to_dict()
        assert isinstance(result, dict)
        assert result.get("id") == "123"
        assert "created_at" in result

    def test_document_to_dict_excludes_none(self):
        """Test that to_dict excludes None values."""
        doc = Document()
        result = doc.to_dict()
        assert result == {}

    def test_get_collection_name_simple(self):
        """Test collection name generation for simple class name."""
        class User(Document):
            pass
        
        assert User.get_collection_name() == "users"

    def test_get_collection_name_camel_case(self):
        """Test collection name generation for camel case."""
        class UserProfile(Document):
            pass
        
        assert UserProfile.get_collection_name() == "user_profiles"

    def test_get_collection_name_pluralization(self):
        """Test proper pluralization of collection names."""
        class Category(Document):
            pass
        
        assert Category.get_collection_name() == "categories"

    def test_get_collection_name_ends_with_y(self):
        """Test pluralization for words ending with 'y'."""
        class Story(Document):
            pass
        
        assert Story.get_collection_name() == "stories"

    def test_get_collection_name_ends_with_s(self):
        """Test pluralization for words ending with 's'."""
        class Box(Document):
            pass
        
        assert Box.get_collection_name() == "boxes"
