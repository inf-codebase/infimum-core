from core.base.entity import Document
from core.database.mongo import MongoManagerBase
import pytest

class TestMongoManagerBase:
    """Test cases for MongoManagerBase class."""

    def test_get_collection_name_string(self):
        """Test getting collection name from string."""
        manager = MongoManagerBase("mongodb://localhost", "testdb")
        assert manager._get_collection_name("users") == "users"

    def test_get_collection_name_document_class(self):
        """Test getting collection name from Document class."""
        class User(Document):
            pass
        
        manager = MongoManagerBase("mongodb://localhost", "testdb")
        assert manager._get_collection_name(User) == "users"

    def test_get_collection_name_document_instance(self):
        """Test getting collection name from Document instance."""
        class User(Document):
            pass
        
        user = User()
        manager = MongoManagerBase("mongodb://localhost", "testdb")
        assert manager._get_collection_name(user) == "users"

    def test_get_collection_name_invalid_type(self):
        """Test getting collection name with invalid type raises error."""
        manager = MongoManagerBase("mongodb://localhost", "testdb")
        with pytest.raises(TypeError):
            manager._get_collection_name(123)
