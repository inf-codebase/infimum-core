"""
Tests for database configuration models.

This module tests the Pydantic configuration models for type safety
and validation.
"""

import pytest
from pydantic import ValidationError
from core.database.base import (
    VectorIndexConfig,
    VectorCollectionConfig,
    DatabaseConnectionConfig,
)


class TestVectorIndexConfig:
    """Test cases for VectorIndexConfig."""
    
    def test_default_config(self):
        """Test creating config with defaults."""
        config = VectorIndexConfig()
        
        assert config.metric_type == "L2"
        assert config.index_type == "IVF_FLAT"
        assert config.params == {"nlist": 1024}
    
    def test_custom_config(self):
        """Test creating config with custom values."""
        config = VectorIndexConfig(
            metric_type="COSINE",
            index_type="HNSW",
            params={"M": 16, "ef_construction": 200}
        )
        
        assert config.metric_type == "COSINE"
        assert config.index_type == "HNSW"
        assert config.params["M"] == 16
        assert config.params["ef_construction"] == 200
    
    def test_extra_params_allowed(self):
        """Test that extra parameters are allowed."""
        config = VectorIndexConfig(
            metric_type="L2",
            custom_param="custom_value"
        )
        
        assert config.custom_param == "custom_value"


class TestVectorCollectionConfig:
    """Test cases for VectorCollectionConfig."""
    
    def test_default_config(self):
        """Test creating config with defaults."""
        config = VectorCollectionConfig()
        
        assert config.vector_size == 1536
        assert config.has_vector is True
        assert config.index_config is None
    
    def test_custom_config(self):
        """Test creating config with custom values."""
        index_config = VectorIndexConfig(metric_type="COSINE")
        config = VectorCollectionConfig(
            vector_size=768,
            has_vector=True,
            index_config=index_config
        )
        
        assert config.vector_size == 768
        assert config.has_vector is True
        assert config.index_config.metric_type == "COSINE"
    
    def test_vector_size_validation(self):
        """Test that vector_size must be positive."""
        with pytest.raises(ValidationError):
            VectorCollectionConfig(vector_size=-1)
        
        with pytest.raises(ValidationError):
            VectorCollectionConfig(vector_size=0)


class TestDatabaseConnectionConfig:
    """Test cases for DatabaseConnectionConfig."""
    
    def test_minimal_config(self):
        """Test creating minimal config."""
        config = DatabaseConnectionConfig()
        
        assert config.host is None
        assert config.port is None
        assert config.database is None
    
    def test_full_config(self):
        """Test creating config with all fields."""
        config = DatabaseConnectionConfig(
            host="localhost",
            port=5432,
            database="test_db",
            user="test_user",
            password="test_pass",
            connection_string="postgresql://test_user:test_pass@localhost:5432/test_db"
        )
        
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "test_db"
        assert config.user == "test_user"
        assert config.password == "test_pass"
    
    def test_port_validation(self):
        """Test that port must be in valid range."""
        with pytest.raises(ValidationError):
            DatabaseConnectionConfig(port=-1)
        
        with pytest.raises(ValidationError):
            DatabaseConnectionConfig(port=65536)
        
        # Valid ports should work
        config1 = DatabaseConnectionConfig(port=1)
        config2 = DatabaseConnectionConfig(port=65535)
        assert config1.port == 1
        assert config2.port == 65535
    
    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed for database-specific configs."""
        config = DatabaseConnectionConfig(
            host="localhost",
            use_memory=True,
            timeout=30
        )
        
        assert config.host == "localhost"
        assert config.use_memory is True
        assert config.timeout == 30
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = DatabaseConnectionConfig(
            host="localhost",
            port=5432,
            database="test_db"
        )
        
        config_dict = config.model_dump(exclude_none=True)
        
        assert config_dict["host"] == "localhost"
        assert config_dict["port"] == 5432
        assert config_dict["database"] == "test_db"
        assert "user" not in config_dict  # Should be excluded if None
